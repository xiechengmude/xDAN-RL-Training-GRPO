DATASET="/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json"
MODEL_CPK_NAME="xDAN-L2-RL-32B-Instruct"
PRETRAIN_MODEL="/data/vayu/train/models/xDAN-L2-Qwen25-32b-Instruct"
SAVE_PATH="./ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/tensorboard"

# deploy remote reward function at 127.0.0.1:5000
python -m openrlhf.models.remote_rm.math_verifier --dataset $DATASET --input_key message --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!

# 清理现有Ray进程和缓存
cleanup_ray() {
    echo "Cleaning up Ray..."
    ray stop
    pkill -9 ray
    rm -rf /tmp/ray/*
    rm -rf ~/.cache/ray/*
    rm -rf /data/vayu/train/ray/*
}

# 等待Ray集群就绪
wait_for_cluster() {
    local max_attempts=30
    local attempt=1
    local head_ip=$1
    
    echo "Waiting for Ray cluster to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if ray status --address="$head_ip:6379" 2>/dev/null | grep -q "1 node(s) in total"; then
            if [ "$2" = "head" ]; then
                echo "Head node is ready."
                return 0
            fi
        fi
        
        if [ "$2" = "worker" ] && ray status --address="$head_ip:6379" 2>/dev/null | grep -q "2 node(s) in total"; then
            echo "Worker node has joined the cluster."
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: Waiting for cluster... (5s)"
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo "Timeout waiting for Ray cluster!"
    return 1
}

# 根据参数启动Ray节点
if [ "$1" == "head" ]; then
    # 清理
    cleanup_ray
    
    # 启动reward model
    echo "Starting reward model..."
    python -m openrlhf.models.remote_rm.math_verifier --dataset $DATASET --input_key message --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
    childpid=$!
    
    # 等待reward model启动
    echo "Waiting for reward model to start..."
    sleep 10
    
    # 启动head节点
    echo "Starting Ray head node..."
    ray start --head \
        --node-ip-address=$2 \
        --port=6379 \
        --redis-password="123456" \
        --num-gpus=8 \
        --dashboard-port=8265 \
        --temp-dir=/data/vayu/train/ray

    # 等待head节点就绪
    if ! wait_for_cluster $2 "head"; then
        echo "Failed to start head node!"
        cleanup_ray
        exit 1
    fi

    # 提交训练任务
    echo "Submitting training job..."
    ray job submit --address="http://$2:8265" \
       --runtime-env-json='{"working_dir": "/data/vayu/train/xDAN-RL-Training-GRPO"}' \
       -- python3 -m openrlhf.cli.train_ppo_ray \
       --ref_num_nodes 2 \
       --ref_num_gpus_per_node 4 \
       --remote_rm_url http://127.0.0.1:5000/get_reward \
       --actor_num_nodes 2 \
       --actor_num_gpus_per_node 4 \
       --vllm_num_engines 2 \
       --vllm_tensor_parallel_size 4 \
       --colocate_all_models \
       --vllm_enable_sleep \
       --vllm_gpu_memory_utilization 0.85 \
       --vllm_sync_backend gloo \
       --enable_prefix_caching \
       --pretrain $PRETRAIN_MODEL \
       --save_path $SAVE_PATH/$MODEL_CPK_NAME \
       --micro_train_batch_size 2 \
       --train_batch_size 128 \
       --micro_rollout_batch_size 4 \
       --rollout_batch_size 256 \
       --temperature 1 \
       --n_samples_per_prompt 8 \
       --max_epochs 1 \
       --num_episodes 30 \
       --prompt_max_len 4096 \
       --max_samples 100000 \
       --generate_max_len 4096 \
       --advantage_estimator rloo \
       --zero_stage 3 \
       --adam_offload \
       --bf16 \
       --actor_learning_rate 5e-7 \
       --init_kl_coef 0.02 \
       --prompt_data $DATASET \
       --input_key message \
       --normalize_reward \
       --flash_attn \
       --gradient_checkpointing \
       --save_steps 10 \
       --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
       --save_hf_ckpt \
       --use_wandb $SAVE_PATH/$MODEL_CPK_NAME/logs 

elif [ "$1" == "worker" ]; then
    # 清理
    cleanup_ray
    
    # 启动worker节点
    echo "Starting Ray worker node..."
    ray start \
        --address=$2:6379 \
        --redis-password="123456" \
        --num-gpus=8 \
        --temp-dir=/data/vayu/train/ray

    # 等待worker节点就绪
    if ! wait_for_cluster $2 "worker"; then
        echo "Failed to start worker node!"
        cleanup_ray
        exit 1
    fi
else
    echo "Usage: $0 [head|worker] IP_ADDRESS"
    echo "Examples:"
    echo "  Start head node:   $0 head 192.168.1.100"
    echo "  Start worker node: $0 worker 192.168.1.100"
    exit 1
fi

# 等待清理
trap 'cleanup_ray' EXIT

ray stop
kill $childpid