#!/bin/bash

DATASET="/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json"
MODEL_CPK_NAME="xDAN-L2-RL-32B-Instruct"
PRETRAIN_MODEL="/data/vayu/train/models/xDAN-L2-Qwen25-32B-Instruct"
SAVE_PATH="./ckpts"
CLUSTER_DIR="/data/vayu/train/ray/vayu_cluster"

# 创建必要的目录
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"/{logs,tensorboard}
mkdir -p "${CLUSTER_DIR}"/{storage,spill}
chmod 700 "${CLUSTER_DIR}" "${CLUSTER_DIR}"/{storage,spill}

# 生成随机密码
RAY_PASSWORD=$(openssl rand -hex 16)
echo "$RAY_PASSWORD" > "${CLUSTER_DIR}/.ray_password"
chmod 600 "${CLUSTER_DIR}/.ray_password"

# 部署reward model服务（使用非默认端口5001）
python -m openrlhf.models.remote_rm.math_verifier \
    --dataset $DATASET \
    --input_key message \
    --prompt-template chatml \
    --port 5001 \
    > "${SAVE_PATH}/${MODEL_CPK_NAME}/logs/remote_rm.log" 2>&1 &
childpid=$!

# 启动Ray head节点
ray start --head \
    --node-ip-address=0.0.0.0 \
    --port=8100 \
    --dashboard-port=8101 \
    --ray-client-server-port=8102 \
    --temp-dir="${CLUSTER_DIR}" \
    --storage="${CLUSTER_DIR}/storage" \
    --num-gpus=8 \
    --dashboard-host=0.0.0.0 \
    --disable-usage-stats \
    --system-config="{\"automatic_object_spilling_enabled\":true,\"object_spilling_config\":{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"${CLUSTER_DIR}/spill\"}}}" \
    --redis-password="$RAY_PASSWORD"

# 等待服务启动
sleep 5

# 提交训练任务
RAY_ADDRESS="http://127.0.0.1:8101" ray job submit \
    --working-dir="/data/vayu/train/xDAN-RL-Training-GRPO" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 2 \
    --ref_num_gpus_per_node 4 \
    --remote_rm_url http://127.0.0.1:5001/get_reward \
    --actor_num_nodes 2 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 4 \
    --colocate_all_models \
    --vllm_enable_sleep \
    --vllm_gpu_memory_utilization 0.95 \
    --vllm_sync_backend gloo \
    --enable_prefix_caching \
    --pretrain $PRETRAIN_MODEL \
    --save_path $SAVE_PATH/$MODEL_CPK_NAME \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 64 \
    --temperature 1 \
    --n_samples_per_prompt 4 \
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
    --save_steps 10 \
    --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
    --save_hf_ckpt \
    --use_wandb $SAVE_PATH/$MODEL_CPK_NAME/logs 

# 清理函数
cleanup() {
    echo "Cleaning up..."
    kill $childpid 2>/dev/null
    ray stop
    rm -f "${CLUSTER_DIR}/.ray_password"
}

# 设置清理钩子
trap cleanup EXIT

# 等待训练完成
wait