#DATASET="/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json"
#DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/mathlv345_8k_chatml.json"
DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/xDAN_Agentic_openMath_r1_full.json"

MODEL_CPK_NAME="xDAN-L2-RL-32B-Instruct"
PRETRAIN_MODEL="/data/vayu/train/models/xDAN-L2-32b-Reasoning-SFT-Alignment-0216-ckp2364"
SAVE_PATH="./ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/tensorboard"

# deploy remote reward function at 127.0.0.1:5000
python -m openrlhf.models.remote_rm.math_verifier --dataset $DATASET --input_key prompt --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!

# Set NCCL environment variables for better multi-node communication
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_4,mlx5_5
IS_HEAD_NODE=true
NUM_NODES=2

# Check if this is the head node
if [ "$IS_HEAD_NODE" = "true" ]; then
    # Start Ray on head node
    ray start --head --node-ip-address 10.11.50.33 --num-gpus 8 --temp-dir /data/vayu/train/ray
    
    # Wait for other nodes to join
    echo "Waiting 60 seconds for other nodes to join..."
    for i in {60..1}; do
        echo -ne "\rTime remaining: $i seconds..."
        sleep 1
    done
    echo -e "\rAll nodes should be connected now. Starting the job..."

    # Submit the training job
    ray job submit --address="http://0.0.0.0:8265" \
       --runtime-env-json='{"working_dir": "/data/vayu/train/xDAN-RL-Training-GRPO"}' \
       -- python3 -m openrlhf.cli.train_ppo_ray \
       --ref_num_nodes 1 \
       --ref_num_gpus_per_node 4 \
       --remote_rm_url http://127.0.0.1:5000/get_reward \
       --actor_num_nodes 1 \
       --actor_num_gpus_per_node 4 \
       --vllm_num_engines 1 \
       --vllm_tensor_parallel_size 8 \
       --vllm_gpu_memory_utilization 0.5 \
       --vllm_max_model_len 4096 \
       --vllm_block_size 16 \
       --vllm_swap_space 16 \
       --vllm_max_num_batched_tokens 4096 \
       --vllm_max_num_seqs 64 \
       --vllm_enable_sleep \
       --vllm_sync_backend nccl \
       --vllm_device_map "auto" \
       --vllm_gpu_ids 0,1,2,3,4,5,6,7 \
       --colocate_all_models \
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
       --input_key prompt \
       --normalize_reward \
       --flash_attn \
       --gradient_checkpointing \
       --save_steps 200 \
       --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
       --save_hf_ckpt \
       --use_wandb $SAVE_PATH/$MODEL_CPK_NAME/logs 
else
    # Worker nodes join the cluster
    ray start --address=$HEAD_NODE_ADDRESS --num-gpus 8 --temp-dir /data/vayu/train/ray
fi

ray stop
kill $childpid