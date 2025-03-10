#!/bin/bash

# Set CUDA related environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG_SUBSYS=ALL

DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/mathlv345_8k_chatml.json"
MODEL_CPK_NAME="xDAN-L2-RL-32B-Alignment-Instruct-nodes"
PRETRAIN_MODEL="/data/vayu/train/eval/models/xDAN-L2-Thinking-Alignment-0216-ckp2364"
SAVE_PATH="/data/vayu/train/models/rlhf/ckps"

mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"

# Clean up existing processes
pkill -f -9 "openrlhf.models.remote_rm.math_verifier"
pkill -f "ray"
sleep 20

# Start reward model server
python -m openrlhf.models.remote_rm.math_verifier --dataset $DATASET --input_key prompt --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!

# Start Ray on head node
ray start --head \
    --node-ip-address 0.0.0.0 \
    --port 6379 \
    --num-gpus 8 \
    --temp-dir /data/vayu/train/ray

echo "Head node started. Please start worker node with:"
echo "ray start --address='10.11.50.33:6379' --num-gpus 8 --temp-dir /data/vayu/train/ray"

sleep 20

# Submit training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/vayu/train/xDAN-RL-Training-GRPO"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 4 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.25 \
   --vllm_max_num_batched_tokens 1024 \
   --vllm_max_num_seqs 64 \
   --vllm_sync_backend nccl \
   --enable_prefix_caching \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --micro_train_batch_size 2 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 128 \
   --temperature 1 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --num_episodes 30 \
   --prompt_max_len 2048 \
   --max_samples 100000 \
   --generate_max_len 3000 \
   --advantage_estimator rloo \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0.0 \
   --prompt_data $DATASET \
   --input_key prompt \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 10 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --use_tensorboard $SAVE_PATH/$MODEL_CPK_NAME/logs

# Stop Ray on all nodes
ray stop
