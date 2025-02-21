#!/bin/bash

DATASET="/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json"
MODEL_CPK_NAME="xDAN-L2-RL-32B-Instruct"
PRETRAIN_MODEL="/data/vayu/train/models/xDAN-L2-Qwen25-32b-Instruct"
SAVE_PATH="./ckpts"

if [ $# -ne 1 ]; then
    echo "Usage: $0 HEAD_NODE_IP"
    echo "Example: $0 gpu005"
    exit 1
fi

HEAD_IP=$1

# 检查Ray集群状态
echo "Checking Ray cluster status..."
if ! ray status --address="$HEAD_IP:8100" 2>/dev/null | grep -q "2 node(s) in total"; then
    echo "Error: Ray cluster is not ready. Make sure both nodes are running."
    exit 1
fi

# 检查Reward Model服务
echo "Checking reward model service..."
if ! curl -s http://127.0.0.1:5000/health_check > /dev/null; then
    echo "Error: Reward model service is not running."
    exit 1
fi

# 提交训练任务
echo "Submitting training job..."
ray job submit --address="http://$HEAD_IP:8101" \
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

echo "Training job submitted. Check Ray dashboard at http://$HEAD_IP:8101"
