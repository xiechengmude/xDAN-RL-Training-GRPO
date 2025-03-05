#!/bin/bash
set -x

# 指定NCCL使用ibs13接口
export NCCL_SOCKET_IFNAME=ibs13

DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/xDAN-level5-math-aime-chatml.json"

#ray start --head --temp-dir /data/vayu/train/ray
MODEL_CPK_NAME="xDAN-L2-RL-32B-Instruct-0219-RL-level5-math-0305"
#PRETRAIN_MODEL="/data/vayu/train/models/xDAN-L2-32b-Reasoning-SFT-Alignment-0216-ckp2364"
#PRETRAIN_MODEL="/data/vayu/train/models/xDAN-L2-Qwen25-32B-Instruct"
PRETRAIN_MODEL="/data/vayu/train/models/xDAN-L2-Thinking-Alignment-mixed-0219"
SAVE_PATH="/data/vayu/train/models/ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/tensorboard"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/logs"

# deploy remote reward function at 127.0.0.1:5000
python -m openrlhf.models.remote_rm.math_verifier --dataset $DATASET --input_key prompt --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!
#   --colocate_actor_ref \

# Ray环境变量配置
IB_ENV_VARS='{
  "NCCL_SOCKET_IFNAME": "ibs13",
  "MASTER_ADDR": "10.11.50.36",
  "MASTER_PORT": "24999"
}'

ray job submit --address="http://0.0.0.0:8265" \
   --runtime-env-json="{\"working_dir\": \"/data/vayu/train/xDAN-RL-Training-GRPO\", \"env_vars\": ${IB_ENV_VARS}}" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 2 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 8 \
   --remote_rm_url http://localhost:5000/get_reward \
   --vllm_gpu_memory_utilization 0.75 \
   --advantage_estimator rloo \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 16 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --max_samples 50000 \
   --generate_max_len 16384 \
   --zero_stage 3 \
   --actor_learning_rate 3e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.02 \
   --prompt_data $DATASET \
   --input_key prompt \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --flash_attn \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --save_steps 4 \
   --use_wandb 1b2653c58df0ccf5b38f3ffa1bf21b78d48fd620 \
   --bf16 \
   --trust_remote_code \
   --use_kl_estimator_k3

   # 如果需要使用trust_remote_code，请取消下面这行的注释
   # --trust_remote_code

# You could also try
#   --use_kl_loss \
#   --use_kl_estimator_k3 \

# also supports --advantage_estimator rloo | reinforce_baseline
# 计数save steps
#grep -c "model.layers.63.post_attention_layernorm.weight" xDAN-RL-32b-hybrid-0301.log

# # 验证InfiniBand配置
# echo "===== 验证InfiniBand配置 ====="
# echo "NCCL配置已设置为允许使用InfiniBand"
# echo "NCCL_DEBUG=$NCCL_DEBUG"
# echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-未设置(启用)}"
# echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-未设置(启用)}"

# # 检查InfiniBand设备
# if command -v ibstat &> /dev/null; then
#     echo "===== InfiniBand设备状态 ====="
#     ibstat | grep -E "CA |State:"
# else
#     echo "未找到ibstat命令，无法检查InfiniBand设备状态"
# fi

# # 提示查看NCCL日志
# echo "===== 如何验证InfiniBand正在使用 ====="
# echo "请在训练日志中查找以下内容："
# echo "  - 'NCCL INFO NET/IB: Using [x] IB devices' (表示检测到InfiniBand设备)"
# echo "  - 'NCCL INFO NET/IB: Selected dev mlx5_x' (表示选择了InfiniBand设备)"
# echo "  - 如果看不到上述日志，可能表示InfiniBand未被使用"