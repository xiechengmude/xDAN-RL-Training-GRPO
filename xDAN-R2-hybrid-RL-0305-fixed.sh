#!/bin/bash
set -x

# 设置NCCL环境变量以提高稳定性
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_SOCKET_IFNAME=eth0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL

# 数据集路径
DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/xDAN-level5-math-aime-chatml.json"

MODEL_CPK_NAME="xDAN-L2-RL-32B-Instruct-RL-0305-fixed"
PRETRAIN_MODEL="/data/vayu/train/models/xDAN-L2-32b-Reasoning-SFT-Alignment-0216-ckp2364"

SAVE_PATH="./ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/tensorboard"

# 部署远程奖励函数
python -m openrlhf.models.remote_rm.math_verifier --dataset $DATASET --input_key prompt --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!

# 等待奖励模型启动
sleep 10

# 提交Ray任务
ray job submit --address="http://0.0.0.0:8265" \
   --runtime-env-json='{"working_dir": "/data/vayu/train/xDAN-RL-Training-GRPO", "env_vars": {"MASTER_ADDR": "10.11.50.36", "MASTER_PORT": "24999", "NCCL_DEBUG": "INFO", "NCCL_IB_TIMEOUT": "23", "NCCL_IB_RETRY_CNT": "7", "NCCL_ASYNC_ERROR_HANDLING": "1", "NCCL_P2P_LEVEL": "NVL"}}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 4 \
   --remote_rm_url http://localhost:5000/get_reward \
   --vllm_gpu_memory_utilization 0.7 \
   --advantage_estimator rloo \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 16 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 64 \
   --n_samples_per_prompt 4 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 50000 \
   --generate_max_len 8192 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.02 \
   --prompt_data $DATASET \
   --input_key prompt \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend gloo \
   --enforce_eager \
   --vllm_enable_sleep \
   --save_steps 4 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --trust_remote_code \
   --model_name_or_path $PRETRAIN_MODEL \
   --use_wandb 1b2653c58df0ccf5b38f3ffa1bf21b78d48fd620

# 清理奖励模型进程
kill $childpid
