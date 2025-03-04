#!/bin/bash
set -x
DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/xDAN-level5-math-aime-chatml.json"

#ray start --head --temp-dir /data/vayu/train/ray
MODEL_CPK_NAME="xDAN-L2-RL-32B-Instruct-0219-RL-level5-math-0304"
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

ray job submit --address="http://0.0.0.0:8265" \
   --runtime-env-json='{"working_dir": "/data/vayu/train/xDAN-RL-Training-GRPO", "env_vars": {"MASTER_ADDR": "10.11.50.32", "MASTER_PORT": "24999"}}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 8 \
   --remote_rm_url http://localhost:5000/get_reward \
   --vllm_gpu_memory_utilization 0.7 \
   --advantage_estimator rloo \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 8 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 32 \
   --n_samples_per_prompt 16 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --max_samples 50000 \
   --generate_max_len 8192 \
   --zero_stage 3 \
   --actor_learning_rate 3e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.02 \
   --prompt_data $DATASET \
   --input_key prompt \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --flash_attn \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --save_steps 10 \
   --use_wandb $SAVE_PATH/$MODEL_CPK_NAME/logs \
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