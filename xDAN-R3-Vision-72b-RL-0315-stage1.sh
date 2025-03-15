#!/bin/bash
set -x
# wget https://raw.githubusercontent.com/TideDra/lmm-r1/refs/heads/main/examples/data/mathlv345_8k_chatml.json
#DATASET="examples/data/mathlv345_8k_chatml_1p.json"
#DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/xDAN-level5-math-aime-chatml.json"
PRETRAIN_MODEL="/data/vayu/train/models/xDAN-L3-VL-72b-RL-Base"
DATASET_PATH="/data/vayu/train/datasets/deepscaler/deepscaler_message.jsonl" 

export WORKSPACE_DIR="$(pwd)"  
# Wandb configuration (optional)
export WANDB_DIR="${WORKSPACE_DIR}"                # Directory for wandb files
export WANDB_API_KEY="1b2653c58df0ccf5b38f3ffa1bf21b78d48fd620"          # Your wandb API key (if online)
# Get script PID and setup directories
SCRIPT_PID=$$
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
export CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"
 


MODEL_CPK_NAME="xDAN-V3-RL-72B-0315"
SAVE_PATH="/data/vayu/train/models/ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/logs"


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/vayu/train/xDAN-RL-Training-GRPO", "env_vars": {"MASTER_ADDR": "10.11.50.36", "MASTER_PORT": "24999"}}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 2 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 8 \
   --vllm_gpu_memory_utilization 0.85 \
   --vllm_sync_backend nccl \
   --enable_prefix_caching \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --micro_train_batch_size 1 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 64 \
   --temperature 1 \
   --n_samples_per_prompt 16 \
   --max_epochs 2 \
   --num_episodes 2 \
   --prompt_max_len 4096 \
   --max_samples 100000 \
   --generate_max_len 4096 \
   --advantage_estimator reinforce_baseline \
   --zero_stage 3 \
   --adam_offload \
   --bf16 \
   --actor_learning_rate 4e-7 \
   --init_kl_coef 0.001 \
   --prompt_data ${DATASET_PATH} \
   --input_key message \
   --normalize_reward \
   --lambd 1 \
   --gamma 1 \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 5 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --load_checkpoint \
   --save_hf_ckpt \
   --use_wandb 1b2653c58df0ccf5b38f3ffa1bf21b78d48fd620 \
   --wandb_run_name ${MODEL_NAME} \
   --wandb_group "xDAN-R3-Vision-RL-training-0315" \

   # for visual dataset
   # --train_vlm