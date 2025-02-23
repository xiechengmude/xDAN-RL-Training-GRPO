#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/mathlv345_8k_chatml.json"
MODEL_CPK_NAME="xDAN-L2-RL-32B-Alignment-Instruct"
PRETRAIN_MODEL="/data/vayu/train/eval/models/xDAN-L2-Thinking-Alignment-0216-ckp2364"
SAVE_PATH="/data/vayu/train/models/rlhf/ckps"

mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"

# Start reward model server
python -m openrlhf.models.remote_rm.math_verifier --dataset $DATASET --input_key prompt --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!

# Start Ray
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --temp-dir /data/vayu/train/ray

# Submit training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/vayu/train/xDAN-RL-Training-GRPO"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 4 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.35 \
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
   --prompt_max_len 1024 \
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

ray stop
