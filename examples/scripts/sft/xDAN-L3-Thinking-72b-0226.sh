#!/bin/bash

DATASET="xDAN2099/xDAN-Thinking-dolphin-chatcn-math"
MODEL_CPK_NAME="xDAN-L3-RL-72B-Alignment-Thinking-0226"
PRETRAIN_MODEL="/data/vayu/train/models/xDAN-L3-Qwen25-72B-Pretrain"
SAVE_PATH="/data/vayu/train/models/thinking/ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/tensorboard"

# ray start --head \
#     --num-gpus 8 \
#     --resources='{"head_node": 1}' \
#     --node-ip-address 10.11.50.33 \
#     --port=6379 \
#     --temp-dir /data/vayu/train/ray \
#     --object-store-memory=100000000000

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/vayu/train/xDAN-RL-Training-GRPO"}' \
   -- python3 -m openrlhf.cli.train_sft \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --max_len 32768 \
   --dataset Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT \
   --input_key instruction \
   --output_key output \
   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --max_samples 500000 \
   --save_steps 200 \
   --save_hf_ckpt \
   --use_wandb  \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --adam_offload \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --packing_samples \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing

    
 