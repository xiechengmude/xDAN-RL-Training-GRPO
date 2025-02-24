#!/bin/bash

DATASET="/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json"
DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/mathlv345_8k_chatml.json"
DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/xDAN_Agentic_openMath_r1_full.json"

MODEL_CPK_NAME="xDAN-L2-RL-32B-Instruct"
PRETRAIN_MODEL="/data/vayu/train/models/xDAN-L2-Qwen25-32B-Instruct"
SAVE_PATH="./ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/tensorboard"

# deploy remote reward function at 127.0.0.1:5000
python -m openrlhf.models.remote_rm.math_verifier --dataset $DATASET --input_key prompt --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!

ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --temp-dir /data/vayu/train/ray

# Wait for other nodes to join
echo "Waiting 30 seconds for other nodes to join..."
for i in {30..1}; do
    echo -ne "\rTime remaining: $i seconds..."
    sleep 1
done
echo -e "\rAll nodes should be connected now. Starting the job..."

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/vayu/train/xDAN-RL-Training-GRPO"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --actor_num_nodes 1 \
   --pretrain "${PRETRAIN_MODEL}" \
   --model "qwen" \
   --strategy colossalai_zero2 \
   --dataset $DATASET \
   --save_path "${SAVE_PATH}/${MODEL_CPK_NAME}" \
   --max_epochs 1 \
   --micro_batch_size 1 \
   --batch_size 1 \
   --max_length 4096 \
   --rm_pretrain "${PRETRAIN_MODEL}" \
   --rm_model "qwen" \
   --rm_strategy colossalai_zero2 \
   --rm_dataset $DATASET \
   --rm_remote_ref_url "http://127.0.0.1:5000" \
   --rm_remote_ref_input_key "prompt" \
   --rm_remote_ref_output_key "score" \
   --rm_remote_ref_prompt_template "chatml" \
   --rm_remote_ref_batch_size 1 \
   --rm_remote_ref_max_length 4096 \
   --rm_remote_ref_max_new_tokens 512 \
   --rm_remote_ref_temperature 0.7 \
   --rm_remote_ref_top_k 50 \
   --rm_remote_ref_top_p 0.9 \
   --rm_remote_ref_do_sample True \
   --rm_remote_ref_num_return_sequences 1 \
   --exp_name "xDAN-L2-RL-32B-Instruct" \
   --seed 42 \
   --logging_steps 1

kill $childpid
