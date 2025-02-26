set -x
DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/xDAN_Agentic_openMath_r1_full.json"

MODEL_CPK_NAME="xDAN-L2-RL-32B-Instruct"
PRETRAIN_MODEL="/data/vayu/train/models/xDAN-L2-32b-Reasoning-SFT-Alignment-0216-ckp2364"
SAVE_PATH="./ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/tensorboard"

# deploy remote reward function at 127.0.0.1:5000
python -m openrlhf.models.remote_rm.math_verifier --dataset $DATASET --input_key prompt --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!


ray job submit --address="http://0.0.0.0:8265" \
   --runtime-env-json='{"working_dir": "data/vayu/train/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 4 \
   --remote_rm_url http://localhost:5000/get_reward \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.5 \
   --advantage_estimator reinforce_baseline \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --micro_train_batch_size 4 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 1e-4 \
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
   --save_steps 200 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --use_wandb $SAVE_PATH/$MODEL_CPK_NAME/logs 
# You could also try
#   --use_kl_loss \
#   --use_kl_estimator_k3 \

# also supports --advantage_estimator rloo | reinforce_baseline