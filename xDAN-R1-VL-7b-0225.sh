set -x

# 确保reward model服务器在正确的地址上运行
export REWARD_MODEL_HOST="0.0.0.0"
export REWARD_MODEL_PORT="5000"

DATASET="/data/vayu/train/datasets/math/xDAN_Agentic_openMath_r1_full.json"
MODEL_CPK_NAME="xDAN-R1-VL-7b-0225"
PRETRAIN_MODEL="/data/vayu/train/models/Qwen2.5-VL-7B-Instruct"
SAVE_PATH="./ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/tensorboard"

# 确保reward model服务器正确启动并等待它准备就绪
python -m openrlhf.models.remote_rm.math_verifier \
    --dataset $DATASET \
    --input_key prompt \
    --prompt-template chatml \
    --host $REWARD_MODEL_HOST \
    --port $REWARD_MODEL_PORT > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &

# 等待reward model服务器启动
sleep 30

# 测试reward model服务器是否可用
curl -X POST "http://${REWARD_MODEL_HOST}:${REWARD_MODEL_PORT}/get_reward" -H "Content-Type: application/json" -d '{"query": "test", "prompts": ["test"]}' || {
    echo "Reward model server is not responding. Please check the logs at ${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log"
    exit 1
}

ray job submit --address="http://0.0.0.0:8265" \
   --runtime-env-json='{
       "working_dir": "/data/vayu/train/xDAN-RL-Training-GRPO",
       "excludes": [
           "**/transformers/.git/objects/pack/*.pack",
           "**/.git",
           "**/__pycache__",
           "**/*.pyc"
       ]
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 2 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 2 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 4 \
   --remote_rm_url "http://${REWARD_MODEL_HOST}:${REWARD_MODEL_PORT}/get_reward" \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.5 \
   --advantage_estimator reinforce_baseline \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --micro_train_batch_size 2 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 4 \
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
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --save_steps 200 \
   --save_hf_ckpt \
   --use_wandb $SAVE_PATH/$MODEL_CPK_NAME/logs