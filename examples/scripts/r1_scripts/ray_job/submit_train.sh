#!/bin/bash

# 基础配置
RAY_DASHBOARD_PORT=8101            # Dashboard端口
REWARD_MODEL_PORT=5001             # RM端口
DATASET="/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json"
MODEL_NAME="xDAN-L2-RL-32B-Instruct"
PRETRAIN="/data/vayu/train/models/xDAN-L2-Qwen25-32b-Instruct"

# 检查是否在gpu005上运行
if [ "$(hostname)" != "gpu005" ]; then
    echo "Error: This script must be run on gpu005"
    exit 1
fi

# 训练参数配置
REF_NUM_NODES=2                    # 参考模型节点数
REF_GPUS_PER_NODE=4               # 每节点参考模型GPU数
ACTOR_NUM_NODES=2                  # Actor模型节点数
ACTOR_GPUS_PER_NODE=4             # 每节点Actor模型GPU数
VLLM_NUM_ENGINES=2                # vLLM引擎数
VLLM_TP_SIZE=4                    # 张量并行大小
VLLM_MEM_UTIL=0.85               # GPU内存利用率

# 批处理参数
MICRO_TRAIN_BATCH=2              # 微训练批次大小
TRAIN_BATCH=128                  # 训练批次大小
MICRO_ROLLOUT_BATCH=4           # 微滚动批次大小
ROLLOUT_BATCH=256               # 滚动批次大小

# 训练控制参数
MAX_EPOCHS=1                     # 最大训练轮数
NUM_EPISODES=30                  # 训练回合数
PROMPT_MAX_LEN=4096             # 提示最大长度
MAX_SAMPLES=100000              # 最大样本数
GEN_MAX_LEN=4096               # 生成最大长度
TEMPERATURE=1                   # 采样温度
N_SAMPLES_PER_PROMPT=8         # 每个提示的采样数

# 优化器参数
ACTOR_LR=5e-7                  # Actor学习率
INIT_KL_COEF=0.02             # 初始KL系数

# 提交训练任务
echo "Submitting training job..."
RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
    --working-dir="." \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes $REF_NUM_NODES \
    --ref_num_gpus_per_node $REF_GPUS_PER_NODE \
    --remote_rm_url "http://127.0.0.1:$REWARD_MODEL_PORT/get_reward" \
    --actor_num_nodes $ACTOR_NUM_NODES \
    --actor_num_gpus_per_node $ACTOR_GPUS_PER_NODE \
    --vllm_num_engines $VLLM_NUM_ENGINES \
    --vllm_tensor_parallel_size $VLLM_TP_SIZE \
    --colocate_all_models \
    --vllm_enable_sleep \
    --vllm_gpu_memory_utilization $VLLM_MEM_UTIL \
    --vllm_sync_backend gloo \
    --enable_prefix_caching \
    --pretrain $PRETRAIN \
    --save_path "./ckpts/$MODEL_NAME" \
    --micro_train_batch_size $MICRO_TRAIN_BATCH \
    --train_batch_size $TRAIN_BATCH \
    --micro_rollout_batch_size $MICRO_ROLLOUT_BATCH \
    --rollout_batch_size $ROLLOUT_BATCH \
    --temperature $TEMPERATURE \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --max_epochs $MAX_EPOCHS \
    --num_episodes $NUM_EPISODES \
    --prompt_max_len $PROMPT_MAX_LEN \
    --max_samples $MAX_SAMPLES \
    --generate_max_len $GEN_MAX_LEN \
    --advantage_estimator rloo \
    --zero_stage 3 \
    --adam_offload \
    --bf16 \
    --actor_learning_rate $ACTOR_LR \
    --init_kl_coef $INIT_KL_COEF \
    --prompt_data $DATASET \
    --input_key message \
    --normalize_reward \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 10 \
    --ckpt_path "./ckpts/$MODEL_NAME/ckpt" \
    --save_hf_ckpt

echo "Training job submitted successfully"
echo "You can monitor the training progress at: http://gpu005:$RAY_DASHBOARD_PORT"
echo "Check logs with: tail -f ./ckpts/$MODEL_NAME/logs/*"
