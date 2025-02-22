#!/bin/bash

DATASET="/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json"
MODEL_CPK_NAME="xDAN-L2-RL-32B-Instruct"
SAVE_PATH="./ckpts"

# 创建必要的目录
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/tensorboard"

# 清理已存在的reward model进程
pkill -f "openrlhf.models.remote_rm.math_verifier"

# 启动reward model服务
echo "Starting reward model..."
python -m openrlhf.models.remote_rm.math_verifier \
    --dataset $DATASET \
    --input_key message \
    --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &

# 保存进程ID
echo $! > "${SAVE_PATH}/${MODEL_CPK_NAME}/reward_model.pid"

echo "Reward model started. PID saved to ${SAVE_PATH}/${MODEL_CPK_NAME}/reward_model.pid"
echo "Waiting 10 seconds for service to initialize..."
sleep 10

# 验证服务健康状态
if ! curl -s http://localhost:5001/health > /dev/null; then
    echo "Failed to start reward model service"
    exit 1
fi
echo "Reward model service is healthy"
