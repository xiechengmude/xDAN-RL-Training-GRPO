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

# 检查服务是否正常运行
if curl -s http://127.0.0.1:5000/health_check > /dev/null; then
    echo "Reward model service is running."
else
    echo "Warning: Reward model service might not be running properly."
fi
