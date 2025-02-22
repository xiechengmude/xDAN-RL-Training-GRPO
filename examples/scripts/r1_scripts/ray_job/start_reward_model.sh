#!/bin/bash

# 基础配置
REWARD_MODEL_PORT=5001             # 避免默认5000端口
DATASET="/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json"
SAVE_PATH="/data/vayu/train/tmp"
MODEL_CPK_NAME="reward_model"

# 检查数据集是否存在
if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset not found at $DATASET"
    exit 1
fi

# 检查端口是否已被占用
if netstat -tuln | grep -q ":$REWARD_MODEL_PORT "; then
    echo "Error: Port $REWARD_MODEL_PORT is already in use"
    exit 1
fi

echo "Starting Reward Model service on $(hostname):$REWARD_MODEL_PORT..."

# 启动Reward Model服务
nohup python -m openrlhf.models.remote_rm.math_verifier \
    --dataset $DATASET \
    --input_key message \
    --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &

# 保存进程ID
echo $! > "${SAVE_PATH}/${MODEL_CPK_NAME}/reward_model.pid"

echo "Reward Model started. PID saved to ${SAVE_PATH}/${MODEL_CPK_NAME}/reward_model.pid"
echo "Waiting 10 seconds for service to initialize..."
sleep 10

# 验证服务健康状态
if ! curl -s http://127.0.0.1:$REWARD_MODEL_PORT/health > /dev/null; then
    echo "Failed to start reward model service"
    exit 1
fi
echo "Reward model service is healthy"

# 使进程与终端完全分离
disown $(cat "${SAVE_PATH}/${MODEL_CPK_NAME}/reward_model.pid")

# 提示如何停止服务
echo -e "\nTo stop the service:"
echo "kill \$(cat ${SAVE_PATH}/${MODEL_CPK_NAME}/reward_model.pid)"
