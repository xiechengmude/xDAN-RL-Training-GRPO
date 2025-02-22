#!/bin/bash

# 基础配置
REWARD_MODEL_PORT=5001             # 避免默认5000端口
DATASET="/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json"

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
python -m openrlhf.models.remote_rm.math_verifier \
    --dataset $DATASET \
    --input_key message \
    --prompt-template chatml \
    --port $REWARD_MODEL_PORT &

# 保存进程ID
echo $! > /tmp/reward_model.pid

# 等待服务启动
echo "Waiting for Reward Model service to start..."
sleep 5

# 检查服务是否成功启动
if curl -s http://127.0.0.1:$REWARD_MODEL_PORT/health_check > /dev/null; then
    echo "Reward Model service started successfully on $(hostname)"
    echo "Health check endpoint: http://127.0.0.1:$REWARD_MODEL_PORT/health_check"
    echo "Reward endpoint: http://127.0.0.1:$REWARD_MODEL_PORT/get_reward"
    echo "PID saved to /tmp/reward_model.pid"
else
    echo "Error: Failed to start Reward Model service"
    if [ -f /tmp/reward_model.pid ]; then
        kill $(cat /tmp/reward_model.pid)
        rm /tmp/reward_model.pid
    fi
    exit 1
fi

# 提示如何停止服务
echo -e "\nTo stop the service:"
echo "kill \$(cat /tmp/reward_model.pid)"
