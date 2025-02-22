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
    --prompt-template chatml 

# 保存进程ID
echo $! > /tmp/reward_model.pid

# 等待服务启动
echo "Waiting for Reward Model service to start..."
sleep 5

# 准备测试数据
TEST_DATA='{
    "messages": [
        {"from": "user", "value": "What is 1+1?"},
        {"from": "assistant", "value": "Let me solve this step by step:\n<think>\n1. This is a basic addition problem\n2. Adding 1 and 1 together\n</think>\n\nThe answer is \\boxed{2}"}
    ]
}'

# 检查服务是否成功启动
MAX_RETRIES=6
RETRY_INTERVAL=5
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    RESPONSE=$(curl -s -f "http://127.0.0.1:$REWARD_MODEL_PORT/get_reward" \
        -H "Content-Type: application/json" \
        -d "$TEST_DATA" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ ! -z "$RESPONSE" ]; then
        echo "Reward Model service started successfully on $(hostname)"
        echo "Reward endpoint: http://127.0.0.1:$REWARD_MODEL_PORT/get_reward"
        echo "PID saved to /tmp/reward_model.pid"
        exit 0
    fi
    
    echo "Waiting for service to be ready... ($(($RETRY_COUNT + 1))/$MAX_RETRIES)"
    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep $RETRY_INTERVAL
done

echo "Error: Failed to start Reward Model service after $MAX_RETRIES attempts"
if [ -f /tmp/reward_model.pid ]; then
    kill $(cat /tmp/reward_model.pid)
    rm /tmp/reward_model.pid
fi
exit 1

# 提示如何停止服务
echo -e "\nTo stop the service:"
echo "kill \$(cat /tmp/reward_model.pid)"
