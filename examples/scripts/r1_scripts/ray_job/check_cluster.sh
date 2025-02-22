#!/bin/bash

# 基础配置
RAY_GCS_PORT=8100
RAY_DASHBOARD_PORT=8101
REWARD_MODEL_PORT=5001

# 检查Ray集群状态
echo "Checking Ray cluster status..."
ray status --address=gpu005:$RAY_GCS_PORT

# 检查Ray Dashboard是否可访问
echo -e "\nChecking Ray Dashboard..."
curl -s -o /dev/null -w "%{http_code}" http://gpu005:$RAY_DASHBOARD_PORT
if [ $? -eq 0 ]; then
    echo "Ray Dashboard is accessible at: http://gpu005:$RAY_DASHBOARD_PORT"
else
    echo "Warning: Ray Dashboard is not accessible"
fi

# 检查Reward Model服务
echo -e "\nChecking Reward Model service..."

# 1. 检查健康状态
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" http://127.0.0.1:$REWARD_MODEL_PORT/health)
HEALTH_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
HEALTH_BODY=$(echo "$HEALTH_RESPONSE" | head -n1)

if [ "$HEALTH_CODE" != "200" ] || ! echo "$HEALTH_BODY" | grep -q "healthy"; then
    echo "Warning: Reward Model service health check failed (HTTP $HEALTH_CODE)"
    echo "Response: $HEALTH_BODY"
    exit 1
fi

# 2. 验证数学验证功能
echo "Testing math verification..."
TEST_DATA='{
    "query": ["\nuser\nWhat is 1+1?\n\nassistant\nLet me solve this step by step:\n<think>\n1. This is a basic addition problem\n2. Adding 1 and 1 together\n</think>\n<answer>The answer is $2$</answer>\n"],
    "prompts": ["\nuser\nWhat is 1+1?\n\n"]
}'

VERIFY_RESPONSE=$(curl -s -w "\n%{http_code}" -H "Content-Type: application/json" -d "$TEST_DATA" http://127.0.0.1:$REWARD_MODEL_PORT/get_reward)
VERIFY_CODE=$(echo "$VERIFY_RESPONSE" | tail -n1)
VERIFY_BODY=$(echo "$VERIFY_RESPONSE" | head -n1)

if [ "$VERIFY_CODE" = "200" ] && echo "$VERIFY_BODY" | grep -q "rewards"; then
    echo "Reward Model service is running and functioning properly"
else
    echo "Warning: Reward Model service verification failed (HTTP $VERIFY_CODE)"
    echo "Response: $VERIFY_BODY"
    exit 1
fi

# 检查GPU可用性
echo -e "\nChecking GPU availability..."
nvidia-smi --query-gpu=gpu_name,memory.total,memory.free,memory.used --format=csv,noheader
