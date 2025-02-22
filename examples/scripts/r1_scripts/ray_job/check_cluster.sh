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
RESPONSE=$(curl -s -w "\n%{http_code}" http://127.0.0.1:$REWARD_MODEL_PORT/health)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n1)

if [ "$HTTP_CODE" = "200" ] && echo "$BODY" | grep -q "healthy"; then
    echo "Reward Model service is running"
else
    echo "Warning: Reward Model service is not running properly (HTTP $HTTP_CODE)"
    echo "Response: $BODY"
fi

# 检查GPU可用性
echo -e "\nChecking GPU availability..."
nvidia-smi --query-gpu=gpu_name,memory.total,memory.free,memory.used --format=csv,noheader
