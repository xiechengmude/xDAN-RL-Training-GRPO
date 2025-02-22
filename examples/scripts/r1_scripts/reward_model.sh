#!/bin/bash

# 配置参数
DATASET="/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json"
LOG_DIR="/data/vayu/train/logs"
PORT=5000

# 创建日志目录
mkdir -p $LOG_DIR

# 启动服务（默认端口5000）
nohup python -m openrlhf.models.remote_rm.math_verifier \
    --dataset $DATASET \
    --input_key message \
    --prompt-template chatml \
    --port $PORT \
    > $LOG_DIR/reward_model.log 2>&1 &

echo "Reward model started with PID: $!"
echo "Log file: $LOG_DIR/reward_model.log"
echo "Service will be available at http://127.0.0.1:$PORT"
