#!/bin/bash

# 帮助信息
show_help() {
    echo "Usage: $0 [--head HOSTNAME | --worker MASTER_HOSTNAME]"
    echo
    echo "Options:"
    echo "  --head HOSTNAME          Start a head node on the specified hostname"
    echo "  --worker MASTER_HOSTNAME Start a worker node connecting to the specified master hostname"
    echo
    echo "Examples:"
    echo "  $0 --head gpu005         # Start head node on gpu005"
    echo "  $0 --worker gpu005       # Start worker node connecting to gpu005"
    exit 1
}

# 参数解析
if [ "$#" -ne 2 ]; then
    show_help
fi

MODE=""
TARGET_HOST=""

case "$1" in
    --head)
        MODE="head"
        TARGET_HOST="$2"
        ;;
    --worker)
        MODE="worker"
        TARGET_HOST="$2"
        ;;
    *)
        show_help
        ;;
esac

# 基础配置
RAY_PASSWORD="xdan_user_ray_2024"  # 简单固定密码
RAY_GCS_PORT=8100                  # GCS端口(避免默认6379)
RAY_DASHBOARD_PORT=8101            # Dashboard端口(避免默认8265)
RAY_CLIENT_PORT=8102               # Client端口(避免默认10001)
REWARD_MODEL_PORT=5001             # RM端口(避免默认5000)

# 确保目录存在并设置权限
mkdir -p /data/vayu/train/ray/vayu_cluster/storage
chmod 700 /data/vayu/train/ray/vayu_cluster

# 根据模式启动对应节点
if [ "$MODE" = "head" ]; then
    if [ "$(hostname)" != "$TARGET_HOST" ]; then
        echo "Error: Current hostname ($(hostname)) does not match specified head node ($TARGET_HOST)"
        exit 1
    fi
    
    echo "Starting Ray head node on $TARGET_HOST..."
    ray start --head \
        --port=$RAY_GCS_PORT \
        --dashboard-port=$RAY_DASHBOARD_PORT \
        --ray-client-server-port=$RAY_CLIENT_PORT \
        --temp-dir=/data/vayu/train/ray/vayu_cluster \
        --storage=/data/vayu/train/ray/vayu_cluster/storage \
        --dashboard-host=0.0.0.0 \
        --num-gpus=8 \
        --disable-usage-stats \
        --object-store-memory=100000000000 \
        --redis-password="$RAY_PASSWORD"
        
    echo "Head node started successfully"
    echo "Dashboard URL: http://$TARGET_HOST:$RAY_DASHBOARD_PORT"
    
elif [ "$MODE" = "worker" ]; then
    echo "Starting Ray worker node connecting to $TARGET_HOST..."
    ray start \
        --address=$TARGET_HOST:$RAY_GCS_PORT \
        --temp-dir=/data/vayu/train/ray/vayu_cluster \
        --storage=/data/vayu/train/ray/vayu_cluster/storage \
        --num-gpus=8 \
        --redis-password="$RAY_PASSWORD"
        
    echo "Worker node started successfully"
fi
