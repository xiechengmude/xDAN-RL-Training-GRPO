#!/bin/bash

# 清理Ray进程和缓存
cleanup_ray() {
    echo "Cleaning up Ray..."
    ray stop
    pkill -9 ray
    rm -rf /tmp/ray/*
    rm -rf ~/.cache/ray/*
    rm -rf /data/vayu/train/ray/*
}

# 等待Ray集群就绪
wait_for_cluster() {
    local max_attempts=30
    local attempt=1
    local head_ip=$1
    
    echo "Waiting for Ray cluster to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if [ "$2" = "head" ]; then
            if ray status --address="$head_ip:6379" 2>/dev/null | grep -q "1 node(s) in total"; then
                echo "Head node is ready."
                return 0
            fi
        elif [ "$2" = "worker" ]; then
            if ray status --address="$head_ip:6379" 2>/dev/null | grep -q "2 node(s) in total"; then
                echo "Worker node has joined the cluster."
                return 0
            fi
        fi
        
        echo "Attempt $attempt/$max_attempts: Waiting for cluster... (5s)"
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo "Timeout waiting for Ray cluster!"
    return 1
}

# 主逻辑
if [ $# -ne 2 ]; then
    echo "Usage: $0 [head|worker] HEAD_NODE_IP"
    echo "Examples:"
    echo "  Start head node:   $0 head gpu005"
    echo "  Start worker node: $0 worker gpu005"
    exit 1
fi

MODE=$1
HEAD_IP=$2

# 清理现有Ray进程
cleanup_ray

if [ "$MODE" = "head" ]; then
    echo "Starting Ray head node..."
    ray start --head \
        --node-ip-address=$HEAD_IP \
        --port=6379 \
        --redis-password="123456" \
        --num-gpus=8 \
        --dashboard-port=8265 \
        --temp-dir=/data/vayu/train/ray

    if ! wait_for_cluster $HEAD_IP "head"; then
        echo "Failed to start head node!"
        cleanup_ray
        exit 1
    fi
    
    echo "Ray head node is ready at $HEAD_IP"
    echo "Dashboard available at http://$HEAD_IP:8265"

elif [ "$MODE" = "worker" ]; then
    echo "Starting Ray worker node..."
    ray start \
        --address=$HEAD_IP:6379 \
        --redis-password="123456" \
        --num-gpus=8 \
        --temp-dir=/data/vayu/train/ray

    if ! wait_for_cluster $HEAD_IP "worker"; then
        echo "Failed to join cluster!"
        cleanup_ray
        exit 1
    fi
    
    echo "Successfully joined Ray cluster"
fi

# 显示集群状态
echo "Current cluster status:"
ray status
