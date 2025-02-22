#!/bin/bash

# 清理当前用户的Ray进程和缓存
cleanup_ray() {
    echo "Cleaning up Ray..."
    ps aux | grep "ray:::" | grep "$(whoami)" | grep -v grep | awk '{print $2}' | xargs -r kill -9
    rm -rf /tmp/ray/*
    rm -rf ~/.cache/ray/*
    rm -rf /data/vayu/train/ray/vayu_cluster/*
}

# 创建并设置安全的集群目录
setup_cluster_dir() {
    echo "Setting up cluster directory..."
    mkdir -p /data/vayu/train/ray/vayu_cluster/{storage,spill}
    chmod 700 /data/vayu/train/ray/vayu_cluster
    chmod 700 /data/vayu/train/ray/vayu_cluster/{storage,spill}
}

# 生成和保存随机密码
generate_password() {
    local password=$(openssl rand -hex 16)
    echo "$password" > /data/vayu/train/ray/vayu_cluster/.ray_password
    chmod 600 /data/vayu/train/ray/vayu_cluster/.ray_password
    echo "$password"
}

# 启动head节点
start_head() {
    local head_ip=$1
    echo "Starting Ray head node at $head_ip"
    cleanup_ray
    setup_cluster_dir
    
    local ray_password=$(generate_password)
    echo "Generated Ray cluster password: $ray_password"
    
    ray start --head \
        --node-ip-address=$head_ip \
        --port=8100 \
        --dashboard-port=8101 \
        --ray-client-server-port=8102 \
        --temp-dir=/data/vayu/train/ray/vayu_cluster \
        --storage=/data/vayu/train/ray/vayu_cluster/storage \
        --num-gpus=8 \
        --dashboard-host=0.0.0.0 \
        --disable-usage-stats \
        --system-config='{"automatic_object_spilling_enabled":true,"object_spilling_config":{"type":"filesystem","params":{"directory_path":"/data/vayu/train/ray/vayu_cluster/spill"}}}' \
        --redis-password="$ray_password"
}

# 启动worker节点
start_worker() {
    local head_ip=$1
    echo "Starting Ray worker node, connecting to $head_ip"
    cleanup_ray
    setup_cluster_dir
    
    # 读取Ray密码
    if [ ! -f "/data/vayu/train/ray/vayu_cluster/.ray_password" ]; then
        echo "Error: Ray password file not found. Please ensure head node is running."
        exit 1
    fi
    local ray_password=$(cat /data/vayu/train/ray/vayu_cluster/.ray_password)
    
    ray start \
        --address=$head_ip:8100 \
        --temp-dir=/data/vayu/train/ray/vayu_cluster \
        --storage=/data/vayu/train/ray/vayu_cluster/storage \
        --num-gpus=8 \
        --redis-password="$ray_password"
}

# 检查Ray集群状态
check_cluster() {
    echo "Checking Ray cluster status..."
    if [ ! -f "/data/vayu/train/ray/vayu_cluster/.ray_password" ]; then
        echo "Error: Ray password file not found. Please ensure head node is running."
        exit 1
    fi
    local ray_password=$(cat /data/vayu/train/ray/vayu_cluster/.ray_password)
    ray status --address=$1:8100 --redis-password="$ray_password"
}

# 使用说明
usage() {
    echo "Usage: $0 [head|worker] IP_ADDRESS"
    echo "Examples:"
    echo "  Start head node:   $0 head gpu005"
    echo "  Start worker node: $0 worker gpu005"
    exit 1
}

# 主函数
main() {
    if [ $# -ne 2 ]; then
        usage
    fi

    local mode=$1
    local ip_address=$2

    case $mode in
        head)
            start_head "$ip_address"
            ;;
        worker)
            start_worker "$ip_address"
            ;;
        *)
            usage
            ;;
    esac
}

main "$@"
