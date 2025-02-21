#!/bin/bash

# 清理现有Ray进程和缓存
cleanup_ray() {
    echo "Cleaning up Ray..."
    ray stop
    pkill -9 ray
    rm -rf /tmp/ray/*
    rm -rf ~/.cache/ray/*
    rm -rf /data/vayu/train/ray/*
}

# 启动head节点
start_head() {
    local head_ip=$1
    echo "Starting Ray head node at $head_ip"
    cleanup_ray
    ray start --head \
        --node-ip-address=$head_ip \
        --port=6379 \
        --redis-password="123456" \
        --num-gpus=8 \
        --dashboard-port=8265 \
        --temp-dir=/data/vayu/train/ray
}

# 启动worker节点
start_worker() {
    local head_ip=$1
    echo "Starting Ray worker node, connecting to $head_ip"
    cleanup_ray
    ray start \
        --address=$head_ip:6379 \
        --redis-password="123456" \
        --num-gpus=8 \
        --temp-dir=/data/vayu/train/ray
}

# 检查Ray集群状态
check_cluster() {
    echo "Checking Ray cluster status..."
    ray status
}

# 使用说明
usage() {
    echo "Usage: $0 [head|worker] IP_ADDRESS"
    echo "Examples:"
    echo "  Start head node:   $0 head gpu005"
    echo "  Start worker node: $0 worker gpu005"
    exit 1
}

# 主逻辑
main() {
    if [ $# -ne 2 ]; then
        usage
    fi

    local mode=$1
    local ip=$2

    case $mode in
        head)
            start_head $ip
            ;;
        worker)
            start_worker $ip
            ;;
        *)
            usage
            ;;
    esac

    # 等待Ray启动
    sleep 5
    check_cluster
}

main "$@"
