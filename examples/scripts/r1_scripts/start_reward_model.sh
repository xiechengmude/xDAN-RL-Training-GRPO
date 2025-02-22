#!/bin/bash

# 配置参数
DATASET="/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json"
MODEL_NAME="xDAN-L2-RL-32B-Instruct"
SAVE_PATH="/data/vayu/train/ckpts"
PORT=5000

# 创建日志目录
LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
mkdir -p "$LOG_DIR"

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/reward_model_${TIMESTAMP}.log"

# 检查是否已有reward model在运行
check_running() {
    if pgrep -f "openrlhf.models.remote_rm.math_verifier" > /dev/null; then
        echo "Warning: Reward model is already running!"
        ps aux | grep "openrlhf.models.remote_rm.math_verifier" | grep -v grep
        read -p "Do you want to kill existing process and start new one? (y/n) " answer
        if [ "$answer" = "y" ]; then
            pkill -f "openrlhf.models.remote_rm.math_verifier"
            sleep 2
        else
            echo "Exiting..."
            exit 1
        fi
    fi
}

# 启动reward model
start_reward_model() {
    echo "Starting reward model API server..."
    echo "Log file: $LOG_FILE"
    
    python -m openrlhf.models.remote_rm.math_verifier \
        --dataset "$DATASET" \
        --input_key message \
        --prompt-template chatml \
        --port "$PORT" \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo $PID > "${LOG_DIR}/reward_model.pid"
    echo "Reward model started with PID: $PID"
    
    # 等待服务启动
    echo "Waiting for service to start..."
    for i in {1..30}; do
        if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null; then
            echo "Service is up and running!"
            return 0
        fi
        sleep 1
        echo -n "."
    done
    
    echo "Warning: Service didn't respond in 30 seconds"
    return 1
}

# 停止reward model
stop_reward_model() {
    if [ -f "${LOG_DIR}/reward_model.pid" ]; then
        PID=$(cat "${LOG_DIR}/reward_model.pid")
        if kill -0 $PID 2>/dev/null; then
            echo "Stopping reward model (PID: $PID)..."
            kill $PID
            rm "${LOG_DIR}/reward_model.pid"
        fi
    fi
}

# 清理函数
cleanup() {
    stop_reward_model
}

# 设置清理钩子
trap cleanup EXIT

# 主函数
main() {
    case "$1" in
        start)
            check_running
            start_reward_model
            ;;
        stop)
            stop_reward_model
            ;;
        restart)
            stop_reward_model
            sleep 2
            start_reward_model
            ;;
        status)
            if pgrep -f "openrlhf.models.remote_rm.math_verifier" > /dev/null; then
                echo "Reward model is running:"
                ps aux | grep "openrlhf.models.remote_rm.math_verifier" | grep -v grep
                echo -n "Service health check: "
                if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null; then
                    echo "OK"
                else
                    echo "Not responding"
                fi
            else
                echo "Reward model is not running"
            fi
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status}"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
