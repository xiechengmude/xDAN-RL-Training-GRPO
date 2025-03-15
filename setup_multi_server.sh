#!/bin/bash
# 多服务器环境设置控制脚本
# 日期: 2025-03-06

set -e  # 出错时退出

# 日志文件
LOG_DIR="/data/vayu/train/xDAN-RL-Training-GRPO"
MAIN_LOG_FILE="$LOG_DIR/env_0306.log"
echo "开始在多服务器上设置环境，时间: $(date)" > $MAIN_LOG_FILE

# 要设置的服务器列表
SERVERS=("gpu004" "gpu005" "gpu006" "gpu007" "gpu008")

# 确保单服务器脚本存在并可执行
SCRIPT_PATH="$LOG_DIR/setup_single_server.sh"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 找不到脚本 $SCRIPT_PATH" | tee -a $MAIN_LOG_FILE
    exit 1
fi

chmod +x $SCRIPT_PATH

# 在远程服务器上执行设置脚本
for server in "${SERVERS[@]}"; do
    echo "==== 在 $server 上开始设置环境 ====" | tee -a $MAIN_LOG_FILE
    
    # 检查是否为本地服务器
    if [ "$(hostname)" == "$server" ]; then
        echo "在本地服务器 $server 上执行脚本" | tee -a $MAIN_LOG_FILE
        $SCRIPT_PATH | tee -a $MAIN_LOG_FILE &
    else
        echo "在远程服务器 $server 上执行脚本" | tee -a $MAIN_LOG_FILE
        # 复制脚本到远程服务器并执行
        scp $SCRIPT_PATH $server:$SCRIPT_PATH >> $MAIN_LOG_FILE 2>&1
        ssh $server "chmod +x $SCRIPT_PATH && $SCRIPT_PATH" >> $MAIN_LOG_FILE 2>&1 &
    fi
    
    echo "在 $server 上启动了设置进程" | tee -a $MAIN_LOG_FILE
done

# 等待所有后台进程完成
echo "等待所有服务器完成设置..." | tee -a $MAIN_LOG_FILE
wait

# 收集所有服务器的结果
echo "==== 环境设置摘要 ====" | tee -a $MAIN_LOG_FILE
for server in "${SERVERS[@]}"; do
    echo "检查 $server 上的环境:" | tee -a $MAIN_LOG_FILE
    
    if [ "$(hostname)" == "$server" ]; then
        # 本地服务器
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate rl-zero5
        python -c 'import torch; print(f"GPU {torch.cuda.current_device()} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"} - CUDA {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}")' >> $MAIN_LOG_FILE 2>&1
    else
        # 远程服务器
        ssh $server "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate rl-zero5 && python -c 'import torch; print(f\"GPU {torch.cuda.current_device()} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"} - CUDA {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}\")';" >> $MAIN_LOG_FILE 2>&1
    fi
done

echo "所有服务器环境设置完成！详细日志请查看 $MAIN_LOG_FILE" | tee -a $MAIN_LOG_FILE
