#!/bin/bash
# 单服务器环境设置脚本
# 日期: 2025-03-06

set -e  # 出错时退出

# 日志文件
LOG_FILE="env_setup_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
echo "开始在 $(hostname) 上设置环境，时间: $(date)" > $LOG_FILE

# 设置环境函数
setup_environment() {
    echo "==== 在 $(hostname) 上设置环境 ====" | tee -a $LOG_FILE
    
    # 创建conda环境
    echo "创建conda环境: rl-zero5" | tee -a $LOG_FILE
    conda create -n rl-zero5 python=3.11.11 -y | tee -a $LOG_FILE
    
    # 激活环境
    echo "激活conda环境" | tee -a $LOG_FILE
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate rl-zero5
    
    # 安装PyTorch
    echo "安装PyTorch" | tee -a $LOG_FILE
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 | tee -a $LOG_FILE
    
    # 导航到项目目录
    echo "切换到项目目录" | tee -a $LOG_FILE
    cd /data/vayu/train/xDAN-RL-Training-GRPO
    
    # 安装依赖，包含重试逻辑
    echo "安装项目依赖" | tee -a $LOG_FILE
    max_attempts=3
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        echo "尝试 $attempt 安装requirements.txt..." | tee -a $LOG_FILE
        if pip install -r requirements.txt | tee -a $LOG_FILE; then
            echo "依赖安装成功，尝试次数: $attempt" | tee -a $LOG_FILE
            break
        else
            echo "尝试 $attempt 失败" | tee -a $LOG_FILE
            if [ $attempt -eq $max_attempts ]; then
                echo "所有安装尝试均失败" | tee -a $LOG_FILE
                exit 1
            fi
            attempt=$((attempt+1))
            sleep 5
        fi
    done
    
    # 打印环境信息
    echo "==== 环境信息 ====" | tee -a $LOG_FILE
    {
        echo "Python版本: $(python --version 2>&1)"
        echo "Conda环境: $(conda info --envs | grep "*")"
        echo "PyTorch版本: $(python -c "import torch; print(torch.__version__)")"
        echo "CUDA可用: $(python -c "import torch; print(torch.cuda.is_available())")"
        echo "CUDA版本: $(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')")"
        echo "GPU数量: $(python -c "import torch; print(torch.cuda.device_count())")"
        echo "当前GPU: $(python -c "import torch; print(torch.cuda.current_device())")"
        echo "GPU名称: $(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')")"
    } | tee -a $LOG_FILE
    
    echo "==== 在 $(hostname) 上完成环境设置 ====" | tee -a $LOG_FILE
}

# 主执行
echo "开始设置环境..." | tee -a $LOG_FILE
setup_environment

echo "环境设置完成！详细日志请查看 $LOG_FILE" | tee -a $LOG_FILE
