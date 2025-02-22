#!/bin/bash

# 定义颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Checking System Environment ===${NC}"

# 1. 系统信息
echo -e "\n=== System Info ==="
uname -a

# 2. CUDA 版本
echo -e "\n=== CUDA Version ==="
nvidia-smi | grep 'CUDA Version'

# 3. GPU 驱动版本
echo -e "\n=== GPU Driver Version ==="
nvidia-smi | grep 'Driver Version'

# 4. Python 环境
echo -e "\n=== Python Version ==="
which python3
python3 --version

# 5. 当前激活的 Conda 环境
echo -e "\n=== Active Conda Environment ==="
echo "Conda Path: $(which conda)"
echo "Current Environment: $CONDA_DEFAULT_ENV"
conda env list | grep '*' || echo "No active conda environment"

# 6. 关键 Python 包版本
echo -e "\n=== Key Python Packages ==="
pip list | grep -E 'torch|ray|vllm|transformers|openrlhf'

# 7. NCCL 版本
echo -e "\n=== NCCL Version ==="
python3 -c 'import torch; print(f"NCCL Version: {torch.cuda.nccl.version()}")' 2>/dev/null || echo "Failed to get NCCL version"

# 8. 系统内存
echo -e "\n=== System Memory ==="
free -h

# 9. GPU 信息
echo -e "\n=== GPU Info ==="
nvidia-smi

# 10. 关键目录权限
echo -e "\n=== Directory Permissions ==="
echo "Ray directory:"
ls -ld /data/vayu/train/ray 2>/dev/null || echo "Ray directory not found"
echo -e "\nModels directory:"
ls -ld /data/vayu/train/models 2>/dev/null || echo "Models directory not found"

# 11. 网络配置
echo -e "\n=== Network Config ==="
ip addr show | grep -E 'inet.*global'

# 12. 文件系统
echo -e "\n=== File Systems ==="
df -h /data

# 13. Python 路径
echo -e "\n=== Python Path ==="
echo $PYTHONPATH

# 14. LD_LIBRARY_PATH
echo -e "\n=== LD_LIBRARY_PATH ==="
echo $LD_LIBRARY_PATH
