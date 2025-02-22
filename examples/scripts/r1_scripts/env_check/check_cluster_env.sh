#!/bin/bash

# 定义颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# 基准机器
BASE_HOST="gpu005"
# 要检查的其他机器
OTHER_HOSTS=("gpu004" "gpu006")

# 临时文件存储目录
TEMP_DIR="/data/vayu/cluster_check"
if [ ! -d "$TEMP_DIR" ]; then
    mkdir -p $TEMP_DIR
    chmod 777 $TEMP_DIR
fi

# 获取基准机器的环境信息
get_env_info() {
    local host=$1
    local output_file="$TEMP_DIR/${host}_env.txt"
    
    ssh $host "
        # 确保目标目录存在且有权限
        if [ ! -d '$TEMP_DIR' ]; then
            mkdir -p '$TEMP_DIR'
            chmod 777 '$TEMP_DIR'
        fi
        
        echo '=== System Info ===' > $output_file
        uname -a >> $output_file
        
        echo -e '\n=== CUDA Version ===' >> $output_file
        nvidia-smi | grep 'CUDA Version' >> $output_file
        
        echo -e '\n=== GPU Driver Version ===' >> $output_file
        nvidia-smi | grep 'Driver Version' >> $output_file
        
        echo -e '\n=== Python Version ===' >> $output_file
        python3 --version >> $output_file
        
        echo -e '\n=== Conda Environment ===' >> $output_file
        conda env list | grep '*' >> $output_file
        
        echo -e '\n=== Key Python Packages ===' >> $output_file
        pip list | grep -E 'torch|ray|vllm|transformers|openrlhf' >> $output_file
        
        echo -e '\n=== NCCL Version ===' >> $output_file
        python3 -c 'import torch; print(f\"NCCL Version: {torch.cuda.nccl.version()}\")' >> $output_file 2>/dev/null
        
        echo -e '\n=== System Memory ===' >> $output_file
        free -h >> $output_file
        
        echo -e '\n=== GPU Memory ===' >> $output_file
        nvidia-smi --query-gpu=memory.total,memory.free --format=csv >> $output_file
        
        echo -e '\n=== Directory Permissions ===' >> $output_file
        ls -ld /data/vayu/train/ray >> $output_file 2>/dev/null || echo 'Ray directory not found' >> $output_file
        ls -ld /data/vayu/train/models >> $output_file 2>/dev/null || echo 'Models directory not found' >> $output_file
        
        echo -e '\n=== Network Config ===' >> $output_file
        ip addr show | grep -E 'inet.*global' >> $output_file
        
        echo -e '\n=== File Systems ===' >> $output_file
        df -h /data >> $output_file
        
        # 设置文件权限确保其他机器可以读取
        chmod 644 $output_file
    "
}

# 比较环境信息
compare_env_info() {
    local base_file="$TEMP_DIR/${BASE_HOST}_env.txt"
    local compare_file="$TEMP_DIR/$1_env.txt"
    local host=$1
    
    # 检查文件是否存在且可读
    if [ ! -r "$base_file" ]; then
        echo -e "${RED}Error: Cannot read base file $base_file${NC}"
        return 1
    fi
    
    if [ ! -r "$compare_file" ]; then
        echo -e "${RED}Error: Cannot read comparison file $compare_file${NC}"
        return 1
    fi
    
    echo -e "\n${GREEN}Comparing $host with $BASE_HOST:${NC}"
    
    # 比较各个部分
    while IFS= read -r section; do
        echo -e "\n=== Checking $section ==="
        
        # 获取每个部分的内容
        base_content=$(sed -n "/^=== $section ===$/,/^===/p" $base_file | grep -v "^===")
        compare_content=$(sed -n "/^=== $section ===$/,/^===/p" $compare_file | grep -v "^===")
        
        if [ "$base_content" = "$compare_content" ]; then
            echo -e "${GREEN}✓ $section matches${NC}"
        else
            echo -e "${RED}✗ $section differs${NC}"
            echo "Base ($BASE_HOST):"
            echo "$base_content"
            echo "Compare ($host):"
            echo "$compare_content"
        fi
    done < <(grep "^=== .* ===$" $base_file)
}

# 主函数
main() {
    echo "Starting cluster environment check..."
    
    # 获取基准机器信息
    echo "Getting base host ($BASE_HOST) environment info..."
    get_env_info $BASE_HOST
    
    # 获取其他机器信息并比较
    for host in "${OTHER_HOSTS[@]}"; do
        echo "Getting $host environment info..."
        get_env_info $host
        compare_env_info $host
    done
    
    # 清理临时文件
    echo -e "\nCleaning up temporary files..."
    rm -rf $TEMP_DIR
}

# 运行主函数
main
