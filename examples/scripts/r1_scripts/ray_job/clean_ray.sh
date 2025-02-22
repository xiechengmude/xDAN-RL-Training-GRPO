#!/bin/bash

# 基础配置
RAY_GCS_PORT=8100
RAY_DASHBOARD_PORT=8101

# 只清理当前用户的Ray进程
echo "Cleaning Ray processes for current user: $(whoami)"

# 1. 停止Ray服务
ray stop

# 2. 清理Ray相关进程
echo "Cleaning Ray processes..."
ps aux | grep "ray:::" | grep "$(whoami)" | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux | grep "ray_" | grep "$(whoami)" | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux | grep "redis" | grep "$(whoami)" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# 3. 检查并清理端口
echo "Cleaning ports..."
for port in $RAY_GCS_PORT $RAY_DASHBOARD_PORT; do
    pid=$(lsof -t -i:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "Killing process using port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null
    fi
done

# 4. 清理Ray临时文件和Redis数据
echo "Cleaning temporary files..."
rm -rf /tmp/ray/* 2>/dev/null
rm -rf /tmp/ray_tmp_* 2>/dev/null
rm -rf /dev/shm/ray_* 2>/dev/null
rm -rf /tmp/redis_* 2>/dev/null
rm -rf ~/.ray/session_* 2>/dev/null

# 5. 等待端口释放
echo "Waiting for ports to be released..."
sleep 5

# 6. 验证清理结果
ray_processes=$(ps aux | grep "ray\|redis" | grep "$(whoami)" | grep -v grep | wc -l)
if [ $ray_processes -eq 0 ]; then
    echo "Ray processes cleaned successfully"
else
    echo "Warning: Some Ray processes may still be running"
    ps aux | grep "ray\|redis" | grep "$(whoami)" | grep -v grep
fi

# 检查端口状态
for port in $RAY_GCS_PORT $RAY_DASHBOARD_PORT; do
    if lsof -i:$port >/dev/null 2>&1; then
        echo "Warning: Port $port is still in use"
        lsof -i:$port
    else
        echo "Port $port is free"
    fi
done
