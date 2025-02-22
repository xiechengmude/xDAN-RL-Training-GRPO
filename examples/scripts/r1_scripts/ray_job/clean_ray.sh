#!/bin/bash

# 只清理当前用户的Ray进程
echo "Cleaning Ray processes for current user: $(whoami)"
ps aux | grep "ray:::" | grep "$(whoami)" | grep -v grep | awk '{print $2}' | xargs -r kill -9

echo "Ray processes cleaned successfully"
