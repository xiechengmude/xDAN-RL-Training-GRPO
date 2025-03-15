# 多服务器环境设置脚本

这些脚本用于在多个GPU服务器（gpu004至gpu008）上设置RL训练环境。

## 脚本说明

1. `setup_single_server.sh` - 单服务器环境设置脚本
   - 在单个服务器上创建并配置conda环境
   - 安装PyTorch和项目依赖
   - 生成环境信息报告

2. `setup_multi_server.sh` - 多服务器控制脚本
   - 在多个服务器上协调环境设置
   - 将单服务器脚本分发到各个目标服务器
   - 收集所有服务器的设置结果
   - 生成汇总报告

## 使用方法

### 在单个服务器上设置环境

```bash
# 确保脚本有执行权限
chmod +x setup_single_server.sh

# 在当前服务器上执行
./setup_single_server.sh
```

### 在多个服务器上设置环境

```bash
# 确保脚本有执行权限
chmod +x setup_multi_server.sh setup_single_server.sh

# 从gpu004执行（或任何可以ssh到其他服务器的主机）
./setup_multi_server.sh
```

## 日志文件

- 单服务器设置日志: `env_setup_<hostname>_<timestamp>.log`
- 多服务器汇总日志: `/data/vayu/train/xDAN-RL-Training-GRPO/env_0306.log`

## 环境配置

脚本将设置以下环境：

- Conda环境: `rl-zero5`
- Python版本: `3.11.11`
- PyTorch版本: `2.5.1`
- CUDA版本: `cu124`
- 项目依赖: 从项目根目录的`requirements.txt`安装

## 注意事项

1. 确保服务器之间已配置SSH密钥认证，以实现无密码登录
2. 确保所有服务器上的conda已正确安装
3. 确保所有服务器上的项目目录结构相同
4. 脚本包含重试逻辑，最多尝试3次安装项目依赖
