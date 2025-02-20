# xDAN-RL 分布式训练指南

## 环境要求

- 两台服务器：gpu005（主节点）和gpu004（次节点）
- 每台服务器8张GPU
- 相同的代码和数据路径结构
- 节点间网络互通

## 目录结构

```bash
/data/vayu/train/
├── xDAN-RL-Training-GRPO/      # 代码目录
├── datasets/                    # 数据集目录
│   └── xDAN-Agentic-openMath-r1-chatml.json
├── models/                      # 预训练模型目录
│   └── xDAN-L2-Qwen25-32b-Instruct/
└── ray/                        # Ray临时目录
```

## 训练脚本说明

主要训练脚本：
- `xDAN-L2-RL-32b-Instruct-0220-nodes.sh`: 32B模型双节点训练脚本
- `xDAN-L1-Edge-7b-Math-RL-0220.sh`: 7B模型训练脚本
- `xDAN-L1-VL-RL-Vision-0220.sh`: 视觉模型训练脚本

## 启动步骤

### 1. 准备工作

确保两台服务器上都已经：
- 克隆代码仓库
- 准备好数据集和预训练模型
- 创建必要的目录

```bash
# 在两台服务器上都执行
cd /data/vayu/train/
mkdir -p ray
```

### 2. 启动训练

#### 2.1 在主节点（gpu005）上：

```bash
# 1. 进入项目目录
cd /data/vayu/train/xDAN-RL-Training-GRPO

# 2. 启动主节点
./examples/scripts/r1_scripts/xDAN-L2-RL-32b-Instruct-0220-nodes.sh head gpu005
```

#### 2.2 在次节点（gpu004）上：

```bash
# 1. 进入项目目录
cd /data/vayu/train/xDAN-RL-Training-GRPO

# 2. 启动次节点
./examples/scripts/r1_scripts/xDAN-L2-RL-32b-Instruct-0220-nodes.sh worker gpu005
```

### 3. 验证集群状态

在任一节点上运行：
```bash
ray status
```

应该能看到两个节点都已加入集群。

## 训练配置说明

当前配置：
- 参考模型（Reference Model）：2节点，每节点4 GPU
- Actor模型：2节点，每节点4 GPU
- vLLM引擎：2引擎，每引擎4卡张量并行
- 总计使用：8张GPU（分布在两个节点上）

主要参数：
```bash
--ref_num_nodes 2                # 参考模型节点数
--ref_num_gpus_per_node 4       # 每节点分配给参考模型的GPU数
--actor_num_nodes 2             # Actor节点数
--actor_num_gpus_per_node 4     # 每节点分配给Actor的GPU数
--vllm_num_engines 2            # vLLM引擎数量
--vllm_tensor_parallel_size 4   # 每个引擎的张量并行度
```

## 错误处理

如果遇到启动问题：

1. 清理Ray进程和缓存：
```bash
ray stop
pkill -9 ray
rm -rf /tmp/ray/*
rm -rf ~/.cache/ray/*
rm -rf /data/vayu/train/ray/*
```

2. 检查网络连接：
```bash
ping gpu005  # 在gpu004上执行
ping gpu004  # 在gpu005上执行
```

3. 检查端口占用：
```bash
netstat -tunlp | grep 6379  # Ray默认端口
netstat -tunlp | grep 8265  # Dashboard端口
```

## 监控

- Ray Dashboard: http://gpu005:8265
- 训练日志：`./ckpts/xDAN-L2-RL-32B-Instruct/`
- Reward Model日志：`./ckpts/xDAN-L2-RL-32B-Instruct/remote_rm.log`

## 停止训练

在两个节点上都运行：
```bash
ray stop
pkill -9 ray
```
