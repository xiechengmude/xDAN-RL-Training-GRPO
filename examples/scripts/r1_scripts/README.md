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
- `start_ray_cluster.sh`: Ray集群启动脚本
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

### 2. 启动Ray集群

#### 2.1 在主节点（gpu005）上：

```bash
# 1. 进入项目目录
cd /data/vayu/train/xDAN-RL-Training-GRPO

# 2. 启动Ray主节点
./examples/scripts/r1_scripts/start_ray_cluster.sh head gpu005
```

#### 2.2 在次节点（gpu004）上：

```bash
# 1. 进入项目目录
cd /data/vayu/train/xDAN-RL-Training-GRPO

# 2. 启动Ray次节点
./examples/scripts/r1_scripts/start_ray_cluster.sh worker gpu005
```

### 3. 启动训练

在主节点（gpu005）上运行训练脚本：

```bash
./examples/scripts/r1_scripts/xDAN-L2-RL-32b-Instruct-0220-nodes.sh
```

### 4. 验证集群状态

在任一节点上运行：
```bash
ray status --address=gpu005:8100
```

应该能看到两个节点都已加入集群，每个节点显示8个可用GPU。

## 训练配置说明

当前配置：
- 参考模型（Reference Model）：2节点，每节点4 GPU
- Actor模型：2节点，每节点4 GPU
- vLLM引擎：2引擎，每引擎4卡张量并行
- 总计使用：16张GPU（分布在两个节点上）

主要参数：
```bash
--ref_num_nodes 2                # 参考模型节点数
--ref_num_gpus_per_node 4       # 每节点分配给参考模型的GPU数
--actor_num_nodes 2             # Actor节点数
--actor_num_gpus_per_node 4     # 每节点分配给Actor的GPU数
--vllm_num_engines 2            # vLLM引擎数量
--vllm_tensor_parallel_size 4   # 每个引擎的张量并行度
```

## 端口配置

Ray集群使用以下端口：
- Ray主端口：8100（默认6379）
- Dashboard端口：8101（默认8265）

这些非标准端口的使用可以避免与其他用户的Ray实例冲突。

## 错误处理

如果遇到启动问题：

1. 清理Ray进程：
```bash
# 只清理当前用户的Ray进程
ps aux | grep "ray:::" | grep "$(whoami)" | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

2. 检查网络连接：
```bash
# 测试节点间连接
ping gpu004  # 在gpu005上运行
ping gpu005  # 在gpu004上运行
```

3. 检查端口占用：
```bash
# 检查Ray端口是否被占用
lsof -i :8100
lsof -i :8101
```

## 注意事项

1. 确保两个节点上的代码版本一致
2. 训练前检查数据集和模型文件是否就绪
3. 建议使用tmux或screen运行长时间训练任务
4. 定期检查训练日志和模型保存点
5. 集群启动脚本只会清理当前用户的Ray进程，不会影响其他用户
6. 使用非标准端口可以避免与其他用户的Ray实例冲突
