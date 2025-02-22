# xDAN-RL 分布式训练快速指南

本指南提供了启动分布式训练的核心命令，省去了脚本中的各种检查和错误处理逻辑。

## 环境要求

- 主节点：gpu005
- 从节点：gpu004
- 每节点8张GPU
- Ray GCS端口：8100（避免默认6379）
- Ray Dashboard端口：8101（避免默认8265）
- Ray Client端口：8102（避免默认10001）
- Reward Model端口：5001（避免默认5000）
- 临时目录：/data/vayu/train/ray/vayu_cluster
- 存储目录：/data/vayu/train/ray/vayu_cluster/storage
- 数据集：`/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json`
- 预训练模型：`/data/vayu/train/models/xDAN-L2-Qwen25-32b-Instruct`

## 1. 启动Ray Head节点

在gpu005上执行：

```bash
cd /data/vayu/train/xDAN-RL-Training-GRPO

# 基础配置
RAY_PASSWORD="xdan_user_ray_2024"  # 简单固定密码
RAY_GCS_PORT=8100                  # GCS端口(避免默认6379)
RAY_DASHBOARD_PORT=8101            # Dashboard端口(避免默认8265)
RAY_CLIENT_PORT=8102               # Client端口(避免默认10001)
REWARD_MODEL_PORT=5001             # RM端口(避免默认5000)

# 确保目录存在并设置权限
mkdir -p /data/vayu/train/ray/vayu_cluster/storage
chmod 700 /data/vayu/train/ray/vayu_cluster

ray start --head \
    --port=$RAY_GCS_PORT \
    --dashboard-port=$RAY_DASHBOARD_PORT \
    --ray-client-server-port=$RAY_CLIENT_PORT \
    --temp-dir=/data/vayu/train/ray/vayu_cluster \
    --storage=/data/vayu/train/ray/vayu_cluster/storage \
    --dashboard-host=0.0.0.0 \
    --num-gpus=8 \
    --disable-usage-stats \
    --system-config='{"automatic_object_spilling_enabled":true,"object_spilling_config":{"type":"filesystem","params":{"directory_path":"/data/vayu/train/ray/vayu_cluster/spill"}},"debug_mode":true}' \
    --redis-password="$RAY_PASSWORD"
```

## 2. 启动Ray Worker节点

在gpu004上执行：

```bash
cd /data/vayu/train/xDAN-RL-Training-GRPO

# 基础配置（与head节点相同）
RAY_PASSWORD="xdan_user_ray_2024"
RAY_GCS_PORT=8100

# 确保目录存在
mkdir -p /data/vayu/train/ray/vayu_cluster/storage

ray start \
    --address=gpu005:$RAY_GCS_PORT \
    --temp-dir=/data/vayu/train/ray/vayu_cluster \
    --storage=/data/vayu/train/ray/vayu_cluster/storage \
    --num-gpus=8 \
    --redis-password="$RAY_PASSWORD"
```

## 3. 启动Reward Model服务

在gpu005上执行：

```bash
cd /data/vayu/train/xDAN-RL-Training-GRPO

# 基础配置
REWARD_MODEL_PORT=5001             # 避免默认5000端口
DATASET="/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json"

python -m openrlhf.models.remote_rm.math_verifier \
    --dataset $DATASET \
    --input_key message \
    --prompt-template chatml \
    --port $REWARD_MODEL_PORT &
```

等待几秒确保服务启动成功。可以通过以下命令检查服务状态：
```bash
curl http://127.0.0.1:$REWARD_MODEL_PORT/health_check
```

## 4. 提交训练任务

在gpu005上执行：

```bash
cd /data/vayu/train/xDAN-RL-Training-GRPO

# 基础配置
RAY_DASHBOARD_PORT=8101            # Dashboard端口
REWARD_MODEL_PORT=5001             # RM端口
DATASET="/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json"
MODEL_NAME="xDAN-L2-RL-32B-Instruct"
PRETRAIN="/data/vayu/train/models/xDAN-L2-Qwen25-32b-Instruct"

# 训练参数配置
REF_NUM_NODES=2                    # 参考模型节点数
REF_GPUS_PER_NODE=4               # 每节点参考模型GPU数
ACTOR_NUM_NODES=2                  # Actor模型节点数
ACTOR_GPUS_PER_NODE=4             # 每节点Actor模型GPU数
VLLM_NUM_ENGINES=2                # vLLM引擎数
VLLM_TP_SIZE=4                    # 张量并行大小
VLLM_MEM_UTIL=0.85               # GPU内存利用率

# 批处理参数
MICRO_TRAIN_BATCH=2              # 微训练批次大小
TRAIN_BATCH=128                  # 训练批次大小
MICRO_ROLLOUT_BATCH=4           # 微滚动批次大小
ROLLOUT_BATCH=256               # 滚动批次大小

# 训练控制参数
MAX_EPOCHS=1                     # 最大训练轮数
NUM_EPISODES=30                  # 训练回合数
PROMPT_MAX_LEN=4096             # 提示最大长度
MAX_SAMPLES=100000              # 最大样本数
GEN_MAX_LEN=4096               # 生成最大长度
TEMPERATURE=1                   # 采样温度
N_SAMPLES_PER_PROMPT=8         # 每个提示的采样数

# 优化器参数
ACTOR_LR=5e-7                  # Actor学习率
INIT_KL_COEF=0.02             # 初始KL系数

RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
    --working-dir="." \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes $REF_NUM_NODES \
    --ref_num_gpus_per_node $REF_GPUS_PER_NODE \
    --remote_rm_url "http://127.0.0.1:$REWARD_MODEL_PORT/get_reward" \
    --actor_num_nodes $ACTOR_NUM_NODES \
    --actor_num_gpus_per_node $ACTOR_GPUS_PER_NODE \
    --vllm_num_engines $VLLM_NUM_ENGINES \
    --vllm_tensor_parallel_size $VLLM_TP_SIZE \
    --colocate_all_models \
    --vllm_enable_sleep \
    --vllm_gpu_memory_utilization $VLLM_MEM_UTIL \
    --vllm_sync_backend gloo \
    --enable_prefix_caching \
    --pretrain $PRETRAIN \
    --save_path "./ckpts/$MODEL_NAME" \
    --micro_train_batch_size $MICRO_TRAIN_BATCH \
    --train_batch_size $TRAIN_BATCH \
    --micro_rollout_batch_size $MICRO_ROLLOUT_BATCH \
    --rollout_batch_size $ROLLOUT_BATCH \
    --temperature $TEMPERATURE \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --max_epochs $MAX_EPOCHS \
    --num_episodes $NUM_EPISODES \
    --prompt_max_len $PROMPT_MAX_LEN \
    --max_samples $MAX_SAMPLES \
    --generate_max_len $GEN_MAX_LEN \
    --advantage_estimator rloo \
    --zero_stage 3 \
    --adam_offload \
    --bf16 \
    --actor_learning_rate $ACTOR_LR \
    --init_kl_coef $INIT_KL_COEF \
    --prompt_data $DATASET \
    --input_key message \
    --normalize_reward \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 10 \
    --ckpt_path "./ckpts/$MODEL_NAME/ckpt" \
    --save_hf_ckpt
```

## 训练配置说明

### 1. 分布式配置
- 参考模型（Reference Model）：2节点，每节点4 GPU
- Actor模型：2节点，每节点4 GPU
- vLLM引擎：2引擎，每引擎4卡张量并行
- 总计使用：16张GPU（分布在两个节点上）

### 2. 批处理参数
- 微训练批次：2
- 训练批次：128
- 微滚动批次：4
- 滚动批次：256

### 3. 训练控制
- 最大轮数：1
- 训练回合：30
- 提示最大长度：4096
- 生成最大长度：4096
- 最大样本数：100000

### 4. 优化器设置
- Actor学习率：5e-7
- 初始KL系数：0.02
- 使用Adam优化器
- 使用BF16精度
- 启用梯度检查点

## 常见问题处理

1. 清理Ray进程：
```bash
# 只清理当前用户的Ray进程
ps aux | grep "ray:::" | grep "$(whoami)" | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

2. 检查Ray集群状态：
```bash
# 使用之前设置的GCS端口
RAY_GCS_PORT=8100
ray status --address=gpu005:$RAY_GCS_PORT
```

3. 检查Reward Model服务：
```bash
# 使用之前设置的RM端口
REWARD_MODEL_PORT=5001
curl http://127.0.0.1:$REWARD_MODEL_PORT/health_check
```

4. 查看训练日志：
```bash
# 可以通过Ray dashboard查看：http://gpu005:8101
# 或者查看本地日志文件：
MODEL_NAME="xDAN-L2-RL-32B-Instruct"
tail -f "./ckpts/$MODEL_NAME/logs/*"
```

## 安全注意事项

1. 目录权限
```bash
# 设置目录权限（仅当前用户可访问）
chmod 700 /data/vayu/train/ray/vayu_cluster
chmod 700 /data/vayu/train/ray/vayu_cluster/storage
```

2. 端口配置
- 使用非默认端口，避免与root用户或其他用户冲突：
  * GCS: 8100 (避免默认6379)
  * Dashboard: 8101 (避免默认8265)
  * Client: 8102 (避免默认10001)
  * Reward Model: 5001 (避免默认5000)

3. 密码管理
- 使用简单固定密码：`xdan_user_ray_2024`
- 确保head节点和worker节点使用相同的密码
- 密码主要用于防止不同用户的Ray集群互相干扰

4. Debug模式
- Ray集群默认启用debug模式
- 可以在system-config中看到更详细的日志
- 有助于排查问题

5. 网络安全
- Dashboard仅允许指定IP访问
- 使用防火墙限制端口访问
- 监控异常连接尝试
