# xDAN-RL 分布式训练快速指南

本指南提供了启动分布式训练的核心命令，省去了脚本中的各种检查和错误处理逻辑。

## 环境要求

- 主节点：gpu005
- 从节点：gpu004
- 每节点8张GPU
- Ray GCS端口：8100（默认6379）
- Ray Dashboard端口：8101（默认8265）
- Ray Client端口：8102（默认10001）
- 临时目录：/data/vayu/train/ray/vayu_cluster
- 存储目录：/data/vayu/train/ray/vayu_cluster/storage
- 数据集：`/data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json`
- 预训练模型：`/data/vayu/train/models/xDAN-L2-Qwen25-32b-Instruct`

## 1. 启动Ray Head节点

在gpu005上执行：

```bash
cd /data/vayu/train/xDAN-RL-Training-GRPO

# 确保目录存在并设置权限
mkdir -p /data/vayu/train/ray/vayu_cluster/storage
chmod 700 /data/vayu/train/ray/vayu_cluster

# 生成随机密码（每次启动集群时更新）
RAY_PASSWORD=$(openssl rand -hex 16)
echo "Ray cluster password: $RAY_PASSWORD"

ray start --head \
    --port=8100 \
    --dashboard-port=8101 \
    --ray-client-server-port=8102 \
    --temp-dir=/data/vayu/train/ray/vayu_cluster \
    --storage=/data/vayu/train/ray/vayu_cluster/storage \
    --dashboard-host=0.0.0.0 \
    --num-gpus=8 \
    --disable-usage-stats \
    --system-config='{"automatic_object_spilling_enabled":true,"object_spilling_config":{"type":"filesystem","params":{"directory_path":"/data/vayu/train/ray/vayu_cluster/spill"}}}' \
    --redis-password="$RAY_PASSWORD"

# 保存连接信息（仅当前用户可读）
echo "$RAY_PASSWORD" > /data/vayu/train/ray/vayu_cluster/.ray_password
chmod 600 /data/vayu/train/ray/vayu_cluster/.ray_password
```

## 2. 启动Ray Worker节点

在gpu004上执行：

```bash
cd /data/vayu/train/xDAN-RL-Training-GRPO

# 确保目录存在
mkdir -p /data/vayu/train/ray/vayu_cluster/storage

# 读取Ray密码
RAY_PASSWORD=$(cat /data/vayu/train/ray/vayu_cluster/.ray_password)

ray start \
    --address=gpu005:8100 \
    --temp-dir=/data/vayu/train/ray/vayu_cluster \
    --storage=/data/vayu/train/ray/vayu_cluster/storage \
    --num-gpus=8 \
    --redis-password="$RAY_PASSWORD"
```

## 3. 启动Reward Model服务

在gpu005上执行：

```bash
cd /data/vayu/train/xDAN-RL-Training-GRPO
python -m openrlhf.models.remote_rm.math_verifier \
    --dataset /data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json \
    --input_key message \
    --prompt-template chatml &
```

等待几秒确保服务启动成功。可以通过以下命令检查服务状态：
```bash
curl http://127.0.0.1:5000/health_check
```

## 4. 提交训练任务

在gpu005上执行：

```bash
cd /data/vayu/train/xDAN-RL-Training-GRPO
RAY_ADDRESS='http://127.0.0.1:8101' ray job submit --working-dir . -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 2 \
   --ref_num_gpus_per_node 4 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 2 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 4 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.85 \
   --vllm_sync_backend gloo \
   --enable_prefix_caching \
   --pretrain /data/vayu/train/models/xDAN-L2-Qwen25-32b-Instruct \
   --save_path ./ckpts/xDAN-L2-RL-32B-Instruct \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 256 \
   --temperature 1 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --num_episodes 30 \
   --prompt_max_len 4096 \
   --max_samples 100000 \
   --generate_max_len 4096 \
   --advantage_estimator rloo \
   --zero_stage 3 \
   --adam_offload \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.02 \
   --prompt_data /data/vayu/train/datasets/xDAN-Agentic-openMath-r1-chatml.json \
   --input_key message \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 10 \
   --ckpt_path ./ckpts/xDAN-L2-RL-32B-Instruct/ckpt --save_hf_ckpt
```

## 训练配置说明

- 参考模型（Reference Model）：2节点，每节点4 GPU
- Actor模型：2节点，每节点4 GPU
- vLLM引擎：2引擎，每引擎4卡张量并行
- 总计使用：16张GPU（分布在两个节点上）

## 常见问题处理

1. 清理Ray进程：
```bash
# 只清理当前用户的Ray进程
ps aux | grep "ray:::" | grep "$(whoami)" | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

2. 检查Ray集群状态：
```bash
ray status --address=gpu005:8100
```

3. 检查Reward Model服务：
```bash
curl http://127.0.0.1:5000/health_check
```

4. 查看训练日志：
```bash
# 可以通过Ray dashboard查看：http://gpu005:8101
# 或者查看本地日志文件：
tail -f ./ckpts/xDAN-L2-RL-32B-Instruct/logs/*
```

## 安全注意事项

1. 目录权限
```bash
# 设置目录权限（仅当前用户可访问）
chmod 700 /data/vayu/train/ray/vayu_cluster
chmod 700 /data/vayu/train/ray/vayu_cluster/storage
```

2. 密码管理
```bash
# 更新Ray集群密码
RAY_PASSWORD=$(openssl rand -hex 16)
echo "$RAY_PASSWORD" > /data/vayu/train/ray/vayu_cluster/.ray_password
chmod 600 /data/vayu/train/ray/vayu_cluster/.ray_password
```

3. 端口安全
- 使用非默认端口（8100, 8101, 8102）
- 避免使用常见端口（如6379, 8265, 10001）
- 建议定期更换端口

4. 网络安全
- Dashboard仅允许指定IP访问
- 使用防火墙限制端口访问
- 监控异常连接尝试
