#!/bin/bash

# Create and activate conda environment
conda create -n openrlhf_new python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate openrlhf_new

# Install CUDA toolkit and OpenMP
conda install -y cudatoolkit=11.8
conda install -y libomp

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install deepspeed first with specific version
pip install ninja
CUDA_HOME=/usr/local/cuda-11.8 pip install deepspeed==0.16.3 --no-cache-dir

# Install wandb and its dependencies first
pip install \
    wandb \
    docker-pycreds \
    sentry-sdk \
    psutil

# Install trl and its dependencies
pip install \
    trl \
    accelerate \
    bitsandbytes \
    datasets \
    einops \
    flask \
    isort \
    jsonlines \
    loralib \
    math-verify \
    levenshtein \
    optimum \
    packaging \
    peft \
    "pynvml>=12.0.0" \
    qwen_vl_utils \
    "ray[default]==2.42.0" \
    tensorboard \
    torchmetrics \
    tqdm \
    transformers_stream_generator \
    wheel

# Install transformers from source
pip install git+https://github.com/huggingface/transformers@main

# Verify installations
python -c "import torch; import deepspeed; import wandb; import trl; print('PyTorch version:', torch.__version__); print('DeepSpeed version:', deepspeed.__version__); print('Wandb version:', wandb.__version__); print('TRL version:', trl.__version__)"

echo "Environment setup completed. Please check for any errors above."
echo "You may need to run 'conda activate openrlhf_new' again to use the environment."
