#!/bin/bash
# Script to be executed on gpu004 to set up environments on gpu004-gpu008
# Date: 2025-03-06

set -e  # Exit on error

# Log file
LOG_FILE="/data/vayu/train/xDAN-RL-Training-GRPO/env_0306.log"
echo "Starting environment setup from gpu004 at $(date)" > $LOG_FILE

# Function to run setup on the local machine
setup_local() {
    echo "==== Setting up environment on $(hostname) ====" >> $LOG_FILE
    
    # Create conda environment
    conda create -n rl-zero5 python=3.11.11 -y
    
    # Activate the environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate rl-zero5
    
    # Install PyTorch
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    
    # Navigate to project directory
    cd /data/vayu/train/xDAN-RL-Training-GRPO
    
    # Install requirements with retry logic
    max_attempts=3
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt to install requirements..." >> $LOG_FILE
        if pip install -r requirements.txt; then
            echo "Requirements installed successfully on attempt $attempt" >> $LOG_FILE
            break
        else
            echo "Attempt $attempt failed" >> $LOG_FILE
            if [ $attempt -eq $max_attempts ]; then
                echo "All attempts to install requirements failed" >> $LOG_FILE
                exit 1
            fi
            attempt=$((attempt+1))
            sleep 5
        fi
    done
    
    # Print environment info
    {
        echo "Python version: $(python --version 2>&1)"
        echo "Conda environment: $(conda info --envs | grep "*")"
        echo "PyTorch version: $(python -c "import torch; print(torch.__version__)")"
        echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
        echo "CUDA version: $(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')")"
        echo "GPU count: $(python -c "import torch; print(torch.cuda.device_count())")"
        echo "Current GPU: $(python -c "import torch; print(torch.cuda.current_device())")"
        echo "GPU name: $(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')")"
    } >> $LOG_FILE 2>&1
    
    echo "==== Completed setup on $(hostname) ====" >> $LOG_FILE
    echo "" >> $LOG_FILE
}

# Function to run setup on a remote GPU
setup_remote_gpu() {
    local gpu_host=$1
    
    echo "==== Setting up environment on $gpu_host ====" >> $LOG_FILE
    
    # Use ssh to execute commands on the remote host
    ssh $gpu_host "bash -c '
        echo \"Starting setup on $gpu_host at $(date)\"
        
        # Create conda environment
        conda create -n rl-zero5 python=3.11.11 -y
        
        # Activate the environment
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate rl-zero5
        
        # Install PyTorch
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
        
        # Navigate to project directory
        cd /data/vayu/train/xDAN-RL-Training-GRPO
        
        # Install requirements with retry logic
        max_attempts=3
        attempt=1
        while [ $attempt -le $max_attempts ]; do
            echo \"Attempt $attempt to install requirements...\"
            if pip install -r requirements.txt; then
                echo \"Requirements installed successfully on attempt $attempt\"
                break
            else
                echo \"Attempt $attempt failed\"
                if [ $attempt -eq $max_attempts ]; then
                    echo \"All attempts to install requirements failed\"
                    exit 1
                fi
                attempt=$((attempt+1))
                sleep 5
            fi
        done
        
        # Print environment info
        echo \"Python version: $(python --version 2>&1)\"
        echo \"Conda environment: $(conda info --envs | grep \"*\")\"
        echo \"PyTorch version: $(python -c \"import torch; print(torch.__version__)\")\"
        echo \"CUDA available: $(python -c \"import torch; print(torch.cuda.is_available())\")\"
        echo \"CUDA version: $(python -c \"import torch; print(torch.version.cuda if torch.cuda.is_available() else \"N/A\")\")\"
        echo \"GPU count: $(python -c \"import torch; print(torch.cuda.device_count())\")\"
        echo \"Current GPU: $(python -c \"import torch; print(torch.cuda.current_device())\")\"
        echo \"GPU name: $(python -c \"import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")\")\"
    '" >> $LOG_FILE 2>&1
    
    echo "==== Completed setup on $gpu_host ====" >> $LOG_FILE
    echo "" >> $LOG_FILE
}

# Main execution
echo "Setting up environment on gpu004 (local)..." >> $LOG_FILE
setup_local

# Run setup on remote GPUs
echo "Setting up environment on remote GPUs (gpu005-gpu008)..." >> $LOG_FILE
for i in {5..8}; do
    gpu_host="gpu00$i"
    setup_remote_gpu $gpu_host &
done

# Wait for all background processes to complete
wait

echo "All GPU environment setups completed at $(date)" >> $LOG_FILE

# Print summary
echo "==== SUMMARY ====" >> $LOG_FILE
for i in {4..8}; do
    gpu_host="gpu00$i"
    if [ $i -eq 4 ]; then
        # Local machine
        echo "Checking environment on $gpu_host (local):" >> $LOG_FILE
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate rl-zero5
        python -c 'import torch; print(f"GPU {torch.cuda.current_device()} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"} - CUDA {torch.version.cuda if torch.cuda.is_available() else "N/A"}")' >> $LOG_FILE 2>&1
    else
        # Remote machine
        echo "Checking environment on $gpu_host:" >> $LOG_FILE
        ssh $gpu_host "source $(conda info --base)/etc/profile.d/conda.sh && conda activate rl-zero5 && python -c 'import torch; print(f\"GPU {torch.cuda.current_device()} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"} - CUDA {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}\")'" >> $LOG_FILE 2>&1
    fi
done

echo "Environment setup complete! Results logged to $LOG_FILE"
