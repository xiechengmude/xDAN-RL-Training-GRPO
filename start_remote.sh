
#!/bin/bash

# Model configuration
export MODEL_NAME="xDAN-V3-RL-72B-0315"

SCRIPT_PID=$$
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
export CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"


export DATASET_PATH="/data/vayu/train/datasets/deepscaler/deepscaler_message.jsonl" 

SAVE_PATH="/data/vayu/train/models/ckpts"

# Start remote reward model server
echo "Starting remote reward model server..."
python -m openrlhf.models.remote_rm.math_verifier \
    --dataset "${DATASET_PATH}" \
    --input_key message \
    --prompt-template chatml 2>&1 | tee -a "${CUR_LOG_DIR}/remote_rm.log" &
REMOTE_RM_PID=$!