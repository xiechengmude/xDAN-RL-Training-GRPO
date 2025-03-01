#!/bin/bash

# Configuration
DATASET="/data/vayu/train/datasets/math/xDAN_Agentic_openMath_r1_full.json"
DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/xDAN-Terrible-level-math-collection_chatml_rl.json"

MODEL_CPK_NAME="xDAN-L2-RL-32B-Instruct"
SAVE_PATH="./ckpts"

# Create directories
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/tensorboard"

# Check if the remote reward service is already running
if pgrep -f "openrlhf.models.remote_rm.math_verifier" > /dev/null; then
    echo "Remote reward service is already running. Stopping it first..."
    pkill -f "openrlhf.models.remote_rm.math_verifier"
    sleep 2
fi

# Start the remote reward service
echo "Starting remote reward service..."
python -m openrlhf.models.remote_rm.math_verifier \
    --dataset $DATASET \
    --input_key prompt \
    --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &

# Save the PID
echo $! > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.pid"
echo "Remote reward service started with PID: $!"
echo "Logs are being written to: ${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log"
