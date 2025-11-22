#!/bin/bash

# Full training script for VLM PPO
# This runs a complete training session

echo "Starting VLM PPO Training (Full Mode)..."
echo "This will run a full training session - may take several hours"
echo ""

# Set rendering backend (if not already set)
# This prevents MuJoCo rendering errors
if [ -z "$MUJOCO_GL" ] && [ -z "$MUJOCO_EGL_DEVICE_ID" ]; then
    echo "Setting default rendering backend..."
    if nvidia-smi &> /dev/null; then
        export MUJOCO_EGL_DEVICE_ID=0
        echo "GPU detected, using EGL rendering on GPU 0"
    else
        export MUJOCO_GL=osmesa
        echo "No GPU detected, using OSMesa (CPU) rendering"
    fi
fi

# Run full training
.venv/bin/python train_vlm_ppo.py \
    --mode train \
    --total-timesteps 100000 \
    --rollout-steps 128 \
    --learning-rate 3e-5 \
    --save-freq 10000 \
    --project-name "vlm-robot-rl" \
    --run-name "vlm_ppo_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Training complete! Check the checkpoints/ directory for saved models."
