#!/bin/bash

# Quick training script for VLM PPO
# This runs a short training session for testing

echo "Starting VLM PPO Training (Quick Mode)..."
echo "This will run a short training session for testing purposes"
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

# Run training with reduced timesteps for quick testing
.venv/bin/python train_vlm_ppo.py \
    --mode train \
    --total-timesteps 2000 \
    --rollout-steps 64 \
    --learning-rate 3e-5 \
    --save-freq 1000 \
    --project-name "vlm-robot-rl-test" \
    --run-name "quick_test_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Training complete! Check the checkpoints/ directory for saved models."
