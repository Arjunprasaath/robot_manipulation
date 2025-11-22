#!/bin/bash

# Quick training script for VLM PPO
# This runs a short training session for testing

echo "Starting VLM PPO Training (Quick Mode)..."
echo "This will run a short training session for testing purposes"
echo ""

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
