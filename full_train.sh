#!/bin/bash

# Full training script for VLM PPO
# This runs a complete training session

echo "Starting VLM PPO Training (Full Mode)..."
echo "This will run a full training session - may take several hours"
echo ""

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
