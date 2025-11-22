# VLM Robot RL Training with PPO

This project trains a Vision-Language Model (Qwen2-VL-2B) using Proximal Policy Optimization (PPO) to control a robot arm for picking and placing objects in a simulated environment.

## Overview

The training pipeline fine-tunes the VLM to:
- Observe the robot's eye-in-hand camera
- Decide on actions to pick up a can
- Place the can in a different location
- Learn from rewards using PPO reinforcement learning

## Components

### 1. VLM RL Policy (`rl_policy.py`)
- Wraps Qwen2-VL-2B model for RL training
- Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Includes value head for state value estimation
- Includes action head for direct action prediction

### 2. PPO Trainer (`ppo_trainer.py`)
- Implements Proximal Policy Optimization algorithm
- Handles rollout collection and advantage estimation (GAE)
- Performs policy and value network updates
- Manages experience buffer

### 3. Training Script (`train_vlm_ppo.py`)
- Main training loop
- Environment setup
- Wandb logging integration
- Checkpoint saving and evaluation

## Installation

All dependencies are already installed in the virtual environment:
```bash
source .venv/bin/activate  # Not needed on your system
```

Key packages:
- `transformers` - For VLM model
- `torch` - Deep learning framework
- `peft` - LoRA fine-tuning
- `wandb` - Experiment tracking
- `robosuite` - Robot simulation environment

## Usage

### Quick Test (2000 timesteps)
```bash
./quick_train.sh
```

### Full Training (100,000 timesteps)
```bash
./full_train.sh
```

### Custom Training
```bash
.venv/bin/python train_vlm_ppo.py \
    --mode train \
    --total-timesteps 50000 \
    --rollout-steps 128 \
    --learning-rate 3e-5 \
    --save-freq 10000 \
    --project-name "my-vlm-robot" \
    --run-name "experiment_1"
```

### Training Without Wandb
```bash
.venv/bin/python train_vlm_ppo.py \
    --mode train \
    --total-timesteps 10000 \
    --no-wandb
```

### Evaluation
```bash
.venv/bin/python train_vlm_ppo.py \
    --mode eval \
    --checkpoint checkpoints/vlm_ppo_best.pt \
    --num-eval-episodes 10
```

## Configuration

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 100000 | Total environment steps |
| `rollout_steps` | 128 | Steps per rollout collection |
| `learning_rate` | 3e-5 | Learning rate for optimizer |
| `clip_range` | 0.2 | PPO clipping parameter |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda parameter |
| `n_epochs` | 4 | Training epochs per update |
| `batch_size` | 4 | Batch size for training |
| `value_coef` | 0.5 | Value loss coefficient |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |

### Action Space

The robot has 8 discrete actions:
- `forward` - Move gripper forward
- `backward` - Move gripper backward
- `left` - Move gripper left
- `right` - Move gripper right
- `up` - Move gripper up
- `down` - Move gripper down
- `close gripper` - Close gripper
- `open gripper` - Open gripper

### Reward Function

The environment provides shaped rewards for:
- Moving closer to the can
- Grasping the can
- Lifting the can
- Placing the can in the target location

## Training Outputs

### Checkpoints
Saved in `checkpoints/` directory:
- `{run_name}_step{N}.pt` - Periodic checkpoints
- `{run_name}_best.pt` - Best model based on episode reward

### Logs
- Console output shows training progress
- Wandb dashboard (if enabled) shows:
  - Episode rewards
  - Episode lengths
  - Policy loss
  - Value loss
  - Entropy

## Wandb Integration

To use Wandb:

1. Login to Wandb:
```bash
wandb login
```

2. Run training (Wandb enabled by default):
```bash
.venv/bin/python train_vlm_ppo.py --mode train
```

View results at: https://wandb.ai/your-username/vlm-robot-rl

## Tips for Training

### Faster Training
- Reduce `rollout_steps` to 64 or 32
- Reduce `total_timesteps`
- Use fewer `n_epochs` (e.g., 2)

### Better Performance
- Increase `total_timesteps` to 200000+
- Tune `learning_rate` (try 1e-5 to 1e-4)
- Adjust `clip_range` (try 0.1 to 0.3)
- Increase `rollout_steps` to 256

### Debugging
- Use `--no-wandb` for faster iteration
- Check `checkpoints/` for saved models
- Run evaluation to test learned policy

## Architecture

```
VLM RL Policy
├── Qwen2-VL-2B (vision encoder + language model)
│   └── LoRA adapters (trainable)
├── Action Head (Linear layers)
│   └── Outputs action logits
└── Value Head (Linear layers)
    └── Outputs state value estimate

PPO Trainer
├── Rollout Buffer
│   └── Stores transitions
├── GAE Computation
│   └── Computes advantages
└── Policy Update
    ├── Policy loss (clipped objective)
    ├── Value loss (MSE)
    └── Entropy bonus
```

## Expected Training Time

On CPU:
- Quick test (2000 steps): ~30-60 minutes
- Full training (100k steps): ~24-48 hours

Note: Training is slow on CPU. For production use, consider using GPU/MPS acceleration.

## Troubleshooting

### Out of Memory
- Reduce `batch_size` to 2
- Reduce `rollout_steps` to 32

### Slow Training
- Training on CPU is slow
- Consider reducing image resolution in environment
- Use smaller rollout steps

### Poor Performance
- Increase training timesteps
- Adjust learning rate
- Check reward shaping in environment
- Ensure LoRA is properly applied

## Next Steps

After training:
1. Evaluate the best checkpoint
2. Visualize learned behaviors
3. Fine-tune hyperparameters
4. Try longer training runs
5. Experiment with different reward functions
