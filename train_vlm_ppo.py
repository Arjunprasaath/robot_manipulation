import numpy as np
import torch
import robosuite as rb
import wandb
import os
from datetime import datetime
from rl_policy import VLMRLPolicy
from ppo_trainer import PPOTrainer


def create_env():
    """Create the robosuite environment."""
    import os as os_module

    # Set EGL device ID if running on GPU server
    if 'MUJOCO_EGL_DEVICE_ID' not in os_module.environ and 'MUJOCO_GL' not in os_module.environ:
        # Default to GPU 0 if available, otherwise use OSMesa
        os_module.environ['MUJOCO_EGL_DEVICE_ID'] = '0'

    try:
        # Try with offscreen renderer (for camera observations)
        env = rb.make(
            env_name="PickPlaceCan",
            robots="Panda",
            use_camera_obs=True,
            camera_names="robot0_eye_in_hand",
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
            reward_shaping=True,
            render_gpu_device_id=0  # Use GPU 0
        )
    except RuntimeError as e:
        print(f"Warning: Could not create environment with offscreen renderer: {e}")
        print("Trying with CPU rendering...")

        # Fallback: Use CPU rendering
        os_module.environ['MUJOCO_GL'] = 'osmesa'  # Use OSMesa (CPU) rendering

        env = rb.make(
            env_name="PickPlaceCan",
            robots="Panda",
            use_camera_obs=True,
            camera_names="robot0_eye_in_hand",
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
            reward_shaping=True
        )

    return env


def train(
    total_timesteps: int = 100000,
    rollout_steps: int = 128,
    learning_rate: float = 3e-5,
    save_freq: int = 10000,
    eval_freq: int = 5000,
    project_name: str = "vlm-robot-rl",
    run_name: str = None,
    use_wandb: bool = True,
    device: str = None,
):
    """
    Train VLM policy with PPO.

    Args:
        total_timesteps: Total number of environment steps
        rollout_steps: Steps per rollout
        learning_rate: Learning rate
        save_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        project_name: Wandb project name
        run_name: Wandb run name
        use_wandb: Whether to use wandb logging
        device: Device to use (cuda/cpu/mps), auto-detect if None
    """
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"vlm_ppo_{timestamp}"

    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "total_timesteps": total_timesteps,
                "rollout_steps": rollout_steps,
                "learning_rate": learning_rate,
                "clip_range": 0.2,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "n_epochs": 4,
                "batch_size": 4,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
            }
        )

    # Define action mapping
    action_map = {
        'backward': np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'forward': np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'left': np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'right': np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'up': np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
        'down': np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]),
        'close gripper': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        'open gripper': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    }

    # Create environment
    print("Creating environment...")
    env = create_env()
    env.reset()

    # Create policy
    print("Creating VLM RL policy...")
    model_path = os.path.join(os.path.dirname(__file__), "models/Qwen2-VL-2B")
    policy = VLMRLPolicy(model_path, action_map, use_lora=True, device=device)

    # Create trainer
    print("Creating PPO trainer...")
    trainer = PPOTrainer(
        policy=policy,
        learning_rate=learning_rate,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=4,
        batch_size=4,
    )

    # Training loop
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"Rollout steps: {rollout_steps}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)

    num_updates = 0
    total_steps = 0
    best_reward = -float('inf')

    while total_steps < total_timesteps:
        # Collect rollout
        print(f"\n[Update {num_updates + 1}] Collecting rollout ({rollout_steps} steps)...")
        rollout_stats, returns, advantages = trainer.collect_rollout(env, rollout_steps)

        total_steps += rollout_steps

        # Update policy
        print("Training policy...")
        loss_stats = trainer.train_step(returns, advantages)

        # Logging
        mean_reward = rollout_stats['mean_episode_reward']
        mean_length = rollout_stats['mean_episode_length']
        num_episodes = rollout_stats['num_episodes']

        print(f"Steps: {total_steps}/{total_timesteps}")
        print(f"Episodes: {num_episodes}")
        print(f"Mean reward: {mean_reward:.4f}")
        print(f"Mean length: {mean_length:.1f}")
        print(f"Policy loss: {loss_stats['policy_loss']:.4f}")
        print(f"Value loss: {loss_stats['value_loss']:.4f}")
        print(f"Entropy: {-loss_stats['entropy_loss']:.4f}")

        if use_wandb:
            wandb.log({
                'rollout/mean_episode_reward': mean_reward,
                'rollout/mean_episode_length': mean_length,
                'rollout/num_episodes': num_episodes,
                'train/policy_loss': loss_stats['policy_loss'],
                'train/value_loss': loss_stats['value_loss'],
                'train/entropy': -loss_stats['entropy_loss'],
                'train/total_steps': total_steps,
            }, step=num_updates)

        # Save checkpoint
        if (total_steps % save_freq == 0) or (total_steps >= total_timesteps):
            checkpoint_path = f"checkpoints/{run_name}_step{total_steps}.pt"
            policy.save(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

            if use_wandb:
                wandb.save(checkpoint_path)

        # Save best model
        if mean_reward > best_reward and num_episodes > 0:
            best_reward = mean_reward
            best_path = f"checkpoints/{run_name}_best.pt"
            policy.save(best_path)
            print(f"New best model saved! Reward: {best_reward:.4f}")

            if use_wandb:
                wandb.save(best_path)

        num_updates += 1

    # Cleanup
    env.close()

    if use_wandb:
        wandb.finish()

    print("\nTraining complete!")
    print(f"Best reward: {best_reward:.4f}")
    print(f"Final checkpoint: checkpoints/{run_name}_step{total_steps}.pt")


def evaluate(checkpoint_path: str, num_episodes: int = 10, device: str = None):
    """
    Evaluate a trained policy.

    Args:
        checkpoint_path: Path to checkpoint
        num_episodes: Number of episodes to evaluate
        device: Device to use (cuda/cpu/mps), auto-detect if None
    """
    # Action mapping
    action_map = {
        'backward': np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'forward': np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'left': np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'right': np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'up': np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
        'down': np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]),
        'close gripper': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        'open gripper': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    }

    # Create environment
    print("Creating environment...")
    env = create_env()

    # Create and load policy
    print(f"Loading policy from {checkpoint_path}...")
    model_path = os.path.join(os.path.dirname(__file__), "models/Qwen2-VL-2B")
    policy = VLMRLPolicy(model_path, action_map, use_lora=True, device=device)
    policy.load(checkpoint_path)
    policy.eval()

    # Evaluate
    print(f"\nEvaluating for {num_episodes} episodes...")
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done and episode_length < 200:
            image = obs['robot0_eye_in_hand_image']

            # Select action deterministically
            action_idx, action_name = policy.select_action(image, deterministic=True)
            action_vector = action_map[action_name]

            obs, reward, done, info = env.step(action_vector)

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1}: Reward = {episode_reward:.4f}, Length = {episode_length}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"Mean reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train VLM with PPO for robot control")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode: train or eval")
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--rollout-steps", type=int, default=128,
                        help="Steps per rollout")
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--save-freq", type=int, default=10000,
                        help="Save frequency")
    parser.add_argument("--project-name", type=str, default="vlm-robot-rl",
                        help="Wandb project name")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for evaluation")
    parser.add_argument("--num-eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu", "mps"],
                        help="Device to use (auto-detect if not specified)")

    args = parser.parse_args()

    if args.mode == "train":
        train(
            total_timesteps=args.total_timesteps,
            rollout_steps=args.rollout_steps,
            learning_rate=args.learning_rate,
            save_freq=args.save_freq,
            project_name=args.project_name,
            run_name=args.run_name,
            use_wandb=not args.no_wandb,
            device=args.device,
        )
    elif args.mode == "eval":
        if args.checkpoint is None:
            print("Error: --checkpoint is required for evaluation mode")
        else:
            evaluate(args.checkpoint, args.num_eval_episodes, args.device)
