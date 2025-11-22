import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    observations: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, obs, action, reward, value, log_prob, done):
        """Add a transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        """Clear the buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Compute returns and GAE advantages.

        Args:
            last_value: Value estimate of the final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        advantages = []
        returns = []

        gae = 0
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[step]
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]

            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[step])

        return returns, advantages


class PPOTrainer:
    """PPO Trainer for VLM-based robot control."""

    def __init__(
        self,
        policy,
        learning_rate: float = 3e-5,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_epochs: int = 4,
        batch_size: int = 4,
    ):
        """
        Initialize PPO trainer.

        Args:
            policy: The VLM policy to train
            learning_rate: Learning rate for optimizer
            clip_range: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            gamma: Discount factor
            gae_lambda: GAE lambda
            n_epochs: Number of epochs per update
            batch_size: Batch size for training
        """
        self.policy = policy
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Optimizer for policy and value networks
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            eps=1e-5
        )

        self.rollout_buffer = RolloutBuffer()

    def collect_rollout(self, env, n_steps: int) -> Dict[str, float]:
        """
        Collect a rollout of n_steps.

        Args:
            env: The robosuite environment
            n_steps: Number of steps to collect

        Returns:
            stats: Dictionary of rollout statistics
        """
        self.rollout_buffer.clear()

        obs = env.reset()
        episode_rewards = []
        episode_lengths = []
        current_ep_reward = 0
        current_ep_length = 0

        for step in range(n_steps):
            # Get current image observation
            image = obs['robot0_eye_in_hand_image']

            # Get action from policy
            with torch.no_grad():
                action_idx, log_prob, _, value = self.policy.get_action_and_value(image)
                action_idx = action_idx.item()
                log_prob = log_prob.item()
                value = value.item()

            # Execute action in environment
            action_name = self.policy.action_names[action_idx]
            action_vector = self.policy.action_map[action_name]

            next_obs, reward, done, info = env.step(action_vector)

            # Store transition
            self.rollout_buffer.add(
                obs=image,
                action=action_idx,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done
            )

            current_ep_reward += reward
            current_ep_length += 1

            obs = next_obs

            if done:
                episode_rewards.append(current_ep_reward)
                episode_lengths.append(current_ep_length)
                current_ep_reward = 0
                current_ep_length = 0
                obs = env.reset()

        # Compute value of last state for GAE
        image = obs['robot0_eye_in_hand_image']
        with torch.no_grad():
            last_value = self.policy.get_value(image).item()

        # Compute returns and advantages
        returns, advantages = self.rollout_buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        stats = {
            'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0.0,
            'num_episodes': len(episode_rewards),
        }

        return stats, returns, advantages

    def train_step(self, returns: List[float], advantages: List[float]) -> Dict[str, float]:
        """
        Perform PPO update.

        Args:
            returns: Computed returns
            advantages: Computed advantages

        Returns:
            losses: Dictionary of loss values
        """
        # Convert to tensors
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get old values and log probs
        old_log_probs = torch.tensor(self.rollout_buffer.log_probs, dtype=torch.float32)
        old_values = torch.tensor(self.rollout_buffer.values, dtype=torch.float32)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_updates = 0

        # Multiple epochs of updates
        for epoch in range(self.n_epochs):
            # Create mini-batches
            indices = np.arange(len(self.rollout_buffer.observations))
            np.random.shuffle(indices)

            for start_idx in range(0, len(indices), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]

                # Get batch data
                batch_obs = [self.rollout_buffer.observations[i] for i in batch_indices]
                batch_actions = [self.rollout_buffer.actions[i] for i in batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]

                # Compute new log probs, values, and entropy
                batch_log_probs = []
                batch_values = []
                batch_entropies = []

                for obs, action in zip(batch_obs, batch_actions):
                    action_tensor = torch.tensor([action], dtype=torch.long)
                    _, log_prob, entropy, value = self.policy.get_action_and_value(obs, action_tensor)

                    batch_log_probs.append(log_prob)
                    batch_values.append(value)
                    batch_entropies.append(entropy)

                batch_log_probs = torch.stack(batch_log_probs)
                batch_values = torch.stack(batch_values)
                batch_entropies = torch.stack(batch_entropies)

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                policy_loss_1 = batch_advantages * ratio
                policy_loss_2 = batch_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                value_loss = F.mse_loss(batch_values, batch_returns)

                # Entropy loss
                entropy_loss = -batch_entropies.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1

        losses = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy_loss': total_entropy_loss / n_updates,
        }

        return losses
