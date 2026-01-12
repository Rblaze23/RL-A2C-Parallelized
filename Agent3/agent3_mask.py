""" 
Agent 3: A2C with n-step Returns and Action Masking
Reinforcement Learning Mini-Project 2

This implementation features:
- n-step advantage estimation (n=6)
- Action masking for invalid actions
- Proper handling of episode boundaries
- Efficient buffer management
- Comprehensive analysis and visualizations
- Multi-seed training with statistical analysis

Author: RL Researcher
Date: January 2026
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


# ============================================================================
# NEURAL NETWORKS WITH ACTION MASKING
# ============================================================================

class Actor(nn.Module):
    """
    Policy network with 2 hidden layers (64 neurons each).
    Supports action masking for invalid actions.
    Architecture: state_dim → 64 → 64 → action_dim
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_size: int = 64):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action logits"""
        return self.network(state)
    
    def get_action_distribution(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        """
        Return categorical distribution over actions with optional masking.
        
        Args:
            state: State tensor
            action_mask: Boolean tensor where True = valid action, False = invalid
                        Shape: (batch_size, action_dim) or (action_dim,)
        """
        logits = self.forward(state)
        
        # Apply action masking
        if action_mask is not None:
            # Set logits of invalid actions to very negative value
            # This ensures they have near-zero probability
            masked_logits = logits.clone()
            masked_logits[~action_mask] = -1e8
            return torch.distributions.Categorical(logits=masked_logits)
        else:
            return torch.distributions.Categorical(logits=logits)


class Critic(nn.Module):
    """
    Value network with 2 hidden layers (64 neurons each).
    Architecture: state_dim → 64 → 64 → 1
    """
    def __init__(self, state_dim: int = 4, hidden_size: int = 64):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return state value V(s)"""
        return self.network(state).squeeze(-1)


# ============================================================================
# EXPERIENCE BUFFER FOR N-STEP RETURNS WITH MASKING
# ============================================================================

class NStepBuffer:
    """
    Buffer for collecting n-step transitions with action masking support.
    Handles proper bootstrapping at episode boundaries.
    """
    def __init__(self, n_steps: int = 6, gamma: float = 0.99):
        self.n_steps = n_steps
        self.gamma = gamma
        self.reset()
    
    def reset(self):
        """Clear the buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.truncated_flags = []
        self.action_masks = []  # Store action masks
    
    def add(self, state, action, reward, value, log_prob, done, truncated, action_mask=None):
        """Add a transition to the buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.truncated_flags.append(truncated)
        self.action_masks.append(action_mask)
    
    def __len__(self):
        return len(self.states)
    
    def is_ready(self, episode_ended: bool) -> bool:
        """Check if buffer has n steps or episode ended"""
        return len(self) >= self.n_steps or episode_ended
    
    def compute_nstep_returns(self, next_value: float) -> Tuple[List, List, List, List, List]:
        """
        Compute n-step returns and advantages for all transitions in buffer.
        
        Returns:
            states, actions, log_probs, action_masks, returns, advantages
        """
        returns = []
        advantages = []
        
        buffer_size = len(self)
        
        for i in range(buffer_size):
            # Determine how many steps ahead we can look from position i
            steps_ahead = min(self.n_steps, buffer_size - i)
            
            # Compute n-step return for this position
            G = 0.0
            
            # Accumulate rewards
            for j in range(steps_ahead):
                idx = i + j
                G += (self.gamma ** j) * self.rewards[idx]
                
                # Check if episode ended within our lookahead
                if self.dones[idx] or self.truncated_flags[idx]:
                    # Episode ended at step idx
                    if self.dones[idx] and not self.truncated_flags[idx]:
                        # Terminal state - no bootstrap
                        bootstrap = 0.0
                    else:
                        # Truncated - bootstrap with value at termination
                        bootstrap = 0.0 if idx == buffer_size - 1 else self.values[idx]
                    
                    G += (self.gamma ** (j + 1)) * bootstrap
                    break
            else:
                # Didn't hit episode end - bootstrap with next_value
                G += (self.gamma ** steps_ahead) * next_value
            
            returns.append(G)
            advantages.append(G - self.values[i])
        
        return (
            self.states.copy(),
            self.actions.copy(),
            self.log_probs.copy(),
            self.action_masks.copy(),
            returns,
            advantages
        )
    
    def get_data_and_clear(self, next_value: float):
        """Get n-step data and clear buffer"""
        data = self.compute_nstep_returns(next_value)
        self.reset()
        return data


# ============================================================================
# ENVIRONMENT WRAPPER FOR ACTION MASKING
# ============================================================================

class ActionMaskWrapper(gym.Wrapper):
    """
    Wrapper that adds action masking functionality to environments.
    
    For CartPole, we'll add a simple masking rule as an example:
    - Mask actions that would accelerate the cart beyond safe velocity
    """
    def __init__(self, env, mask_threshold: float = 0.4):
        super().__init__(env)
        self.mask_threshold = mask_threshold
        self.action_dim = env.action_space.n
    
    def get_action_mask(self, state: np.ndarray) -> np.ndarray:
        """
        Get action mask for current state.
        Returns boolean array where True = valid action.
        
        For CartPole: mask actions that push cart in direction of high velocity
        State: [cart_pos, cart_vel, pole_angle, pole_vel]
        """
        mask = np.ones(self.action_dim, dtype=bool)
        
        # Simple masking rule: if cart velocity is high, mask action that increases it
        cart_velocity = state[1]
        
        if abs(cart_velocity) > self.mask_threshold:
            if cart_velocity > 0:
                # Moving right fast - mask right action (1)
                mask[1] = False
            else:
                # Moving left fast - mask left action (0)
                mask[0] = False
        
        return mask
    
    def reset(self, **kwargs):
        """Reset environment and return initial state with action mask"""
        state, info = self.env.reset(**kwargs)
        info['action_mask'] = self.get_action_mask(state)
        return state, info
    
    def step(self, action):
        """Step environment and include action mask in info"""
        state, reward, done, truncated, info = self.env.step(action)
        info['action_mask'] = self.get_action_mask(state)
        return state, reward, done, truncated, info


# ============================================================================
# A2C AGENT WITH N-STEP RETURNS AND ACTION MASKING
# ============================================================================

class A2CAgentNStepMasked:
    """
    Advantage Actor-Critic with n-step returns and action masking.
    
    Features:
    - n-step advantage estimation
    - Action masking for invalid actions
    - Proper episode boundary handling
    - Gradient clipping for stability
    - Comprehensive metrics logging
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        n_steps: int = 6,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_steps = n_steps
        self.gamma = gamma
        self.device = device
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Buffer for n-step collection
        self.buffer = NStepBuffer(n_steps=n_steps, gamma=gamma)
        
        # Metrics
        self.metrics_history = defaultdict(list)
    
    def select_action(
        self, 
        state: np.ndarray, 
        action_mask: Optional[np.ndarray] = None,
        greedy: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action with optional masking and compute value.
        
        Args:
            state: Current state
            action_mask: Boolean array indicating valid actions
            greedy: If True, select best valid action deterministically
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Estimated state value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Convert action mask to tensor if provided
        mask_tensor = None
        if action_mask is not None:
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
        
        if greedy:
            with torch.no_grad():
                dist = self.actor.get_action_distribution(state_tensor, mask_tensor)
                action = dist.probs.argmax().item()
                log_prob = 0.0
                value = self.critic(state_tensor).item()
        else:
            dist = self.actor.get_action_distribution(state_tensor, mask_tensor)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor).item()
            
            with torch.no_grad():
                value = self.critic(state_tensor).item()
        
        return action, log_prob, value
    
    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.critic(state_tensor).item()
    
    def update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        log_probs: List[float],
        action_masks: List[Optional[np.ndarray]],
        returns: List[float],
        advantages: List[float]
    ) -> Dict[str, float]:
        """
        Update actor and critic using n-step data with action masking.
        
        Args:
            states: List of states
            actions: List of actions
            log_probs: List of log probabilities (unused, recomputed)
            action_masks: List of action masks (None if no masking)
            returns: List of n-step returns (targets for critic)
            advantages: List of advantages (for actor)
        
        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Convert action masks
        masks_tensor = None
        if action_masks[0] is not None:
            masks_tensor = torch.BoolTensor(np.array(action_masks)).to(self.device)
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
            )
        
        # Get current values
        values = self.critic(states_tensor)
        
        # Get current log probs and entropy with masking
        dist = self.actor.get_action_distribution(states_tensor, masks_tensor)
        log_probs_new = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()
        
        # Critic loss: MSE between values and returns
        critic_loss = nn.MSELoss()(values, returns_tensor)
        
        # Actor loss: Policy gradient with advantages
        actor_loss = -(log_probs_new * advantages_tensor.detach()).mean()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), max_norm=10.0
        )
        self.critic_optimizer.step()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=10.0
        )
        self.actor_optimizer.step()
        
        # Calculate masking statistics
        if masks_tensor is not None:
            valid_actions_per_state = masks_tensor.float().sum(dim=1).mean().item()
        else:
            valid_actions_per_state = self.action_dim
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'actor_grad_norm': actor_grad_norm.item(),
            'critic_grad_norm': critic_grad_norm.item(),
            'mean_advantage': advantages_tensor.mean().item(),
            'mean_return': returns_tensor.mean().item(),
            'mean_value': values.mean().item(),
            'avg_valid_actions': valid_actions_per_state
        }
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'n_steps': self.n_steps
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


# ============================================================================
# TRAINING FUNCTIONS WITH ACTION MASKING
# ============================================================================

def train_episode(
    env: gym.Env,
    agent: A2CAgentNStepMasked,
    max_steps: int = 500
) -> Dict:
    """
    Train for one episode using n-step returns with action masking.
    
    Key algorithm:
    1. Collect transitions in buffer (with action masks)
    2. When buffer has n steps OR episode ends:
       - Compute n-step returns for all buffered transitions
       - Update networks
       - Clear buffer
    3. Continue until episode ends
    """
    state, info = env.reset()
    action_mask = info.get('action_mask', None)
    
    episode_reward = 0.0
    episode_steps = 0
    masked_actions_count = 0
    
    done = False
    truncated = False
    
    update_metrics_list = []
    
    while not (done or truncated) and episode_steps < max_steps:
        # Select action with masking
        action, log_prob, value = agent.select_action(state, action_mask, greedy=False)
        
        # Track masked actions
        if action_mask is not None and not action_mask[action]:
            masked_actions_count += 1
        
        # Step environment
        next_state, reward, done, truncated, info = env.step(action)
        next_action_mask = info.get('action_mask', None)
        
        episode_reward += reward
        episode_steps += 1
        
        # Add to buffer with action mask
        agent.buffer.add(state, action, reward, value, log_prob, done, truncated, action_mask)
        
        # Check if we should update
        episode_ended = done or truncated
        
        if agent.buffer.is_ready(episode_ended):
            # Get next state value for bootstrapping
            if episode_ended and done and not truncated:
                # Terminal state
                next_value = 0.0
            else:
                # Truncated or buffer full (episode ongoing)
                next_value = agent.get_value(next_state)
            
            # Get n-step data and update
            states, actions, log_probs, action_masks, returns, advantages = \
                agent.buffer.get_data_and_clear(next_value)
            
            metrics = agent.update(states, actions, log_probs, action_masks, returns, advantages)
            update_metrics_list.append(metrics)
        
        state = next_state
        action_mask = next_action_mask
    
    # Average metrics across updates
    if update_metrics_list:
        avg_metrics = {
            key: np.mean([m[key] for m in update_metrics_list])
            for key in update_metrics_list[0].keys()
        }
    else:
        avg_metrics = {}
    
    return {
        'episode_reward': episode_reward,
        'episode_steps': episode_steps,
        'metrics': avg_metrics,
        'num_updates': len(update_metrics_list),
        'masked_actions_ratio': masked_actions_count / episode_steps if episode_steps > 0 else 0
    }


def evaluate_agent(
    agent: A2CAgentNStepMasked,
    num_episodes: int = 10,
    env_name: str = 'CartPole-v1',
    use_masking: bool = True
) -> Dict:
    """
    Evaluate agent with greedy policy.
    
    Returns:
        Dictionary with evaluation statistics
    """
    if use_masking:
        env = ActionMaskWrapper(gym.make(env_name))
    else:
        env = gym.make(env_name)
    
    eval_rewards = []
    value_trajectories = []
    episode_lengths = []
    masked_actions_counts = []
    
    for ep in range(num_episodes):
        state, info = env.reset()
        action_mask = info.get('action_mask', None) if use_masking else None
        
        done = False
        truncated = False
        episode_reward = 0.0
        episode_values = []
        steps = 0
        masked_count = 0
        
        while not (done or truncated):
            # Get value and action
            action, _, value = agent.select_action(state, action_mask, greedy=True)
            episode_values.append(value)
            
            if action_mask is not None and not action_mask[action]:
                masked_count += 1
            
            state, reward, done, truncated, info = env.step(action)
            action_mask = info.get('action_mask', None) if use_masking else None
            
            episode_reward += reward
            steps += 1
        
        eval_rewards.append(episode_reward)
        value_trajectories.append(episode_values)
        episode_lengths.append(steps)
        masked_actions_counts.append(masked_count)
    
    env.close()
    
    return {
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'min_reward': np.min(eval_rewards),
        'max_reward': np.max(eval_rewards),
        'rewards': eval_rewards,
        'mean_value': np.mean([np.mean(traj) for traj in value_trajectories]),
        'value_trajectories': value_trajectories,
        'mean_episode_length': np.mean(episode_lengths),
        'mean_masked_actions': np.mean(masked_actions_counts)
    }


def train_agent(
    seed: int,
    n_steps: int = 6,
    use_masking: bool = True,
    max_env_steps: int = 500_000,
    eval_interval: int = 20_000,
    log_interval: int = 1_000,
    save_dir: str = './agent3_masked_results'
) -> Dict:
    """
    Main training loop for Agent 3 with action masking.
    
    Args:
        seed: Random seed
        n_steps: Number of steps for n-step returns
        use_masking: Whether to use action masking
        max_env_steps: Total environment steps budget
        eval_interval: Steps between evaluations
        log_interval: Steps between logging
        save_dir: Directory to save results
    
    Returns:
        Training history dictionary
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment with optional masking
    if use_masking:
        env = ActionMaskWrapper(gym.make('CartPole-v1'))
    else:
        env = gym.make('CartPole-v1')
    env.reset(seed=seed)
    
    # Initialize agent
    agent = A2CAgentNStepMasked(
        state_dim=4,
        action_dim=2,
        n_steps=n_steps,
        actor_lr=1e-5,
        critic_lr=1e-3,
        gamma=0.99,
        device='cpu'
    )
    
    # Training history
    history = {
        'train_steps': [],
        'train_rewards': [],
        'train_episode_lengths': [],
        'actor_losses': [],
        'critic_losses': [],
        'entropies': [],
        'mean_advantages': [],
        'mean_returns': [],
        'mean_values': [],
        'avg_valid_actions': [],
        'masked_actions_ratios': [],
        'eval_steps': [],
        'eval_rewards': [],
        'eval_stds': [],
        'eval_value_means': [],
        'eval_value_trajectories': [],
        'eval_masked_actions': [],
        'updates_per_episode': []
    }
    
    total_steps = 0
    episode_count = 0
    
    mask_str = "WITH Masking" if use_masking else "WITHOUT Masking"
    print(f"\n{'='*70}")
    print(f"AGENT 3: A2C with {n_steps}-step Returns {mask_str} (Seed {seed})")
    print(f"{'='*70}\n")
    
    while total_steps < max_env_steps:
        # Train one episode
        episode_data = train_episode(env, agent)
        
        total_steps += episode_data['episode_steps']
        episode_count += 1
        
        # Log training progress
        if total_steps % log_interval < episode_data['episode_steps'] or total_steps >= max_env_steps:
            history['train_steps'].append(total_steps)
            history['train_rewards'].append(episode_data['episode_reward'])
            history['train_episode_lengths'].append(episode_data['episode_steps'])
            history['updates_per_episode'].append(episode_data['num_updates'])
            history['masked_actions_ratios'].append(episode_data['masked_actions_ratio'])
            
            if episode_data['metrics']:
                history['actor_losses'].append(episode_data['metrics']['actor_loss'])
                history['critic_losses'].append(episode_data['metrics']['critic_loss'])
                history['entropies'].append(episode_data['metrics']['entropy'])
                history['mean_advantages'].append(episode_data['metrics']['mean_advantage'])
                history['mean_returns'].append(episode_data['metrics']['mean_return'])
                history['mean_values'].append(episode_data['metrics']['mean_value'])
                history['avg_valid_actions'].append(episode_data['metrics']['avg_valid_actions'])
            
            if episode_count % 10 == 0:
                print(f"Step {total_steps:7d} | Episode {episode_count:5d} | "
                      f"Reward: {episode_data['episode_reward']:6.1f} | "
                      f"Masked: {episode_data['masked_actions_ratio']:.2%} | "
                      f"Critic Loss: {episode_data['metrics'].get('critic_loss', 0):.4f}")
        
        # Evaluate agent
        if total_steps % eval_interval < episode_data['episode_steps'] or total_steps >= max_env_steps:
            eval_results = evaluate_agent(agent, num_episodes=10, use_masking=use_masking)
            
            history['eval_steps'].append(total_steps)
            history['eval_rewards'].append(eval_results['mean_reward'])
            history['eval_stds'].append(eval_results['std_reward'])
            history['eval_value_means'].append(eval_results['mean_value'])
            history['eval_value_trajectories'].append(eval_results['value_trajectories'][0])
            history['eval_masked_actions'].append(eval_results['mean_masked_actions'])
            
            print(f"\n>>> EVALUATION at {total_steps} steps:")
            print(f"    Mean Reward: {eval_results['mean_reward']:.1f} ± {eval_results['std_reward']:.1f}")
            print(f"    Range: [{eval_results['min_reward']:.1f}, {eval_results['max_reward']:.1f}]")
            print(f"    Mean Value: {eval_results['mean_value']:.2f}")
            print(f"    Masked Actions: {eval_results['mean_masked_actions']:.1f}\n")
    
    # Save final model
    os.makedirs(save_dir, exist_ok=True)
    mask_suffix = "_masked" if use_masking else "_unmasked"
    model_path = os.path.join(save_dir, f'agent3_n{n_steps}{mask_suffix}_seed{seed}.pt')
    agent.save(model_path)
    print(f"Model saved to: {model_path}")
    
    env.close()
    return history


# ============================================================================
# VISUALIZATION WITH MASKING METRICS
# ============================================================================

def plot_comprehensive_results(
    histories: List[Dict],
    n_steps: int,
    save_dir: str,
    use_masking: bool = True
):
    """
    Create comprehensive visualization of training results with masking metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    mask_str = "WITH Action Masking" if use_masking else "WITHOUT Action Masking"
    fig.suptitle(f'Agent 3: A2C with n={n_steps} Returns {mask_str}',
                 fontsize=18, fontweight='bold', y=0.98)
    
    def plot_multiseed(ax, histories, key_steps, key_values, ylabel, title,
                       color='blue', label_prefix='n='):
        for i, h in enumerate(histories):
            if key_values in h and len(h[key_values]) > 0:
                ax.plot(h[key_steps], h[key_values], alpha=0.25, color=color, linewidth=1)
        
        if len(histories) > 0 and key_values in histories[0]:
            steps = histories[0][key_steps]
            values_array = np.array([h[key_values][:len(steps)] for h in histories])
            mean_vals = np.mean(values_array, axis=0)
            min_vals = np.min(values_array, axis=0)
            max_vals = np.max(values_array, axis=0)
            
            ax.plot(steps, mean_vals, color=color, linewidth=2.5, label=f'{label_prefix}{n_steps} (mean)')
            ax.fill_between(steps, min_vals, max_vals, alpha=0.2, color=color)
        
        ax.set_xlabel('Training Steps', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 1. Training Rewards
    ax1 = fig.add_subplot(gs[0, 0])
    plot_multiseed(ax1, histories, 'train_steps', 'train_rewards',
                   'Episode Return', 'Training Performance')
    
    # 2. Evaluation Rewards
    ax2 = fig.add_subplot(gs[0, 1])
    plot_multiseed(ax2, histories, 'eval_steps', 'eval_rewards',
                   'Evaluation Return', 'Evaluation Performance')
    ax2.axhline(y=500, color='red', linestyle='--', linewidth=2, label='Optimal (500)')
    ax2.legend()
    
    # 3. Masked Actions Ratio (NEW)
    ax3 = fig.add_subplot(gs[0, 2])
    if use_masking:
        plot_multiseed(ax3, histories, 'train_steps', 'masked_actions_ratios',
                       'Masked Actions Ratio', 'Action Masking Rate', color='purple')
    else:
        ax3.text(0.5, 0.5, 'No Masking Used', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_axis_off()
    
    # 4. Average Valid Actions (NEW)
    ax4 = fig.add_subplot(gs[0, 3])
    if use_masking:
        plot_multiseed(ax4, histories, 'train_steps', 'avg_valid_actions',
                       'Avg Valid Actions', 'Valid Action Space Size', color='green')
    else:
        ax4.text(0.5, 0.5, 'No Masking Used', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_axis_off()
    
    # 5. Critic Loss
    ax5 = fig.add_subplot(gs[1, 0])
    for i, h in enumerate(histories):
        if 'critic_losses' in h and len(h['critic_losses']) > 10:
            window = 20
            smoothed = np.convolve(h['critic_losses'], np.ones(window)/window, mode='valid')
            steps = h['train_steps'][window-1:window-1+len(smoothed)]
            ax5.semilogy(steps, smoothed, alpha=0.6, label=f'Seed {i+1}')
    ax5.set_xlabel('Training Steps')
    ax5.set_ylabel('Critic Loss (log)')
    ax5.set_title('Critic Loss Evolution', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Actor Loss
    ax6 = fig.add_subplot(gs[1, 1])
    for i, h in enumerate(histories):
        if 'actor_losses' in h and len(h['actor_losses']) > 10:
            window = 20
            smoothed = np.convolve(h['actor_losses'], np.ones(window)/window, mode='valid')
            steps = h['train_steps'][window-1:window-1+len(smoothed)]
            ax6.plot(steps, smoothed, alpha=0.6, label=f'Seed {i+1}')
    ax6.set_xlabel('Training Steps')
    ax6.set_ylabel('Actor Loss')
    ax6.set_title('Actor Loss Evolution', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 7. Value Function Mean
    ax7 = fig.add_subplot(gs[1, 2])
    plot_multiseed(ax7, histories, 'eval_steps', 'eval_value_means',
                   'Mean Value', 'Value Function Convergence')
    
    # 8. Entropy
    ax8 = fig.add_subplot(gs[1, 3])
    for i, h in enumerate(histories):
        if 'entropies' in h and len(h['entropies']) > 10:
            window = 20
            smoothed = np.convolve(h['entropies'], np.ones(window)/window, mode='valid')
            steps = h['train_steps'][window-1:window-1+len(smoothed)]
            ax8.plot(steps, smoothed, alpha=0.6, label=f'Seed {i+1}')
    ax8.set_xlabel('Training Steps')
    ax8.set_ylabel('Entropy')
    ax8.set_title('Policy Entropy', fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    # 9. Updates per Episode
    ax9 = fig.add_subplot(gs[2, 0])
    for i, h in enumerate(histories):
        if 'updates_per_episode' in h:
            ax9.plot(h['train_steps'], h['updates_per_episode'], alpha=0.6, label=f'Seed {i+1}')
    ax9.set_xlabel('Training Steps')
    ax9.set_ylabel('Updates/Episode')
    ax9.set_title(f'Update Frequency (n={n_steps})', fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    # 10. Value Trajectory
    ax10 = fig.add_subplot(gs[2, 1])
    if 'eval_value_trajectories' in histories[0] and len(histories[0]['eval_value_trajectories']) > 0:
        traj = histories[0]['eval_value_trajectories'][-1]
        ax10.plot(traj, 'g-', linewidth=2.5)
        ax10.fill_between(range(len(traj)), traj, alpha=0.3, color='green')
        ax10.set_xlabel('Time Step')
        ax10.set_ylabel('Value V(s)')
        ax10.set_title('Final Value Trajectory', fontweight='bold')
        ax10.grid(True, alpha=0.3)
    
    # 11. Masked Actions per Evaluation (NEW)
    ax11 = fig.add_subplot(gs[2, 2])
    if use_masking and 'eval_masked_actions' in histories[0]:
        plot_multiseed(ax11, histories, 'eval_steps', 'eval_masked_actions',
                       'Masked Actions Count', 'Eval Masked Actions', color='orange')
    else:
        ax11.text(0.5, 0.5, 'No Masking Data', ha='center', va='center', transform=ax11.transAxes)
        ax11.set_axis_off()
    
    # 12. Episode Lengths
    ax12 = fig.add_subplot(gs[2, 3])
    plot_multiseed(ax12, histories, 'train_steps', 'train_episode_lengths',
                   'Episode Length', 'Episode Length Over Time', color='brown')
    
    # 13. Summary Statistics
    ax13 = fig.add_subplot(gs[3, :2])
    final_rewards = [h['eval_rewards'][-1] for h in histories if len(h['eval_rewards']) > 0]
    final_masked = [h['masked_actions_ratios'][-1] for h in histories if len(h.get('masked_actions_ratios', [])) > 0]
    
    summary_text = f"""Final Performance Summary (n={n_steps}, Masking={use_masking})
{'='*50}
Evaluation Rewards:
  Mean:     {np.mean(final_rewards):.1f}
  Std:      {np.std(final_rewards):.1f}
  Min/Max:  {np.min(final_rewards):.1f} / {np.max(final_rewards):.1f}
  
Seeds Trained: {len(final_rewards)}
"""
    
    if use_masking and final_masked:
        summary_text += f"""
Masking Statistics:
  Avg Masked Ratio: {np.mean(final_masked):.2%}
  Masked Range:     {np.min(final_masked):.2%} - {np.max(final_masked):.2%}
"""
    
    ax13.text(0.05, 0.95, summary_text, fontsize=11, family='monospace',
              verticalalignment='top', transform=ax13.transAxes)
    ax13.axis('off')
    
    # 14. Pseudo-code
    ax14 = fig.add_subplot(gs[3, 2:])
    pseudocode = """A2C with n-step Returns & Action Masking:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Initialize: π(a|s,mask), V(s), buffer B, n
For episode in episodes:
  s ← env.reset(), B ← []
  For t = 1, 2, ... until done:
    mask ← get_action_mask(s)
    a ~ π(·|s,mask)  // sample from masked dist
    s', r, done ← env.step(a)
    B.add(s,a,r,V(s),mask)
    
    If |B| ≥ n or done:
      // Compute n-step returns for all in B
      For each (sᵢ,aᵢ,rᵢ,Vᵢ,maskᵢ) in B:
        k = min(n, |B| - i)
        Gᵢ = Σⱼ₌₀ᵏ⁻¹ γʲrᵢ₊ⱼ + γᵏV(s')
        Âᵢ = Gᵢ - Vᵢ
      
      // Update with masked distributions
      ∇θ ← Σᵢ ∇log π(aᵢ|sᵢ,maskᵢ) · Âᵢ
      ∇w ← Σᵢ (Gᵢ - V(sᵢ))²
      
      B ← []  // clear buffer
"""
    ax14.text(0.0, 1.0, pseudocode, fontsize=8.5, family='monospace',
              verticalalignment='top', transform=ax14.transAxes)
    ax14.axis('off')
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_suffix = "_masked" if use_masking else "_unmasked"
    save_path = os.path.join(save_dir, f'agent3_n{n_steps}{mask_suffix}_{timestamp}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to: {save_path}")


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

if __name__ == "__main__":
    SEEDS = [0, 1, 2]
    N_STEPS = 6
    USE_MASKING = True
    SAVE_DIR = "./agent3_masked_results"
    
    histories = []
    
    for seed in SEEDS:
        history = train_agent(
            seed=seed,
            n_steps=N_STEPS,
            use_masking=USE_MASKING,
            max_env_steps=500_000,
            eval_interval=20_000,
            log_interval=1_000,
            save_dir=SAVE_DIR
        )
        histories.append(history)
        
        mask_suffix = "_masked" if USE_MASKING else "_unmasked"
        with open(os.path.join(SAVE_DIR, f'history_n{N_STEPS}{mask_suffix}_seed{seed}.pkl'), 'wb') as f:
            pickle.dump(history, f)
    
    # Plot results
    plot_comprehensive_results(
        histories=histories,
        n_steps=N_STEPS,
        save_dir=SAVE_DIR,
        use_masking=USE_MASKING
    )