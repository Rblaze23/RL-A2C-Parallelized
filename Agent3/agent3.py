""" 
Agent 3: A2C with n-step Returns - Optimal Implementation
Reinforcement Learning Mini-Project 2

This implementation features:
- n-step advantage estimation (n=6)
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
# NEURAL NETWORKS
# ============================================================================

class Actor(nn.Module):
    """
    Policy network with 2 hidden layers (64 neurons each).
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
    
    def get_action_distribution(self, state: torch.Tensor):
        """Return categorical distribution over actions"""
        logits = self.forward(state)
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
# EXPERIENCE BUFFER FOR N-STEP RETURNS
# ============================================================================

class NStepBuffer:
    """
    Buffer for collecting n-step transitions.
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
    
    def add(self, state, action, reward, value, log_prob, done, truncated):
        """Add a transition to the buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.truncated_flags.append(truncated)
    
    def __len__(self):
        return len(self.states)
    
    def is_ready(self, episode_ended: bool) -> bool:
        """Check if buffer has n steps or episode ended"""
        return len(self) >= self.n_steps or episode_ended
    
    def compute_nstep_returns(self, next_value: float) -> Tuple[List, List, List]:
        """
        Compute n-step returns and advantages for all transitions in buffer.
        
        Key insight: Each transition gets a different horizon:
        - First transition: n-step return (if episode continues)
        - Second: (n-1)-step return
        - ...
        - Last: 1-step return
        
        Proper bootstrapping:
        - If terminated (done=True, truncated=False): bootstrap_value = 0
        - If truncated or ongoing: bootstrap_value = next_value
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
                        # Since we don't store that, we use the last available value
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
            returns,
            advantages
        )
    
    def get_data_and_clear(self, next_value: float):
        """Get n-step data and clear buffer"""
        data = self.compute_nstep_returns(next_value)
        self.reset()
        return data


# ============================================================================
# A2C AGENT WITH N-STEP RETURNS
# ============================================================================

class A2CAgentNStep:
    """
    Advantage Actor-Critic with n-step returns.
    
    Features:
    - n-step advantage estimation
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
        greedy: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action and compute value.
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Estimated state value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if greedy:
            with torch.no_grad():
                dist = self.actor.get_action_distribution(state_tensor)
                action = dist.probs.argmax().item()
                log_prob = 0.0
                value = self.critic(state_tensor).item()
        else:
            dist = self.actor.get_action_distribution(state_tensor)
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
        returns: List[float],
        advantages: List[float]
    ) -> Dict[str, float]:
        """
        Update actor and critic using n-step data.
        
        Args:
            states: List of states
            actions: List of actions
            log_probs: List of log probabilities (unused, recomputed)
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
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
            )
        
        # Get current values
        values = self.critic(states_tensor)
        
        # Get current log probs and entropy
        dist = self.actor.get_action_distribution(states_tensor)
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
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'actor_grad_norm': actor_grad_norm.item(),
            'critic_grad_norm': critic_grad_norm.item(),
            'mean_advantage': advantages_tensor.mean().item(),
            'mean_return': returns_tensor.mean().item(),
            'mean_value': values.mean().item()
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
# TRAINING FUNCTIONS
# ============================================================================

def train_episode(
    env: gym.Env,
    agent: A2CAgentNStep,
    max_steps: int = 500
) -> Dict:
    """
    Train for one episode using n-step returns.
    
    Key algorithm:
    1. Collect transitions in buffer
    2. When buffer has n steps OR episode ends:
       - Compute n-step returns for all buffered transitions
       - Update networks
       - Clear buffer
    3. Continue until episode ends
    """
    state, _ = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    
    done = False
    truncated = False
    
    update_metrics_list = []
    
    while not (done or truncated) and episode_steps < max_steps:
        # Select action
        action, log_prob, value = agent.select_action(state, greedy=False)
        
        # Step environment
        next_state, reward, done, truncated, _ = env.step(action)
        
        episode_reward += reward
        episode_steps += 1
        
        # Add to buffer
        agent.buffer.add(state, action, reward, value, log_prob, done, truncated)
        
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
            states, actions, log_probs, returns, advantages = \
                agent.buffer.get_data_and_clear(next_value)
            
            metrics = agent.update(states, actions, log_probs, returns, advantages)
            update_metrics_list.append(metrics)
        
        state = next_state
    
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
        'num_updates': len(update_metrics_list)
    }


def evaluate_agent(
    agent: A2CAgentNStep,
    num_episodes: int = 10,
    env_name: str = 'CartPole-v1'
) -> Dict:
    """
    Evaluate agent with greedy policy.
    
    Returns:
        Dictionary with evaluation statistics and value trajectories
    """
    env = gym.make(env_name)
    
    eval_rewards = []
    value_trajectories = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_values = []
        steps = 0
        
        while not (done or truncated):
            # Get value and action
            action, _, value = agent.select_action(state, greedy=True)
            episode_values.append(value)
            
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
        
        eval_rewards.append(episode_reward)
        value_trajectories.append(episode_values)
        episode_lengths.append(steps)
    
    env.close()
    
    return {
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'min_reward': np.min(eval_rewards),
        'max_reward': np.max(eval_rewards),
        'rewards': eval_rewards,
        'mean_value': np.mean([np.mean(traj) for traj in value_trajectories]),
        'value_trajectories': value_trajectories,
        'mean_episode_length': np.mean(episode_lengths)
    }


def train_agent(
    seed: int,
    n_steps: int = 6,
    max_env_steps: int = 500_000,
    eval_interval: int = 20_000,
    log_interval: int = 1_000,
    save_dir: str = './agent3_results'
) -> Dict:
    """
    Main training loop for Agent 3.
    
    Args:
        seed: Random seed
        n_steps: Number of steps for n-step returns
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
    
    # Create environment
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)
    
    # Initialize agent
    agent = A2CAgentNStep(
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
        'eval_steps': [],
        'eval_rewards': [],
        'eval_stds': [],
        'eval_value_means': [],
        'eval_value_trajectories': [],
        'updates_per_episode': []
    }
    
    total_steps = 0
    episode_count = 0
    
    print(f"\n{'='*70}")
    print(f"AGENT 3: A2C with {n_steps}-step Returns (Seed {seed})")
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
            
            if episode_data['metrics']:
                history['actor_losses'].append(episode_data['metrics']['actor_loss'])
                history['critic_losses'].append(episode_data['metrics']['critic_loss'])
                history['entropies'].append(episode_data['metrics']['entropy'])
                history['mean_advantages'].append(episode_data['metrics']['mean_advantage'])
                history['mean_returns'].append(episode_data['metrics']['mean_return'])
                history['mean_values'].append(episode_data['metrics']['mean_value'])
            
            if episode_count % 10 == 0:
                print(f"Step {total_steps:7d} | Episode {episode_count:5d} | "
                      f"Reward: {episode_data['episode_reward']:6.1f} | "
                      f"Updates: {episode_data['num_updates']:2d} | "
                      f"Critic Loss: {episode_data['metrics'].get('critic_loss', 0):.4f}")
        
        # Evaluate agent
        if total_steps % eval_interval < episode_data['episode_steps'] or total_steps >= max_env_steps:
            eval_results = evaluate_agent(agent, num_episodes=10)
            
            history['eval_steps'].append(total_steps)
            history['eval_rewards'].append(eval_results['mean_reward'])
            history['eval_stds'].append(eval_results['std_reward'])
            history['eval_value_means'].append(eval_results['mean_value'])
            history['eval_value_trajectories'].append(eval_results['value_trajectories'][0])
            
            print(f"\n>>> EVALUATION at {total_steps} steps:")
            print(f"    Mean Reward: {eval_results['mean_reward']:.1f} ± {eval_results['std_reward']:.1f}")
            print(f"    Range: [{eval_results['min_reward']:.1f}, {eval_results['max_reward']:.1f}]")
            print(f"    Mean Value: {eval_results['mean_value']:.2f}")
            print(f"    Avg Episode Length: {eval_results['mean_episode_length']:.1f}\n")
    
    # Save final model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'agent3_n{n_steps}_seed{seed}.pt')
    agent.save(model_path)
    print(f"Model saved to: {model_path}")
    
    env.close()
    return history


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comprehensive_results(
    histories: List[Dict],
    n_steps: int,
    save_dir: str,
    comparison_n1_histories: Optional[List[Dict]] = None
):
    """
    Create comprehensive visualization of training results.
    
    Args:
        histories: List of training histories (different seeds)
        n_steps: Value of n used
        save_dir: Directory to save plots
        comparison_n1_histories: Optional histories from n=1 for comparison
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Suptitle
    fig.suptitle(f'Agent 3: A2C with n={n_steps} Returns - Comprehensive Analysis',
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Helper function for multi-seed plotting
    def plot_multiseed(ax, histories, key_steps, key_values, ylabel, title,
                       color='blue', label_prefix='n='):
        # Individual seeds (transparent)
        for i, h in enumerate(histories):
            if key_values in h and len(h[key_values]) > 0:
                ax.plot(h[key_steps], h[key_values], alpha=0.25, color=color,
                       linewidth=1)
        
        # Mean with error bars
        if len(histories) > 0 and key_values in histories[0]:
            steps = histories[0][key_steps]
            values_array = np.array([h[key_values][:len(steps)] for h in histories])
            mean_vals = np.mean(values_array, axis=0)
            std_vals = np.std(values_array, axis=0)
            min_vals = np.min(values_array, axis=0)
            max_vals = np.max(values_array, axis=0)
            
            ax.plot(steps, mean_vals, color=color, linewidth=2.5,
                   label=f'{label_prefix}{n_steps} (mean)')
            ax.fill_between(steps, min_vals, max_vals, alpha=0.2, color=color)
        
        ax.set_xlabel('Training Steps', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 1. Training Rewards
    ax1 = fig.add_subplot(gs[0, 0])
    plot_multiseed(ax1, histories, 'train_steps', 'train_rewards',
                   'Episode Return', 'Training Performance (per episode)')
    
    # 2. Evaluation Rewards
    ax2 = fig.add_subplot(gs[0, 1])
    plot_multiseed(ax2, histories, 'eval_steps', 'eval_rewards',
                   'Evaluation Return', 'Evaluation Performance (10 episodes)')
    ax2.axhline(y=500, color='red', linestyle='--', linewidth=2, label='Optimal (500)')
    ax2.legend()
    
    # 3. Critic Loss
    ax3 = fig.add_subplot(gs[0, 2])
    for i, h in enumerate(histories):
        if 'critic_losses' in h and len(h['critic_losses']) > 10:
            # Smooth
            window = 20
            smoothed = np.convolve(h['critic_losses'], np.ones(window)/window, mode='valid')
            steps = h['train_steps'][window-1:window-1+len(smoothed)]
            ax3.semilogy(steps, smoothed, alpha=0.6, label=f'Seed {i+1}')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Critic Loss (log scale)')
    ax3.set_title('Critic Loss Evolution', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Actor Loss
    ax4 = fig.add_subplot(gs[0, 3])
    for i, h in enumerate(histories):
        if 'actor_losses' in h and len(h['actor_losses']) > 10:
            window = 20
            smoothed = np.convolve(h['actor_losses'], np.ones(window)/window, mode='valid')
            steps = h['train_steps'][window-1:window-1+len(smoothed)]
            ax4.plot(steps, smoothed, alpha=0.6, label=f'Seed {i+1}')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Actor Loss')
    ax4.set_title('Actor Loss Evolution', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Value Function Mean
    ax5 = fig.add_subplot(gs[1, 0])
    plot_multiseed(ax5, histories, 'eval_steps', 'eval_value_means',
                   'Mean Value Estimate', 'Value Function Convergence')
    
    # 6. Entropy
    ax6 = fig.add_subplot(gs[1, 1])
    for i, h in enumerate(histories):
        if 'entropies' in h and len(h['entropies']) > 10:
            window = 20
            smoothed = np.convolve(h['entropies'], np.ones(window)/window, mode='valid')
            steps = h['train_steps'][window-1:window-1+len(smoothed)]
            ax6.plot(steps, smoothed, alpha=0.6, label=f'Seed {i+1}')
    ax6.set_xlabel('Training Steps')
    ax6.set_ylabel('Entropy')
    ax6.set_title('Policy Entropy (Exploration)', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 7. Updates per Episode
    ax7 = fig.add_subplot(gs[1, 2])
    for i, h in enumerate(histories):
        if 'updates_per_episode' in h and len(h['updates_per_episode']) > 0:
            ax7.plot(h['train_steps'], h['updates_per_episode'], 
                    alpha=0.6, label=f'Seed {i+1}')
    ax7.set_xlabel('Training Steps')
    ax7.set_ylabel('Updates per Episode')
    ax7.set_title(f'Update Frequency (n={n_steps})', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # 8. Sample Value Trajectory
    ax8 = fig.add_subplot(gs[1, 3])
    if 'eval_value_trajectories' in histories[0] and len(histories[0]['eval_value_trajectories']) > 0:
        final_traj = histories[0]['eval_value_trajectories'][-1]
        ax8.plot(range(len(final_traj)), final_traj, 'g-', linewidth=2.5)
        ax8.fill_between(range(len(final_traj)), final_traj, alpha=0.3, color='green')
        ax8.set_xlabel('Time Step in Episode')
        ax8.set_ylabel('Value Estimate V(s)')
        ax8.set_title('Final Value Trajectory (Seed 1)', fontweight='bold')
        ax8.grid(True, alpha=0.3)
    
    # 9. Comparison with n=1 (if provided)
    ax9 = fig.add_subplot(gs[2, :2])
    if comparison_n1_histories is not None:
        # n=1
        plot_multiseed(ax9, comparison_n1_histories, 'eval_steps', 'eval_rewards',
                       'Evaluation Return', f'Performance Comparison: n=1 vs n={n_steps}',
                       color='orange', label_prefix='n=')
        # n=n_steps
        steps_n = histories[0]['eval_steps']
        values_n = np.array([h['eval_rewards'][:len(steps_n)] for h in histories])
        mean_n = np.mean(values_n, axis=0)
        std_n = np.std(values_n, axis=0)
        
        ax9.plot(steps_n, mean_n, color='blue', linewidth=2.5,
                 label=f'n={n_steps} (mean)')
        ax9.fill_between(steps_n,
                         mean_n - std_n,
                         mean_n + std_n,
                         alpha=0.2,
                         color='blue')
        ax9.axhline(y=500, color='red', linestyle='--', linewidth=2,
                    label='Optimal (500)')
        ax9.legend()
    else:
        ax9.text(0.5, 0.5, 'No n=1 comparison provided',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax9.transAxes,
                 fontsize=12)
        ax9.set_axis_off()

    # 10. Summary statistics text
    ax10 = fig.add_subplot(gs[2, 2:])
    final_rewards = [h['eval_rewards'][-1] for h in histories if len(h['eval_rewards']) > 0]
    ax10.axis('off')
    ax10.text(
        0.0, 0.9,
        f"Final Evaluation Performance (n={n_steps})\n"
        f"----------------------------------------\n"
        f"Mean Reward: {np.mean(final_rewards):.1f}\n"
        f"Std Reward:  {np.std(final_rewards):.1f}\n"
        f"Min Reward:  {np.min(final_rewards):.1f}\n"
        f"Max Reward:  {np.max(final_rewards):.1f}\n"
        f"Seeds: {len(final_rewards)}",
        fontsize=12,
        family='monospace'
    )

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'agent3_n{n_steps}_analysis_{timestamp}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

    print(f"Comprehensive plot saved to: {save_path}")


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

if __name__ == "__main__":

    SEEDS = [0, 1, 2]        # required ≥ 3 seeds
    N_STEPS = 6
    SAVE_DIR = "./agent3_results"

    histories_n6 = []

    for seed in SEEDS:
        history = train_agent(
            seed=seed,
            n_steps=N_STEPS,
            max_env_steps=500_000,
            eval_interval=20_000,
            log_interval=1_000,
            save_dir=SAVE_DIR
        )
        histories_n6.append(history)

        # Save raw history
        with open(os.path.join(SAVE_DIR, f'history_n{N_STEPS}_seed{seed}.pkl'), 'wb') as f:
            pickle.dump(history, f)

    # Optional: load n=1 histories for comparison
    histories_n1 = []
    for seed in SEEDS:
        path = os.path.join(SAVE_DIR, f'history_n1_seed{seed}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                histories_n1.append(pickle.load(f))

    # Plot results
    plot_comprehensive_results(
        histories=histories_n6,
        n_steps=N_STEPS,
        save_dir=SAVE_DIR,
        comparison_n1_histories=histories_n1 if histories_n1 else None
    )
