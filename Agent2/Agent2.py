"""
Agent 2: A2C with K=6 Parallel Workers (K=6, n=1)

This agent uses 6 parallel environments to collect data simultaneously,
reducing variance in gradient estimates and improving learning stability.

Key differences from Agent 0:
- Uses gym.vector.SyncVectorEnv for parallel environments
- Collects K=6 samples per update
- Batched forward passes for efficiency
- Must handle independent episode resets
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)


# ============================================================================
# NETWORK (Same as Agent 0)
# ============================================================================

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network with shared layers."""
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCriticNetwork, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        shared_features = self.shared(state)
        action_logits = self.actor(shared_features)
        action_probs = torch.softmax(action_logits, dim=-1)
        state_value = self.critic(shared_features)
        return action_probs, state_value



def apply_stochastic_reward(reward, mask_probability=0.9):
    """Apply 90% reward masking to test robustness."""
    if np.random.random() < mask_probability:
        return 0.0
    else:
        return reward / (1 - mask_probability)
# ============================================================================
# AGENT 2: K=6 PARALLEL WORKERS
# ============================================================================

class A2CAgent_K6:
    """
    Agent 2: A2C with K=6 parallel workers (n=1).
    
    Key implementation details:
    - Uses vectorized environments for parallel data collection
    - Batched network forward passes (more efficient)
    - Each environment can reset independently
    - Gradient computed from average of K samples
    """
    
    def __init__(self, num_envs=6, gamma=0.99, actor_lr=1e-5, critic_lr=1e-3, seed=42):
        self.num_envs = num_envs  # K = 6
        self.gamma = gamma
        self.seed = seed
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create vectorized environments (K parallel environments)
        self.envs = gym.vector.SyncVectorEnv([
            lambda: gym.make('CartPole-v1') for _ in range(num_envs)
        ])        
        state_dim = self.envs.single_observation_space.shape[0]
        action_dim = self.envs.single_action_space.n
        
        # Network
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=critic_lr)
        
        # Logging
        self.training_log = {
            'steps': [],
            'episode_returns': [],
            'actor_losses': [],
            'critic_losses': [],
            'values': []
        }
        
        self.eval_log = {
            'steps': [],
            'mean_returns': [],
            'std_returns': [],
            'value_trajectories': []
        }
        
        # Track episode returns for each environment
        self.episode_returns = np.zeros(num_envs)
    
    def select_actions(self, states, greedy=False):
        """
        Select actions for all K environments at once (batched).
        
        Args:
            states: numpy array of shape (K, state_dim)
            greedy: if True, take argmax (for evaluation)
        
        Returns:
            actions: numpy array of shape (K,)
            log_probs: tensor of shape (K,) or None
            values: numpy array of shape (K,) or None
        """
        states_tensor = torch.FloatTensor(states)
        
        if greedy:
            with torch.no_grad():
                action_probs, _ = self.network(states_tensor)
                actions = torch.argmax(action_probs, dim=1).numpy()
            return actions
        else:
            action_probs, state_values = self.network(states_tensor)
            
            # Sample actions for each environment
            dist = Categorical(action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            
            return actions.numpy(), log_probs, state_values.squeeze().detach().numpy()
    
    def compute_targets(self, rewards, next_states, terminateds, truncateds):
        """
        Compute targets for all K environments with correct bootstrapping.
        
        Args:
            rewards: array of shape (K,)
            next_states: array of shape (K, state_dim)
            terminateds: array of shape (K,)
            truncateds: array of shape (K,)
        
        Returns:
            targets: array of shape (K,)
        """
        targets = np.zeros(self.num_envs)
        
        # Get next state values for bootstrapping
        next_states_tensor = torch.FloatTensor(next_states)
        with torch.no_grad():
            _, next_values = self.network(next_states_tensor)
            next_values = next_values.squeeze().numpy()
        
        for i in range(self.num_envs):
            if terminateds[i] and not truncateds[i]:
                # True terminal state
                targets[i] = rewards[i]
            else:
                # Continuing or truncated - bootstrap
                targets[i] = rewards[i] + self.gamma * next_values[i]
        
        return targets
    
    def update(self, states, actions, log_probs, targets, values):
        """
        Perform A2C update using batch of K samples.
        
        Args:
            states: array of shape (K, state_dim)
            actions: array of shape (K,)
            log_probs: tensor of shape (K,)
            targets: array of shape (K,)
            values: array of shape (K,) - old values for advantage
        """
        states_tensor = torch.FloatTensor(states)
        targets_tensor = torch.FloatTensor(targets)
        
        # Forward pass for current values
        _, state_values = self.network(states_tensor)
        state_values = state_values.squeeze()
        
        # Compute advantages (using old values)
        advantages = targets_tensor - torch.FloatTensor(values)
        
        # Losses (averaged over K samples)
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = ((state_values - targets_tensor) ** 2).mean()
        
        total_loss = actor_loss + critic_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def train(self, max_steps=500000, eval_interval=20000, log_interval=1000):
        """
        Main training loop with K=6 parallel environments.
        
        Note: Each iteration collects K samples (one from each environment),
        so we reach max_steps K times faster in terms of iterations.
        """
        print(f"Training Agent 2 (K={self.num_envs}, n=1, seed={self.seed})")
        print(f"Max steps: {max_steps}, Eval every: {eval_interval} steps")
        print("-" * 60)
        
        # Reset all environments
        states, _ = self.envs.reset(seed=self.seed)
        self.episode_returns = np.zeros(self.num_envs)
        total_steps = 0
        
        while total_steps < max_steps:
            # Select actions for all K environments
            actions, log_probs, values = self.select_actions(states, greedy=False)
            
            # Step all environments
            next_states, rewards, terminateds, truncateds, infos = self.envs.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            rewards = np.array([apply_stochastic_reward(r) for r in rewards])
            # Compute targets with correct bootstrapping
            targets = self.compute_targets(rewards, next_states, terminateds, truncateds)
            
            # Update network (using batch of K samples)
            actor_loss, critic_loss = self.update(states, actions, log_probs, targets, values)
            
            # Logging
            self.training_log['steps'].append(total_steps)
            self.training_log['actor_losses'].append(actor_loss)
            self.training_log['critic_losses'].append(critic_loss)
            self.training_log['values'].append(np.mean(values))
            
            # Track episode returns
            self.episode_returns += rewards
            
            # Log completed episodes
            for i in range(self.num_envs):
                if dones[i]:
                    self.training_log['episode_returns'].append(self.episode_returns[i])
                    self.episode_returns[i] = 0
            
            # Print progress
            if len(self.training_log['episode_returns']) > 0 and \
               len(self.training_log['episode_returns']) % 10 == 0:
                recent = self.training_log['episode_returns'][-10:]
                if total_steps % log_interval < self.num_envs:  # Print roughly every log_interval steps
                    print(f"Step {total_steps:6d} | Episodes: {len(self.training_log['episode_returns']):4d} | "
                          f"Avg(10): {np.mean(recent):6.2f} | "
                          f"Actor: {actor_loss:7.4f} | Critic: {critic_loss:8.2f}")
            
            # Update states
            states = next_states
            
            # Increment step counter (we collect K samples per iteration)
            total_steps += self.num_envs
            
            # Evaluation
            if total_steps % eval_interval < self.num_envs:
                mean_ret, std_ret, values_traj = self.evaluate(n_episodes=10)
                self.eval_log['steps'].append(total_steps)
                self.eval_log['mean_returns'].append(mean_ret)
                self.eval_log['std_returns'].append(std_ret)
                self.eval_log['value_trajectories'].append(values_traj)
                
                print(f"\n{'='*60}")
                print(f"EVALUATION @ step {total_steps}")
                print(f"Return: {mean_ret:.2f} ± {std_ret:.2f}")
                print(f"Value:  {np.mean(values_traj):.2f}")
                print(f"{'='*60}\n")
        
        print("\nTraining complete!")
        return self.training_log, self.eval_log
    
    def evaluate(self, n_episodes=10):
        """
        Evaluate with greedy policy.
        Note: We use a separate single environment for evaluation.
        """
        eval_env = gym.make('CartPole-v1')
        returns = []
        values = []
        
        for ep in range(n_episodes):
            state, _ = eval_env.reset()
            done = False
            ep_return = 0
            ep_values = []
            
            while not done:
                # Greedy action selection
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action_probs, value = self.network(state_tensor)
                    action = torch.argmax(action_probs).item()
                    ep_values.append(value.item())
                
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_return += reward
                state = next_state
            
            returns.append(ep_return)
            if ep == 0:
                values = ep_values
        
        eval_env.close()
        return np.mean(returns), np.std(returns), values
    
    def close(self):
        """Close the vectorized environments."""
        self.envs.close()

# ============================================================================
# MULTI-SEED EXPERIMENT
# ============================================================================

def run_experiment_with_seeds(seeds=[42, 123, 456], max_steps=500000):
    """Run Agent 2 with multiple seeds."""
    results = {}
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"Running Experiment {i+1}/{len(seeds)} with seed={seed}")
        print(f"{'='*70}")
        
        agent = A2CAgent_K6(
            num_envs=6,
            gamma=0.99,
            actor_lr=1e-5,
            critic_lr=1e-3,
            seed=seed
        )
        
        training_log, eval_log = agent.train(
            max_steps=max_steps,
            eval_interval=20000,
            log_interval=1000
        )
        
        results[seed] = {
            'training_log': training_log,
            'eval_log': eval_log
        }
        
        agent.close()
        
        print(f"\nSeed {seed} complete!")
        print(f"Final eval return: {eval_log['mean_returns'][-1]:.2f} ± {eval_log['std_returns'][-1]:.2f}")
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'agent2_multi_seed_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_dir / 'agent2_multi_seed_results.pkl'}")
    
    return results


def plot_aggregated_results(results, save_dir='results'):
    """Create aggregated plots with error bars."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    seeds = list(results.keys())
    eval_steps = results[seeds[0]]['eval_log']['steps']
    
    all_eval_means = []
    for seed in seeds:
        all_eval_means.append(results[seed]['eval_log']['mean_returns'])
    
    mean_eval_returns = np.mean(all_eval_means, axis=0)
    std_eval_returns = np.std(all_eval_means, axis=0)
    
    all_values = []
    for seed in seeds:
        if results[seed]['eval_log']['value_trajectories']:
            all_values.append(results[seed]['eval_log']['value_trajectories'][-1])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Training returns
    ax = axes[0, 0]
    for seed in seeds:
        returns = results[seed]['training_log']['episode_returns']
        window = 10
        if len(returns) >= window:
            ma = np.convolve(returns, np.ones(window)/window, mode='valid')
            ax.plot(ma, alpha=0.7, label=f'Seed {seed}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('Agent 2: Training Returns (MA=10) - All Seeds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Aggregated evaluation
    ax = axes[0, 1]
    ax.errorbar(eval_steps, mean_eval_returns, yerr=std_eval_returns,
                marker='o', capsize=5, capthick=2, linewidth=2,
                label='Mean ± Std (across seeds)')
    ax.fill_between(eval_steps, 
                     mean_eval_returns - std_eval_returns,
                     mean_eval_returns + std_eval_returns,
                     alpha=0.3)
    ax.axhline(y=500, color='red', linestyle='--', label='Optimal (500)')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Return')
    ax.set_title('Agent 2: Evaluation Returns - Aggregated (3 Seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Individual evaluations
    ax = axes[0, 2]
    for seed in seeds:
        eval_means = results[seed]['eval_log']['mean_returns']
        ax.plot(eval_steps, eval_means, marker='o', alpha=0.7, label=f'Seed {seed}')
    ax.axhline(y=500, color='red', linestyle='--', label='Optimal', alpha=0.5)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Return')
    ax.set_title('Agent 2: Evaluation Returns - Individual Seeds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Actor losses
    ax = axes[1, 0]
    for seed in seeds:
        steps = results[seed]['training_log']['steps']
        losses = results[seed]['training_log']['actor_losses']
        downsample = max(1, len(losses) // 1000)
        ax.plot(steps[::downsample], losses[::downsample], 
                alpha=0.6, label=f'Seed {seed}')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Actor Loss')
    ax.set_title('Agent 2: Actor Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Critic losses
    ax = axes[1, 1]
    for seed in seeds:
        steps = results[seed]['training_log']['steps']
        losses = results[seed]['training_log']['critic_losses']
        downsample = max(1, len(losses) // 1000)
        ax.plot(steps[::downsample], losses[::downsample], 
                alpha=0.6, label=f'Seed {seed}')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Critic Loss')
    ax.set_title('Agent 2: Critic Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Value functions
    ax = axes[1, 2]
    if all_values:
        # Handle different trajectory lengths
        max_len = max(len(v) for v in all_values)
        padded_values = []
        for values in all_values:
            padded = np.full(max_len, np.nan)
            padded[:len(values)] = values
            padded_values.append(padded)
        
        # Plot individual trajectories
        for i, values in enumerate(all_values):
            ax.plot(values, alpha=0.7, label=f'Seed {seeds[i]}')
        
        # Plot mean (ignoring NaNs)
        mean_values = np.nanmean(padded_values, axis=0)
        ax.plot(mean_values, 'k--', linewidth=2, label='Mean')
        ax.axhline(y=np.nanmean(mean_values), color='red', 
                  linestyle='--', label=f'Mean: {np.nanmean(mean_values):.2f}')
    
    ax.set_xlabel('Step in Episode')
    ax.set_ylabel('V(s)')
    ax.set_title('Agent 2: Value Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'agent2_multi_seed_comprehensive.png', 
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_dir / 'agent2_multi_seed_comprehensive.png'}")
    plt.show()
    
    # Summary
    print("\n" + "="*70)
    print("AGENT 2 SUMMARY (K=6, n=1)")
    print("="*70)
    
    final_returns = [results[seed]['eval_log']['mean_returns'][-1] for seed in seeds]
    final_values = [np.mean(results[seed]['eval_log']['value_trajectories'][-1]) 
                   for seed in seeds if results[seed]['eval_log']['value_trajectories']]
    
    print(f"\nFinal Evaluation Returns:")
    print(f"  Mean: {np.mean(final_returns):.2f}")
    print(f"  Std:  {np.std(final_returns):.2f}")
    
    print(f"\nFinal Value Functions:")
    print(f"  Mean: {np.mean(final_values):.2f}")
    
    print(f"\nSuccess Rate: {sum(r >= 450 for r in final_returns)}/{len(seeds)}")
    print("="*70)


def main():
    """Main function."""
    print("="*70)
    print("Agent 2: A2C with K=6 Parallel Workers")
    print("="*70)
    
    SEEDS = [42, 123, 456]
    MAX_STEPS = 500000
    
    results_file = Path('results/agent2_multi_seed_results.pkl')
    if results_file.exists():
        print(f"\nFound existing results at {results_file}")
        response = input("Load existing results? (y/n): ")
        if response.lower() == 'y':
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            print("Loaded existing results!")
        else:
            results = run_experiment_with_seeds(SEEDS, MAX_STEPS)
    else:
        results = run_experiment_with_seeds(SEEDS, MAX_STEPS)
    
    print("\nCreating plots...")
    plot_aggregated_results(results)
    
    print("\nAgent 2 complete!")


if __name__ == "__main__":
    main()