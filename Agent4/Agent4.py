"""
Agent 4: K=6 Parallel Workers + n=6 Step Returns
COMPLETE STANDALONE VERSION

This is your Agent 4 - ready to run with multiple seeds.
Based on proven techniques with careful implementation.
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
# NETWORK
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
# AGENT 4
# ============================================================================

class Agent4:
    """
    Agent 4: A2C with K=6 parallel workers and n=6 step returns.
    
    Strategy: Collect n steps from K environments, then update.
    Each environment maintains independent buffer.
    """
    
    def __init__(self, num_envs=6, n_steps=6, gamma=0.99, lr=1e-3, seed=42):
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create parallel environments
        self.envs = gym.vector.SyncVectorEnv([
            lambda: gym.make('CartPole-v1') for _ in range(num_envs)
        ])
        
        state_dim = self.envs.single_observation_space.shape[0]
        action_dim = self.envs.single_action_space.n
        
        # Network and optimizer
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
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
        
        self.episode_returns = np.zeros(num_envs)
    
    def select_actions(self, states):
        """Select actions for all K environments."""
        states_tensor = torch.FloatTensor(states)
        action_probs, state_values = self.network(states_tensor)
        
        dist = Categorical(action_probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        return actions.numpy(), log_probs, state_values.squeeze().detach().numpy()
    
    def compute_n_step_returns(self, states_list, rewards_list, next_state, terminated, truncated):
        """
        Compute n-step returns for a sequence of states.
        
        Args:
            states_list: list of states (length up to n)
            rewards_list: list of rewards (length up to n)
            next_state: final next state
            terminated: whether final state is terminal
            truncated: whether final state is truncated
        
        Returns:
            list of returns for each state
        """
        n = len(rewards_list)
        
        # Get bootstrap value
        if terminated and not truncated:
            bootstrap_value = 0.0
        else:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            with torch.no_grad():
                _, value_tensor = self.network(next_state_tensor)
            bootstrap_value = value_tensor.item()
        
        # Compute returns backwards
        returns = []
        R = bootstrap_value
        
        for i in reversed(range(n)):
            R = rewards_list[i] + self.gamma * R
            returns.insert(0, R)
        
        return returns
    
    def train(self, max_steps=500000, eval_interval=20000, log_interval=1000):
        """Main training loop."""
        print(f"Training Agent 4 (K={self.num_envs}, n={self.n_steps}, seed={self.seed})")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"Batch size per update: {self.num_envs * self.n_steps}")
        print(f"Max steps: {max_steps}")
        print("-" * 60)
        
        states, _ = self.envs.reset(seed=self.seed)
        self.episode_returns = np.zeros(self.num_envs)
        total_steps = 0
        
        # Buffers for each environment
        env_buffers = [{'states': [], 'actions': [], 'rewards': [], 
                       'log_probs': [], 'values': [], 
                       'last_next_state': None, 'last_terminated': False, 
                       'last_truncated': False} for _ in range(self.num_envs)]
        
        while total_steps < max_steps:
            # Collect n steps from all K environments
            for step in range(self.n_steps):
                # Select actions
                actions, log_probs, values = self.select_actions(states)
                
                # Step environments
                next_states, rewards, terminateds, truncateds, _ = self.envs.step(actions)
                
                # Apply reward masking
                rewards = np.array([apply_stochastic_reward(r) for r in rewards])
                
                dones = np.logical_or(terminateds, truncateds)
                
                # Store in buffers
                for i in range(self.num_envs):
                    env_buffers[i]['states'].append(states[i])
                    env_buffers[i]['actions'].append(actions[i])
                    env_buffers[i]['rewards'].append(rewards[i])
                    env_buffers[i]['log_probs'].append(log_probs[i])
                    env_buffers[i]['values'].append(values[i])
                    env_buffers[i]['last_next_state'] = next_states[i]
                    env_buffers[i]['last_terminated'] = terminateds[i]
                    env_buffers[i]['last_truncated'] = truncateds[i]
                
                # Track episodes
                self.episode_returns += rewards
                for i in range(self.num_envs):
                    if dones[i]:
                        self.training_log['episode_returns'].append(self.episode_returns[i])
                        self.episode_returns[i] = 0
                
                states = next_states
                total_steps += self.num_envs
            
            # Now process all buffers and update
            all_states = []
            all_actions = []
            all_log_probs = []
            all_returns = []
            all_old_values = []
            
            for i in range(self.num_envs):
                buffer = env_buffers[i]
                
                if len(buffer['states']) > 0:
                    # Compute n-step returns for this environment
                    returns = self.compute_n_step_returns(
                        buffer['states'],
                        buffer['rewards'],
                        buffer['last_next_state'],
                        buffer['last_terminated'],
                        buffer['last_truncated']
                    )
                    
                    # Add to batch
                    all_states.extend(buffer['states'])
                    all_actions.extend(buffer['actions'])
                    all_log_probs.extend([lp for lp in buffer['log_probs']])
                    all_returns.extend(returns)
                    all_old_values.extend(buffer['values'])
            
            # Update network with collected batch
            if len(all_states) > 0:
                # Convert to tensors
                states_tensor = torch.FloatTensor(np.array(all_states))
                actions_tensor = torch.LongTensor(np.array(all_actions))
                log_probs_tensor = torch.stack(all_log_probs)
                returns_tensor = torch.FloatTensor(np.array(all_returns))
                old_values_array = np.array(all_old_values)
                
                # Forward pass
                _, new_values = self.network(states_tensor)
                new_values = new_values.squeeze()
                
                # Compute losses
                advantages = returns_tensor - torch.FloatTensor(old_values_array)
                actor_loss = -(log_probs_tensor * advantages).mean()
                critic_loss = ((new_values - returns_tensor) ** 2).mean()
                
                total_loss = actor_loss + critic_loss
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                # Log
                self.training_log['steps'].append(total_steps)
                self.training_log['actor_losses'].append(actor_loss.item())
                self.training_log['critic_losses'].append(critic_loss.item())
                self.training_log['values'].append(np.mean(old_values_array))
            
            # Clear buffers
            env_buffers = [{'states': [], 'actions': [], 'rewards': [], 
                           'log_probs': [], 'values': [], 
                           'last_next_state': None, 'last_terminated': False, 
                           'last_truncated': False} for _ in range(self.num_envs)]
            
            # Print progress
            if len(self.training_log['episode_returns']) > 0:
                if len(self.training_log['episode_returns']) % 20 == 0:
                    recent = self.training_log['episode_returns'][-20:]
                    if total_steps % log_interval < self.num_envs * self.n_steps:
                        print(f"Step {total_steps:7d} | Episodes: {len(self.training_log['episode_returns']):4d} | "
                              f"Avg(20): {np.mean(recent):6.2f} | "
                              f"Actor: {actor_loss.item():7.4f} | Critic: {critic_loss.item():8.2f}")
            
            # Evaluation
            if total_steps % eval_interval < self.num_envs * self.n_steps:
                mean_ret, std_ret, values_traj = self.evaluate()
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
        """Evaluate with greedy policy."""
        eval_env = gym.make('CartPole-v1')
        returns = []
        values = []
        
        for ep in range(n_episodes):
            state, _ = eval_env.reset()
            done = False
            ep_return = 0
            ep_values = []
            
            while not done:
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
        """Close environments."""
        self.envs.close()


# ============================================================================
# MULTI-SEED EXPERIMENT
# ============================================================================

def run_experiment_with_seeds(seeds=[42, 123, 456], max_steps=500000):
    """Run Agent 4 with multiple seeds."""
    results = {}
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"Running Experiment {i+1}/{len(seeds)} with seed={seed}")
        print(f"{'='*70}")
        
        agent = Agent4(
            num_envs=6,
            n_steps=6,
            gamma=0.99,
            lr=1e-3,
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
        if eval_log['mean_returns']:
            print(f"Final eval return: {eval_log['mean_returns'][-1]:.2f} ± {eval_log['std_returns'][-1]:.2f}")
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'agent4_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_dir / 'agent4_results.pkl'}")
    
    return results


def plot_results(results, save_dir='results'):
    """Create plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    seeds = list(results.keys())
    
    if not results[seeds[0]]['eval_log']['steps']:
        print("No evaluation data to plot")
        return
    
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
        window = 20
        if len(returns) >= window:
            ma = np.convolve(returns, np.ones(window)/window, mode='valid')
            ax.plot(ma, alpha=0.7, label=f'Seed {seed}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('Agent 4: Training Returns (MA=20)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Evaluation returns
    ax = axes[0, 1]
    ax.errorbar(eval_steps, mean_eval_returns, yerr=std_eval_returns,
                marker='o', capsize=5, capthick=2, linewidth=2,
                label='Mean ± Std')
    ax.fill_between(eval_steps, mean_eval_returns - std_eval_returns,
                     mean_eval_returns + std_eval_returns, alpha=0.3)
    ax.axhline(y=500, color='red', linestyle='--', label='Optimal')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Mean Return')
    ax.set_title('Agent 4: Evaluation Returns (K=6, n=6)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Individual evaluations
    ax = axes[0, 2]
    for seed in seeds:
        ax.plot(eval_steps, results[seed]['eval_log']['mean_returns'],
                marker='o', alpha=0.7, label=f'Seed {seed}')
    ax.axhline(y=500, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Mean Return')
    ax.set_title('Agent 4: Individual Seeds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Actor loss
    ax = axes[1, 0]
    for seed in seeds:
        steps = results[seed]['training_log']['steps']
        losses = results[seed]['training_log']['actor_losses']
        downsample = max(1, len(losses) // 1000)
        ax.plot(steps[::downsample], losses[::downsample], 
                alpha=0.6, label=f'Seed {seed}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Actor Loss')
    ax.set_title('Agent 4: Actor Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Critic loss
    ax = axes[1, 1]
    for seed in seeds:
        steps = results[seed]['training_log']['steps']
        losses = results[seed]['training_log']['critic_losses']
        downsample = max(1, len(losses) // 1000)
        ax.plot(steps[::downsample], losses[::downsample], 
                alpha=0.6, label=f'Seed {seed}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Critic Loss')
    ax.set_title('Agent 4: Critic Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Value functions
    ax = axes[1, 2]
    if all_values:
        max_len = max(len(v) for v in all_values)
        padded_values = []
        for values in all_values:
            padded = np.full(max_len, np.nan)
            padded[:len(values)] = values
            padded_values.append(padded)
        
        for i, values in enumerate(all_values):
            ax.plot(values, alpha=0.7, label=f'Seed {seeds[i]}')
        
        mean_values = np.nanmean(padded_values, axis=0)
        ax.plot(mean_values, 'k--', linewidth=2, label='Mean')
        ax.axhline(y=np.nanmean(mean_values), color='red', 
                  linestyle='--', label=f'Mean: {np.nanmean(mean_values):.2f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('V(s)')
    ax.set_title('Agent 4: Value Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'agent4_results.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_dir / 'agent4_results.png'}")
    plt.show()
    
    # Summary
    print("\n" + "="*70)
    print("AGENT 4 SUMMARY (K=6, n=6)")
    print("="*70)
    
    final_returns = [results[seed]['eval_log']['mean_returns'][-1] for seed in seeds if results[seed]['eval_log']['mean_returns']]
    if final_returns:
        print(f"\nFinal Evaluation Returns:")
        print(f"  Mean: {np.mean(final_returns):.2f}")
        print(f"  Std:  {np.std(final_returns):.2f}")
        print(f"\nSuccess Rate (>= 450): {sum(r >= 450 for r in final_returns)}/{len(seeds)}")
    print("="*70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    print("="*70)
    print("Agent 4: A2C with K=6 Parallel Workers and n=6 Step Returns")
    print("="*70)
    
    SEEDS = [42, 123, 456]
    MAX_STEPS = 500000
    
    results_file = Path('results/agent4_results.pkl')
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
    plot_results(results)
    
    print("\nAgent 4 complete!")


if __name__ == "__main__":
    main()