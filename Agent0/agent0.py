"""
Agent 0: Complete Multi-Seed Experiment
Standalone script with everything included
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

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)


# ============================================================================
# NETWORK DEFINITION
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


# ============================================================================
# AGENT DEFINITION
# ============================================================================

class A2CAgent:
    """Agent 0: Basic A2C with K=1, n=1"""
    def __init__(self, env, gamma=0.99, actor_lr=1e-5, critic_lr=1e-3, seed=42):
        self.env = env
        self.gamma = gamma
        self.seed = seed
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.action_space.seed(seed)
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
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
    
    def select_action(self, state, greedy=False):
        """Select action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        if greedy:
            with torch.no_grad():
                action_probs, _ = self.network(state_tensor)
                action = torch.argmax(action_probs).item()
            return action
        else:
            action_probs, _ = self.network(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            _, value = self.network(state_tensor)
            
            return action.item(), log_prob, value.item()
    
    def compute_target(self, reward, next_state, terminated, truncated):
        """
        Compute target return with CORRECT bootstrapping.
        """
        if terminated and not truncated:
            return reward
        else:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            with torch.no_grad():
                _, next_value = self.network(next_state_tensor)
            return reward + self.gamma * next_value.item()
    
    def update(self, state, action, log_prob, target, value):
        """Perform A2C update."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        _, state_value = self.network(state_tensor)
        state_value = state_value.squeeze()
        
        advantage = target - value
        
        actor_loss = -log_prob * advantage
        critic_loss = (state_value - target) ** 2
        
        total_loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def train(self, max_steps=100000, eval_interval=20000, log_interval=1000):
        """Main training loop."""
        print(f"Training Agent 0 (seed={self.seed})")
        print(f"Max steps: {max_steps}, Eval every: {eval_interval} steps")
        print("-" * 60)
        
        state, _ = self.env.reset(seed=self.seed)
        episode_return = 0
        total_steps = 0
        
        while total_steps < max_steps:
            action, log_prob, value = self.select_action(state, greedy=False)
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            target = self.compute_target(reward, next_state, terminated, truncated)
            
            actor_loss, critic_loss = self.update(state, action, log_prob, target, value)
            
            self.training_log['steps'].append(total_steps)
            self.training_log['actor_losses'].append(actor_loss)
            self.training_log['critic_losses'].append(critic_loss)
            self.training_log['values'].append(value)
            
            episode_return += reward
            total_steps += 1
            
            if done:
                self.training_log['episode_returns'].append(episode_return)
                
                if len(self.training_log['episode_returns']) % 10 == 0:
                    recent = self.training_log['episode_returns'][-10:]
                    print(f"Step {total_steps:6d} | Episode {len(self.training_log['episode_returns']):4d} | "
                          f"Return: {episode_return:5.1f} | Avg(10): {np.mean(recent):6.2f} | "
                          f"Actor: {actor_loss:7.4f} | Critic: {critic_loss:8.2f}")
                
                state, _ = self.env.reset()
                episode_return = 0
            else:
                state = next_state
            
            if total_steps % eval_interval == 0:
                mean_ret, std_ret, values = self.evaluate(n_episodes=10)
                self.eval_log['steps'].append(total_steps)
                self.eval_log['mean_returns'].append(mean_ret)
                self.eval_log['std_returns'].append(std_ret)
                self.eval_log['value_trajectories'].append(values)
                
                print(f"\n{'='*60}")
                print(f"EVALUATION @ step {total_steps}")
                print(f"Return: {mean_ret:.2f} ± {std_ret:.2f}")
                print(f"Value:  {np.mean(values):.2f}")
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
                action = self.select_action(state, greedy=True)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    _, value = self.network(state_tensor)
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


# ============================================================================
# MULTI-SEED EXPERIMENT
# ============================================================================

def run_experiment_with_seeds(seeds=[42, 123, 456], max_steps=100000):
    """Run Agent 0 with multiple seeds."""
    results = {}
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"Running Experiment {i+1}/{len(seeds)} with seed={seed}")
        print(f"{'='*70}")
        
        env = gym.make('CartPole-v1')
        
        agent = A2CAgent(
            env=env,
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
        
        env.close()
        
        print(f"\nSeed {seed} complete!")
        print(f"Final eval return: {eval_log['mean_returns'][-1]:.2f} ± {eval_log['std_returns'][-1]:.2f}")
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'agent0_multi_seed_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_dir / 'agent0_multi_seed_results.pkl'}")
    
    return results


def plot_aggregated_results(results, save_dir='results'):
    """Create aggregated plots with error bars."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    seeds = list(results.keys())
    eval_steps = results[seeds[0]]['eval_log']['steps']
    
    # Aggregate evaluation returns
    all_eval_means = []
    for seed in seeds:
        all_eval_means.append(results[seed]['eval_log']['mean_returns'])
    
    mean_eval_returns = np.mean(all_eval_means, axis=0)
    std_eval_returns = np.std(all_eval_means, axis=0)
    
    # Aggregate value functions
    all_values = []
    for seed in seeds:
        if results[seed]['eval_log']['value_trajectories']:
            all_values.append(results[seed]['eval_log']['value_trajectories'][-1])
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Individual training returns
    ax = axes[0, 0]
    for seed in seeds:
        returns = results[seed]['training_log']['episode_returns']
        window = 10
        if len(returns) >= window:
            ma = np.convolve(returns, np.ones(window)/window, mode='valid')
            ax.plot(ma, alpha=0.7, label=f'Seed {seed}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('Training Returns (MA=10) - All Seeds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Aggregated evaluation with error bars
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
    ax.set_title('Evaluation Returns - Aggregated (3 Seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Individual evaluation returns
    ax = axes[0, 2]
    for seed in seeds:
        eval_means = results[seed]['eval_log']['mean_returns']
        ax.plot(eval_steps, eval_means, marker='o', alpha=0.7, label=f'Seed {seed}')
    ax.axhline(y=500, color='red', linestyle='--', label='Optimal', alpha=0.5)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Return')
    ax.set_title('Evaluation Returns - Individual Seeds')
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
    ax.set_title('Actor Loss - All Seeds')
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
    ax.set_title('Critic Loss - All Seeds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Value functions
    ax = axes[1, 2]
    for i, values in enumerate(all_values):
        ax.plot(values, alpha=0.7, label=f'Seed {seeds[i]}')
    if all_values:
        mean_values = np.mean(all_values, axis=0)
        ax.plot(mean_values, 'k--', linewidth=2, label='Mean')
        ax.axhline(y=np.mean(mean_values), color='red', 
                  linestyle='--', label=f'Overall Mean: {np.mean(mean_values):.2f}')
    ax.set_xlabel('Step in Episode')
    ax.set_ylabel('V(s)')
    ax.set_title('Value Function - Last Evaluation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'agent0_multi_seed_comprehensive.png', 
                dpi=150, bbox_inches='tight')
    print(f"Comprehensive plot saved to {save_dir / 'agent0_multi_seed_comprehensive.png'}")
    plt.show()
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (Across All Seeds)")
    print("="*70)
    
    final_returns = [results[seed]['eval_log']['mean_returns'][-1] for seed in seeds]
    final_values = [np.mean(results[seed]['eval_log']['value_trajectories'][-1]) 
                   for seed in seeds if results[seed]['eval_log']['value_trajectories']]
    
    print(f"\nFinal Evaluation Returns:")
    print(f"  Mean: {np.mean(final_returns):.2f}")
    print(f"  Std:  {np.std(final_returns):.2f}")
    print(f"  Min:  {np.min(final_returns):.2f}")
    print(f"  Max:  {np.max(final_returns):.2f}")
    
    print(f"\nFinal Value Functions:")
    print(f"  Mean: {np.mean(final_values):.2f}")
    print(f"  Std:  {np.std(final_values):.2f}")
    
    print(f"\nSuccess Rate (reaching 450+ return): {sum(r >= 450 for r in final_returns)}/{len(seeds)}")
    print("="*70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to run experiments."""
    print("="*70)
    print("Agent 0: Multi-Seed Experiment")
    print("="*70)
    print("This will train Agent 0 with 3 different seeds.")
    print("Each run takes ~10-30 minutes.")
    print("="*70)
    
    SEEDS = [42, 123, 456]
    MAX_STEPS = 100000  # Use 500000 for full training
    
    # Check for existing results
    results_file = Path('results/agent0_multi_seed_results.pkl')
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
    
    # Create plots
    print("\nCreating aggregated plots...")
    plot_aggregated_results(results)
    
    print("\n" + "="*70)
    print("Agent 0 multi-seed experiment complete!")
    print("Check the 'results' folder for saved data and plots.")
    print("="*70)


if __name__ == "__main__":
    main()