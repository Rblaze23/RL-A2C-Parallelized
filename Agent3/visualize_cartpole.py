# visualize_cartpole.py
"""
Visualize Trained A2C Agent Playing CartPole

This script loads your saved Agent 3 models and provides:
1. Live rendering of the CartPole environment
2. Real-time value function display
3. Action probability visualization
4. Episode statistics
5. Video recording capability (with configurable speed)
6. Comparison between different seeds

Usage Examples:
    # Watch a single model play
    python visualize_cartpole.py --model agent3_results/agent3_n6_seed0.pt --episodes 3
    
    # Compare all models
    python visualize_cartpole.py --model_dir agent3_results --episodes 5
    
    # Record video at 2x speed (default for GIFs)
    python visualize_cartpole.py --model agent3_results/agent3_n6_seed0.pt --record --episodes 1
    
    # Record at 4x speed
    python visualize_cartpole.py --model agent3_results/agent3_n6_seed0.pt --record --speed 4
    
    # Slow motion playback
    python visualize_cartpole.py --model agent3_results/agent3_n6_seed0.pt --fps 30

Author: RL Researcher
Date: January 2026
"""

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch
import argparse
import os
import glob
from typing import List, Tuple, Dict
import time


# ============================================================================
# NETWORK DEFINITIONS (must match agent3.py)
# ============================================================================

class Actor(nn.Module):
    """Policy network - must match training architecture"""
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
        return self.network(state)
    
    def get_action_distribution(self, state: torch.Tensor):
        logits = self.forward(state)
        return torch.distributions.Categorical(logits=logits)


class Critic(nn.Module):
    """Value network - must match training architecture"""
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
        return self.network(state).squeeze(-1)


# ============================================================================
# VISUALIZATION AGENT
# ============================================================================

class VisualAgent:
    """
    Wrapper for trained agent that provides action selection and value estimates
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model_path = model_path
        
        # Initialize networks
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Set to evaluation mode
        self.actor.eval()
        self.critic.eval()
        
        # Extract info from path
        self.n_steps = checkpoint.get('n_steps', 6)
        self.seed = self._extract_seed_from_path(model_path)
        
        print(f"✓ Loaded model from: {model_path}")
        print(f"  n-steps: {self.n_steps}, seed: {self.seed}")
    
    def _extract_seed_from_path(self, path: str) -> int:
        """Extract seed number from model path"""
        import re
        match = re.search(r'seed(\d+)', path)
        return int(match.group(1)) if match else 0
    
    def select_action(self, state: np.ndarray, greedy: bool = True) -> Tuple[int, np.ndarray, float]:
        """
        Select action and return diagnostics
        
        Returns:
            action: Selected action (0=left, 1=right)
            action_probs: Probability distribution over actions
            value: State value estimate V(s)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action distribution
            dist = self.actor.get_action_distribution(state_tensor)
            action_probs = dist.probs.cpu().numpy()[0]
            
            if greedy:
                action = dist.probs.argmax().item()
            else:
                action = dist.sample().item()
            
            # Get value estimate
            value = self.critic(state_tensor).item()
        
        return action, action_probs, value


# ============================================================================
# LIVE CARTPOLE VISUALIZATION
# ============================================================================

class CartPoleVisualizer:
    """
    Real-time CartPole visualization with matplotlib
    Shows cart, pole, and live statistics
    """
    
    def __init__(self, agent: VisualAgent, fps: int = 60, figsize: Tuple[int, int] = (14, 8)):
        self.agent = agent
        self.fps = fps
        self.frame_delay = 1.0 / fps
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=figsize)
        gs = self.fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Main CartPole display (larger)
        self.ax_cart = self.fig.add_subplot(gs[:2, :2])
        self.ax_cart.set_xlim(-2.4, 2.4)
        self.ax_cart.set_ylim(-0.5, 2.5)
        self.ax_cart.set_aspect('equal')
        self.ax_cart.set_title('CartPole Environment', fontsize=14, fontweight='bold')
        self.ax_cart.grid(True, alpha=0.3)
        
        # Value function plot
        self.ax_value = self.fig.add_subplot(gs[0, 2])
        self.ax_value.set_title('Value Function V(s)', fontsize=10, fontweight='bold')
        self.ax_value.set_xlabel('Time Step')
        self.ax_value.set_ylabel('V(s)')
        
        # Action probabilities
        self.ax_action = self.fig.add_subplot(gs[1, 2])
        self.ax_action.set_title('Action Probabilities', fontsize=10, fontweight='bold')
        self.ax_action.set_ylim(0, 1)
        self.ax_action.set_ylabel('Probability')
        
        # Episode statistics
        self.ax_stats = self.fig.add_subplot(gs[2, :])
        self.ax_stats.axis('off')
        
        # Initialize graphics objects
        self.cart_patch = None
        self.pole_line = None
        self.value_line = None
        self.action_bars = None
        self.stats_text = None
        
        # Data storage
        self.value_history = []
        self.reward_history = []
        
    def draw_cart(self, x: float, theta: float):
        """Draw cart and pole at current position"""
        # Clear previous
        self.ax_cart.clear()
        self.ax_cart.set_xlim(-2.4, 2.4)
        self.ax_cart.set_ylim(-0.5, 2.5)
        self.ax_cart.set_aspect('equal')
        self.ax_cart.grid(True, alpha=0.3)
        
        # Draw track
        self.ax_cart.plot([-2.4, 2.4], [0, 0], 'k-', linewidth=3, alpha=0.3)
        
        # Cart dimensions
        cart_width = 0.4
        cart_height = 0.2
        
        # Draw cart
        cart = Rectangle(
            (x - cart_width/2, -cart_height/2),
            cart_width, cart_height,
            facecolor='dodgerblue',
            edgecolor='navy',
            linewidth=2
        )
        self.ax_cart.add_patch(cart)
        
        # Draw wheels
        wheel_radius = 0.05
        for wheel_x in [x - cart_width/3, x + cart_width/3]:
            wheel = plt.Circle(
                (wheel_x, -cart_height/2),
                wheel_radius,
                facecolor='black',
                edgecolor='white',
                linewidth=1
            )
            self.ax_cart.add_patch(wheel)
        
        # Pole dimensions
        pole_length = 1.0
        pole_x = x + pole_length * np.sin(theta)
        pole_y = pole_length * np.cos(theta)
        
        # Draw pole
        self.ax_cart.plot(
            [x, pole_x],
            [0, pole_y],
            color='red',
            linewidth=6,
            solid_capstyle='round'
        )
        
        # Draw pole pivot
        pivot = plt.Circle(
            (x, 0),
            0.08,
            facecolor='gold',
            edgecolor='orange',
            linewidth=2,
            zorder=10
        )
        self.ax_cart.add_patch(pivot)
        
        # Draw pole tip
        tip = plt.Circle(
            (pole_x, pole_y),
            0.06,
            facecolor='darkred',
            edgecolor='red',
            linewidth=1,
            zorder=10
        )
        self.ax_cart.add_patch(tip)
        
        # Add state information
        self.ax_cart.text(
            -2.2, 2.2,
            f'Cart Position: {x:+.3f}\nPole Angle: {theta:+.3f} rad ({np.degrees(theta):+.1f}°)',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top'
        )
    
    def update_value_plot(self):
        """Update value function history plot"""
        self.ax_value.clear()
        self.ax_value.set_title('Value Function V(s)', fontsize=10, fontweight='bold')
        self.ax_value.set_xlabel('Time Step')
        self.ax_value.set_ylabel('V(s)')
        
        if self.value_history:
            steps = range(len(self.value_history))
            self.ax_value.plot(steps, self.value_history, 'g-', linewidth=2)
            self.ax_value.fill_between(steps, self.value_history, alpha=0.3, color='green')
            self.ax_value.grid(True, alpha=0.3)
    
    def update_action_plot(self, action_probs: np.ndarray, selected_action: int):
        """Update action probability bar chart"""
        self.ax_action.clear()
        self.ax_action.set_title('Action Probabilities', fontsize=10, fontweight='bold')
        self.ax_action.set_ylim(0, 1)
        self.ax_action.set_ylabel('Probability')
        self.ax_action.set_xticks([0, 1])
        self.ax_action.set_xticklabels(['← Left', 'Right →'])
        
        colors = ['orange' if i == selected_action else 'skyblue' for i in range(2)]
        bars = self.ax_action.bar([0, 1], action_probs, color=colors, edgecolor='navy', linewidth=2)
        
        # Add probability labels on bars
        for i, (bar, prob) in enumerate(zip(bars, action_probs)):
            height = bar.get_height()
            self.ax_action.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.02,
                f'{prob:.3f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        self.ax_action.grid(True, alpha=0.3, axis='y')
    
    def update_stats(self, episode: int, step: int, total_reward: float, value: float, done: bool):
        """Update statistics display"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        status = "✓ COMPLETE" if done else "▶ ONGOING"
        status_color = 'green' if done else 'blue'
        
        stats_text = (
            f"Episode: {episode:3d}  |  "
            f"Step: {step:3d}  |  "
            f"Reward: {total_reward:6.1f}  |  "
            f"Value: {value:7.2f}  |  "
            f"Status: {status}"
        )
        
        self.ax_stats.text(
            0.5, 0.5,
            stats_text,
            ha='center',
            va='center',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=1', facecolor=status_color, alpha=0.2, edgecolor=status_color, linewidth=2),
            transform=self.ax_stats.transAxes
        )
    
    def run_episode(self, env: gym.Env, episode_num: int = 1, max_steps: int = 500):
        """
        Run and visualize one episode
        
        Returns:
            total_reward: Total reward accumulated
            num_steps: Number of steps taken
        """
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        self.value_history = []
        
        plt.ion()  # Interactive mode
        
        while not (done or truncated) and step < max_steps:
            # Select action and get diagnostics
            action, action_probs, value = self.agent.select_action(state, greedy=True)
            
            # Store value
            self.value_history.append(value)
            
            # Extract state components
            cart_pos, cart_vel, pole_angle, pole_vel = state
            
            # Update visualization
            self.draw_cart(cart_pos, pole_angle)
            self.update_value_plot()
            self.update_action_plot(action_probs, action)
            self.update_stats(episode_num, step, total_reward, value, done or truncated)
            
            # Render
            plt.draw()
            plt.pause(self.frame_delay)
            
            # Step environment
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
        
        # Final update
        _, _, value = self.agent.select_action(state, greedy=True)
        self.value_history.append(value)
        cart_pos, cart_vel, pole_angle, pole_vel = state
        
        self.draw_cart(cart_pos, pole_angle)
        self.update_value_plot()
        self.update_stats(episode_num, step, total_reward, value, True)
        plt.draw()
        plt.pause(1.0)  # Pause at end
        
        plt.ioff()
        
        return total_reward, step


# ============================================================================
# VIDEO RECORDING (WITH SPEED CONTROL)
# ============================================================================

def record_episode_video(agent: VisualAgent, save_path: str, num_episodes: int = 1, 
                        fps: int = 30, speed_multiplier: float = 2.0):
    """
    Record agent playing and save as GIF with speed control
    
    Args:
        agent: Trained agent
        save_path: Path to save GIF
        num_episodes: Number of episodes to record
        fps: Base frames per second (before speed multiplier)
        speed_multiplier: How much to speed up playback (2.0 = 2x faster)
    """
    print(f"\nRecording {num_episodes} episode(s) to video...")
    print(f"  Base FPS: {fps}, Speed: {speed_multiplier}x, Effective FPS: {fps * speed_multiplier}")
    
    # Create environment with video recording
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    
    frames = []
    episode_rewards = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # Render frame
            frame = env.render()
            frames.append(frame)
            
            # Select action
            action, _, _ = agent.select_action(state, greedy=True)
            
            # Step
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        
        episode_rewards.append(total_reward)
        print(f"  Episode {ep+1}: {total_reward:.1f} reward, {len(frames)} frames")
    
    env.close()
    
    # Save video using matplotlib animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    im = ax.imshow(frames[0])
    
    def update_frame(i):
        im.set_array(frames[i])
        ax.set_title(f'Agent 3 (n=6) | Frame {i+1}/{len(frames)} | {speed_multiplier}x Speed\nReward: {np.mean(episode_rewards):.1f}',
                     fontsize=12, fontweight='bold')
        return [im]
    
    # KEY CHANGE: Reduced interval for faster playback
    # Original: interval = 1000/fps (e.g., 33.3ms for 30fps)
    # Faster: interval = 1000/(fps * speed_multiplier) (e.g., 16.7ms for 2x speed)
    effective_interval = 1000 / (fps * speed_multiplier)
    
    print(f"  Creating animation with interval: {effective_interval:.1f}ms per frame...")
    
    anim = animation.FuncAnimation(
        fig, update_frame, frames=len(frames),
        interval=effective_interval,  # This controls playback speed!
        blit=True
    )
    
    # Save
    print(f"  Saving GIF (this may take a moment)...")
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    print(f"✓ Video saved to: {save_path}")
    print(f"  Average reward: {np.mean(episode_rewards):.1f}")
    print(f"  Duration: ~{len(frames)/fps:.1f}s at normal speed, ~{len(frames)/(fps*speed_multiplier):.1f}s at {speed_multiplier}x")


# ============================================================================
# COMPARISON MODE
# ============================================================================

def compare_models(model_paths: List[str], num_episodes: int = 5):
    """
    Compare performance of multiple models
    """
    print(f"\n{'='*70}")
    print(f"COMPARING {len(model_paths)} MODELS")
    print(f"{'='*70}\n")
    
    results = []
    
    for model_path in model_paths:
        print(f"\nEvaluating: {os.path.basename(model_path)}")
        agent = VisualAgent(model_path)
        
        env = gym.make('CartPole-v1')
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            while not (done or truncated):
                action, _, _ = agent.select_action(state, greedy=True)
                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        env.close()
        
        results.append({
            'model': os.path.basename(model_path),
            'seed': agent.seed,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths)
        })
        
        print(f"  Mean Reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
        print(f"  Range: [{np.min(episode_rewards):.1f}, {np.max(episode_rewards):.1f}]")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Seed':<6} {'Mean±Std':<20} {'Min':<8} {'Max':<8}")
    print(f"{'-'*70}")
    
    for r in results:
        print(f"{r['model']:<30} {r['seed']:<6} {r['mean_reward']:6.1f}±{r['std_reward']:5.1f}        "
              f"{r['min_reward']:7.1f}  {r['max_reward']:7.1f}")
    
    # Overall statistics
    all_means = [r['mean_reward'] for r in results]
    print(f"{'-'*70}")
    print(f"{'OVERALL':<30} {'---':<6} {np.mean(all_means):6.1f}±{np.std(all_means):5.1f}")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize trained A2C agent playing CartPole',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Watch single model
  python visualize_cartpole.py --model agent3_results/agent3_n6_seed0.pt --episodes 3
  
  # Compare all models
  python visualize_cartpole.py --model_dir agent3_results --episodes 5 --compare
  
  # Record video at 2x speed (default)
  python visualize_cartpole.py --model agent3_results/agent3_n6_seed0.pt --record
  
  # Record at 4x speed (faster GIF)
  python visualize_cartpole.py --model agent3_results/agent3_n6_seed0.pt --record --speed 4
  
  # Record at 0.5x speed (slow motion)
  python visualize_cartpole.py --model agent3_results/agent3_n6_seed0.pt --record --speed 0.5
  
  # Slow motion live playback
  python visualize_cartpole.py --model agent3_results/agent3_n6_seed0.pt --fps 30
        """
    )
    
    parser.add_argument('--model', type=str, help='Path to trained model (.pt file)')
    parser.add_argument('--model_dir', type=str, help='Directory containing multiple models')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second for visualization')
    parser.add_argument('--record', action='store_true', help='Record video to file')
    parser.add_argument('--speed', type=float, default=2.0, help='Speed multiplier for recorded videos (default: 2.0 = 2x faster)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models (with --model_dir)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Find models
    if args.model:
        model_paths = [args.model]
    elif args.model_dir:
        model_paths = sorted(glob.glob(os.path.join(args.model_dir, '*.pt')))
        if not model_paths:
            print(f"Error: No .pt files found in {args.model_dir}")
            return
    else:
        print("Error: Must specify either --model or --model_dir")
        return
    
    # Comparison mode
    if args.compare and len(model_paths) > 1:
        compare_models(model_paths, num_episodes=args.episodes)
        return
    
    # Single model visualization/recording
    model_path = model_paths[0]
    agent = VisualAgent(model_path, device=args.device)
    
    if args.record:
        # Record video
        video_path = model_path.replace('.pt', '_gameplay.gif')
        record_episode_video(agent, video_path, num_episodes=args.episodes, 
                           fps=30, speed_multiplier=args.speed)
    else:
        # Live visualization
        env = gym.make('CartPole-v1')
        visualizer = CartPoleVisualizer(agent, fps=args.fps)
        
        print(f"\n{'='*70}")
        print(f"VISUALIZING AGENT: {os.path.basename(model_path)}")
        print(f"{'='*70}\n")
        
        episode_rewards = []
        
        for ep in range(args.episodes):
            print(f"\nEpisode {ep+1}/{args.episodes}...")
            reward, steps = visualizer.run_episode(env, episode_num=ep+1)
            episode_rewards.append(reward)
            print(f"  Reward: {reward:.1f}, Steps: {steps}")
            
            if ep < args.episodes - 1:
                input("\nPress Enter for next episode...")
        
        env.close()
        
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Episodes: {args.episodes}")
        print(f"Mean Reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
        print(f"Min/Max: [{np.min(episode_rewards):.1f}, {np.max(episode_rewards):.1f}]")
        print(f"{'='*70}\n")
        
        plt.show()


if __name__ == '__main__':
    main()