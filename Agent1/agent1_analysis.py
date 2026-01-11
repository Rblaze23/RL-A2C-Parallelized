"""
Agent 1 Analysis: Complete Visualization Suite
Loads pickle files and generates comprehensive plots

Usage:
    python analyze_agent1.py
    
This will:
1. Load all 3 seed histories from agent1_results/
2. Generate comprehensive multi-panel plots
3. Save high-quality figures
4. Print statistical summary

Author: RL Researcher
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from typing import Dict, List
from datetime import datetime

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10


# ============================================================================
# DATA LOADING
# ============================================================================

def load_histories(results_dir: str = "./agent1_results") -> List[Dict]:
    """Load all pickle files from results directory"""
    histories = []
    
    for seed in [0, 1, 2]:
        pkl_path = os.path.join(results_dir, f"history_seed{seed}.pkl")
        
        if not os.path.exists(pkl_path):
            print(f"⚠ Warning: {pkl_path} not found, skipping seed {seed}")
            continue
        
        with open(pkl_path, 'rb') as f:
            history = pickle.load(f)
            histories.append(history)
            print(f"✓ Loaded seed {seed}: {len(history.get('train_steps', []))} datapoints")
    
    if len(histories) == 0:
        raise FileNotFoundError(f"No history files found in {results_dir}/")
    
    return histories


# ============================================================================
# THEORETICAL ANALYSIS
# ============================================================================

def compute_theoretical_value(gamma: float = 0.99, max_steps: int = 500) -> float:
    """
    Theoretical value for CartPole with proper truncation bootstrapping
    V(s) = sum(gamma^t) for t=0 to T-1
    """
    return (1 - gamma**max_steps) / (1 - gamma)


def compute_expected_value_stochastic(gamma: float = 0.99, max_steps: int = 500, 
                                     p_mask: float = 0.9) -> float:
    """
    Expected value with stochastic rewards (90% masked)
    E[V(s)] = sum(gamma^t * E[r_t]) = sum(gamma^t * (1-p_mask) * 1)
    """
    return (1 - gamma**max_steps) / (1 - gamma) * (1 - p_mask)


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistics(histories: List[Dict]) -> Dict:
    """Compute comprehensive statistics across seeds"""
    stats = {}
    
    # Final performance
    final_rewards = [h['train_rewards'][-1] for h in histories if len(h['train_rewards']) > 0]
    stats['final_reward_mean'] = np.mean(final_rewards)
    stats['final_reward_std'] = np.std(final_rewards)
    
    # Convergence analysis (steps to reach 475 = 95% of optimal)
    convergence_steps = []
    for h in histories:
        rewards = np.array(h['train_rewards'])
        steps = np.array(h['train_steps'])
        idx = np.where(rewards >= 475)[0]
        if len(idx) > 0:
            convergence_steps.append(steps[idx[0]])
    
    if convergence_steps:
        stats['convergence_mean'] = np.mean(convergence_steps)
        stats['convergence_std'] = np.std(convergence_steps)
    else:
        stats['convergence_mean'] = None
        stats['convergence_std'] = None
    
    # Training stability (coefficient of variation during first 80% of training)
    cv_values = []
    for h in histories:
        rewards = np.array(h['train_rewards'])
        cutoff = int(0.8 * len(rewards))
        if cutoff > 1:
            cv = np.std(rewards[:cutoff]) / (np.mean(rewards[:cutoff]) + 1e-8)
            cv_values.append(cv)
    
    stats['stability_cv'] = np.mean(cv_values) if cv_values else None
    
    # Value function convergence (if available)
    if 'mean_value' in histories[0]:
        final_values = [h['mean_value'][-1] for h in histories if len(h.get('mean_value', [])) > 0]
        if final_values:
            stats['final_value_mean'] = np.mean(final_values)
            stats['final_value_std'] = np.std(final_values)
    
    return stats


# ============================================================================
# COMPREHENSIVE PLOTTING
# ============================================================================

def create_comprehensive_plot(histories: List[Dict], save_dir: str = "./agent1_results"):
    """
    Create a single comprehensive figure with all important plots
    """
    
    # Theoretical values
    theoretical_val_deterministic = compute_theoretical_value()
    theoretical_val_stochastic = compute_expected_value_stochastic(p_mask=0.9)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Agent 1: A2C with Stochastic Rewards (n=1, K=1, p_mask=0.9)\nComprehensive Analysis Across 3 Seeds',
                 fontsize=18, fontweight='bold', y=0.98)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # ========================================================================
    # 1. TRAINING REWARDS (Top Left)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    for i, hist in enumerate(histories):
        # Smooth rewards
        window = 20
        if len(hist['train_rewards']) > window:
            smoothed = np.convolve(hist['train_rewards'], 
                                  np.ones(window)/window, mode='valid')
            steps = hist['train_steps'][window-1:window-1+len(smoothed)]
            ax1.plot(steps, smoothed, alpha=0.4, color=colors[i], linewidth=1)
    
    # Mean across seeds
    max_len = max(len(h['train_steps']) for h in histories)
    all_rewards = []
    for hist in histories:
        rewards = hist['train_rewards']
        # Pad if needed
        if len(rewards) < max_len:
            rewards = rewards + [rewards[-1]] * (max_len - len(rewards))
        all_rewards.append(rewards[:max_len])
    
    mean_rewards = np.mean(all_rewards, axis=0)
    min_rewards = np.min(all_rewards, axis=0)
    max_rewards = np.max(all_rewards, axis=0)
    steps = histories[0]['train_steps'] + list(range(len(histories[0]['train_steps']), max_len))
    
    # Smooth the mean
    window = 20
    if len(mean_rewards) > window:
        mean_smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        steps_smooth = steps[window-1:window-1+len(mean_smoothed)]
        ax1.plot(steps_smooth, mean_smoothed, 'b-', linewidth=3, label='Mean (smoothed)')
        ax1.fill_between(steps_smooth, 
                        np.convolve(min_rewards, np.ones(window)/window, mode='valid'),
                        np.convolve(max_rewards, np.ones(window)/window, mode='valid'),
                        alpha=0.2, color='blue', label='Min-Max envelope')
    
    ax1.axhline(y=500, color='red', linestyle='--', linewidth=2, label='Optimal (500)', alpha=0.7)
    ax1.axhline(y=475, color='green', linestyle=':', linewidth=1.5, label='95% threshold', alpha=0.6)
    ax1.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Episode Return (True Reward)', fontsize=11, fontweight='bold')
    ax1.set_title('Training Performance Over Time', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 550)
    
    # ========================================================================
    # 2. LEARNING CURVE (CLOSER VIEW)
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 2:])
    
    for i, hist in enumerate(histories):
        ax2.plot(hist['train_steps'], hist['train_rewards'], 
                alpha=0.3, color=colors[i], linewidth=1, label=f'Seed {i}')
    
    # Mean
    ax2.plot(steps[:len(mean_rewards)], mean_rewards, 'b-', linewidth=2.5, label='Mean')
    ax2.axhline(y=500, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlim(0, 300000)
    ax2.set_ylim(0, 520)
    ax2.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Episode Return', fontsize=11, fontweight='bold')
    ax2.set_title('Convergence Speed (First 300k steps)', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ========================================================================
    # 3. CRITIC LOSS
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    for i, hist in enumerate(histories):
        if 'critic_loss' in hist and len(hist['critic_loss']) > 0:
            window = 20
            if len(hist['critic_loss']) > window:
                smoothed = np.convolve(hist['critic_loss'], 
                                      np.ones(window)/window, mode='valid')
                steps_loss = hist['train_steps'][window-1:window-1+len(smoothed)]
                ax3.semilogy(steps_loss, smoothed, alpha=0.7, 
                           color=colors[i], linewidth=1.5, label=f'Seed {i}')
    
    ax3.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Critic Loss (log scale)', fontsize=10, fontweight='bold')
    ax3.set_title('Critic Loss Evolution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, which='both')
    
    # ========================================================================
    # 4. ACTOR LOSS
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    for i, hist in enumerate(histories):
        if 'actor_loss' in hist and len(hist['actor_loss']) > 0:
            window = 20
            if len(hist['actor_loss']) > window:
                smoothed = np.convolve(hist['actor_loss'], 
                                      np.ones(window)/window, mode='valid')
                steps_loss = hist['train_steps'][window-1:window-1+len(smoothed)]
                ax4.plot(steps_loss, smoothed, alpha=0.7, 
                        color=colors[i], linewidth=1.5, label=f'Seed {i}')
    
    ax4.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Actor Loss', fontsize=10, fontweight='bold')
    ax4.set_title('Actor Loss Evolution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # 5. ENTROPY (EXPLORATION)
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    
    for i, hist in enumerate(histories):
        if 'entropy' in hist and len(hist['entropy']) > 0:
            window = 20
            if len(hist['entropy']) > window:
                smoothed = np.convolve(hist['entropy'], 
                                      np.ones(window)/window, mode='valid')
                steps_ent = hist['train_steps'][window-1:window-1+len(smoothed)]
                ax5.plot(steps_ent, smoothed, alpha=0.7, 
                        color=colors[i], linewidth=1.5, label=f'Seed {i}')
    
    ax5.axhline(y=np.log(2), color='gray', linestyle='--', 
               linewidth=1, alpha=0.5, label='Uniform (ln(2))')
    ax5.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Policy Entropy', fontsize=10, fontweight='bold')
    ax5.set_title('Policy Entropy (Exploration)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # ========================================================================
    # 6. VALUE FUNCTION ESTIMATES
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 3])
    
    # Theoretical lines
    ax6.axhline(y=theoretical_val_deterministic, color='red', linestyle='--', 
               linewidth=2, label=f'Theoretical (det): {theoretical_val_deterministic:.1f}', alpha=0.7)
    ax6.axhline(y=theoretical_val_stochastic, color='orange', linestyle=':', 
               linewidth=2, label=f'Expected (p=0.9): {theoretical_val_stochastic:.1f}', alpha=0.7)
    
    for i, hist in enumerate(histories):
        if 'mean_value' in hist and len(hist['mean_value']) > 0:
            window = 10
            if len(hist['mean_value']) > window:
                smoothed = np.convolve(hist['mean_value'], 
                                      np.ones(window)/window, mode='valid')
                steps_val = hist['train_steps'][window-1:window-1+len(smoothed)]
                ax6.plot(steps_val, smoothed, alpha=0.7, 
                        color=colors[i], linewidth=1.5, label=f'Seed {i}')
    
    ax6.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Mean Value Estimate V(s)', fontsize=10, fontweight='bold')
    ax6.set_title('Value Function Convergence', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8, loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    # ========================================================================
    # 7. EPISODE LENGTH DISTRIBUTION
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    
    all_episode_lengths = []
    for hist in histories:
        if 'train_episode_lengths' in hist:
            all_episode_lengths.extend(hist['train_episode_lengths'])
    
    if all_episode_lengths:
        ax7.hist(all_episode_lengths, bins=50, color='skyblue', 
                edgecolor='navy', alpha=0.7)
        ax7.axvline(x=500, color='red', linestyle='--', linewidth=2, 
                   label='Max (500)', alpha=0.7)
        ax7.axvline(x=np.mean(all_episode_lengths), color='green', 
                   linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(all_episode_lengths):.1f}')
    
    ax7.set_xlabel('Episode Length (steps)', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax7.set_title('Episode Length Distribution', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # 8. REWARD DISTRIBUTION
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    
    all_train_rewards = []
    for hist in histories:
        all_train_rewards.extend(hist['train_rewards'])
    
    ax8.hist(all_train_rewards, bins=50, color='lightgreen', 
            edgecolor='darkgreen', alpha=0.7)
    ax8.axvline(x=500, color='red', linestyle='--', linewidth=2, 
               label='Optimal (500)', alpha=0.7)
    ax8.axvline(x=np.mean(all_train_rewards), color='blue', 
               linestyle='-', linewidth=2, 
               label=f'Mean: {np.mean(all_train_rewards):.1f}')
    
    ax8.set_xlabel('Episode Return', fontsize=10, fontweight='bold')
    ax8.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax8.set_title('Reward Distribution (All Episodes)', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # 9. STABILITY ANALYSIS (ROLLING STD)
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    
    window = 50
    for i, hist in enumerate(histories):
        rewards = np.array(hist['train_rewards'])
        if len(rewards) > window:
            rolling_std = [np.std(rewards[max(0, i-window):i+1]) 
                          for i in range(len(rewards))]
            ax9.plot(hist['train_steps'], rolling_std, alpha=0.7, 
                    color=colors[i], linewidth=1.5, label=f'Seed {i}')
    
    ax9.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
    ax9.set_ylabel('Rolling Std Dev (window=50)', fontsize=10, fontweight='bold')
    ax9.set_title('Training Stability', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    # ========================================================================
    # 10. STATISTICS TABLE
    # ========================================================================
    ax10 = fig.add_subplot(gs[2, 3])
    ax10.axis('off')
    
    stats = compute_statistics(histories)
    
    stats_text = f"""
    AGENT 1 STATISTICS
    ══════════════════════════════════
    
    FINAL PERFORMANCE:
    • Mean Reward: {stats['final_reward_mean']:.1f} ± {stats['final_reward_std']:.1f}
    
    CONVERGENCE:
    """
    
    if stats['convergence_mean'] is not None:
        stats_text += f"    • Steps to 95%: {stats['convergence_mean']:.0f} ± {stats['convergence_std']:.0f}\n"
    else:
        stats_text += "    • Did not reach 95% threshold\n"
    
    stats_text += f"""
    STABILITY:
    • Coeff. of Variation: {stats['stability_cv']:.4f}
    
    VALUE FUNCTION:
    """
    
    if 'final_value_mean' in stats:
        stats_text += f"    • Final V(s): {stats['final_value_mean']:.2f} ± {stats['final_value_std']:.2f}\n"
    
    stats_text += f"""    • Theoretical (det): {theoretical_val_deterministic:.2f}
    • Expected (p=0.9): {theoretical_val_stochastic:.2f}
    
    CONFIGURATION:
    • n-steps: 1 (1-step TD)
    • K-workers: 1 (single env)
    • Reward mask: 90%
    • Seeds: 3 (0, 1, 2)
    • Max steps: 500k
    """
    
    ax10.text(0.1, 0.95, stats_text, 
             transform=ax10.transAxes,
             fontsize=9,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.8))
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'agent1_comprehensive_analysis_{timestamp}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Plot saved to: {save_path}")
    
    return fig, stats


# ============================================================================
# PRINT SUMMARY
# ============================================================================

def print_summary(histories: List[Dict], stats: Dict):
    """Print detailed text summary"""
    
    print("\n" + "="*70)
    print("AGENT 1: COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  • Algorithm: A2C (1-step TD, K=1)")
    print(f"  • Stochastic Rewards: 90% masked (learning only)")
    print(f"  • Seeds: {len(histories)}")
    print(f"  • Training steps: {max(len(h['train_steps']) for h in histories)}")
    
    print(f"\nFinal Performance:")
    print(f"  • Mean Reward: {stats['final_reward_mean']:.1f} ± {stats['final_reward_std']:.1f}")
    print(f"  • Target (optimal): 500.0")
    print(f"  • Achievement: {stats['final_reward_mean']/500*100:.1f}%")
    
    if stats['convergence_mean'] is not None:
        print(f"\nConvergence Speed:")
        print(f"  • Steps to 95% performance: {stats['convergence_mean']:.0f} ± {stats['convergence_std']:.0f}")
    
    print(f"\nStability:")
    print(f"  • Coefficient of Variation: {stats['stability_cv']:.4f}")
    print(f"    (Lower is more stable)")
    
    if 'final_value_mean' in stats:
        theoretical_det = compute_theoretical_value()
        theoretical_stoch = compute_expected_value_stochastic(p_mask=0.9)
        
        print(f"\nValue Function:")
        print(f"  • Learned V(s): {stats['final_value_mean']:.2f} ± {stats['final_value_std']:.2f}")
        print(f"  • Theoretical (deterministic): {theoretical_det:.2f}")
        print(f"  • Expected (90% masked): {theoretical_stoch:.2f}")
        print(f"  • Error from expected: {abs(stats['final_value_mean'] - theoretical_stoch):.2f}")
    
    # Success criteria check
    print(f"\nProject Success Criteria:")
    success_optimal = stats['final_reward_mean'] >= 475  # 95% of optimal
    print(f"  {'✓' if success_optimal else '✗'} Reaches optimal policy: {success_optimal}")
    
    if 'final_value_mean' in stats:
        # Value should be close to expected stochastic value
        theoretical_stoch = compute_expected_value_stochastic(p_mask=0.9)
        value_correct = abs(stats['final_value_mean'] - theoretical_stoch) < 2.0
        print(f"  {'✓' if value_correct else '✗'} Correct value function: {value_correct}")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("AGENT 1 ANALYSIS: Loading and Visualizing Results")
    print("="*70 + "\n")
    
    # Load histories
    try:
        histories = load_histories("./agent1_results")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have run the training first:")
        print("  python agent1.py")
        return
    
    print(f"\n✓ Successfully loaded {len(histories)} training histories\n")
    
    # Create comprehensive plot
    print("Creating comprehensive visualization...")
    fig, stats = create_comprehensive_plot(histories, save_dir="./agent1_results")
    
    # Print summary
    print_summary(histories, stats)
    
    print("Analysis complete!")
    print("\nTo view the plot:")
    print("  • Check agent1_results/ for the PNG file")
    print("  • Or run: plt.show() in interactive mode")
    
    # Optionally show plot (comment out if running in batch)
    # plt.show()


if __name__ == '__main__':
    main()