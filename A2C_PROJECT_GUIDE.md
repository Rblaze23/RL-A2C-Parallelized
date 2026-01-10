# A2C Implementation Guide for Final Project

## What You Need to Know

### 1. Key Differences: DQN vs A2C

**DQN (What you found on GitHub):**
- Value-based method
- Learns Q(s,a) - expected return for state-action pairs
- Uses experience replay buffer
- Off-policy learning
- Single network outputs Q-values for all actions

**A2C (What you need to implement):**
- Policy gradient method with value function baseline
- Has TWO components: Actor (policy π) and Critic (value V)
- On-policy learning (no experience replay)
- Collects trajectories and updates immediately
- Can use parallel workers and n-step returns

### 2. Critical Implementation Details

#### Bootstrapping at Truncation vs Termination

This is **EXTREMELY IMPORTANT** and mentioned explicitly in your assignment!

```python
# WRONG - treating truncation as termination
if done:
    return = reward  # No bootstrapping

# CORRECT - different handling
if terminated and not truncated:
    return = reward  # Terminal state, no bootstrap
elif truncated:
    return = reward + gamma * V(next_state)  # Truncated, bootstrap!
```

**Why this matters:**
- CartPole truncates at 500 steps (max episode length)
- If you DON'T bootstrap at truncation, your value function will be wrong
- The assignment asks you to analyze the value function - you'll see the difference!

#### Agent 0 Value Function (Correct Bootstrapping)

With correct bootstrapping and optimal policy:
- V(s) should converge to approximately **500** (remaining steps to truncation)
- This makes sense: you get reward=1 per step for ~500 steps

With INCORRECT bootstrapping (treating truncation as termination):
- V(s) would be much lower
- The agent thinks episodes end at 500, so it underestimates values

### 3. A2C Algorithm Overview

```
For each iteration:
    1. Collect n steps from K parallel environments:
       - Store: states, actions, rewards, dones, truncateds
    
    2. Compute returns for each step:
       - Start from the last step
       - Work backwards: R = r + γ * R (with proper bootstrapping)
    
    3. Compute advantages:
       - A(s,a) = R - V(s)  [Return - Value baseline]
    
    4. Update Actor (policy):
       - Loss = -log π(a|s) * A(s,a)
       - This increases probability of good actions (positive advantage)
    
    5. Update Critic (value function):
       - Loss = MSE(V(s), R)
       - Learn to predict returns accurately
```

### 4. Implementation Roadmap

#### Agent 0: Basic A2C (K=1, n=1)
```python
# Pseudocode
while steps < max_steps:
    # Collect 1 step
    state, _ = env.reset() if done else state
    action = select_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    
    # Compute return (1-step)
    if terminated and not truncated:
        target = reward
    else:
        target = reward + gamma * V(next_state)
    
    # Update networks
    advantage = target - V(state)
    actor_loss = -log_prob(action) * advantage
    critic_loss = (V(state) - target)^2
    
    update_networks()
```

#### Agent 1: Stochastic Rewards
```python
# Mask rewards with 90% probability
if np.random.random() < 0.9:
    reward_to_learner = 0
else:
    reward_to_learner = reward

# But still log the actual reward for metrics!
```

**Expected behavior:**
- Value function should converge to ~50 (since E[reward] = 0.1)
- Learning will be noisier (more variance in gradients)
- This demonstrates importance of variance reduction

#### Agent 2: K Workers (K=6, n=1)
```python
# Use Gymnasium's vectorized environments
envs = gym.vector.make('CartPole-v1', num_envs=K)

# Collect K steps simultaneously
states = current_states  # Shape: (K, state_dim)
actions = select_actions(states)  # Shape: (K,)
next_states, rewards, terminateds, truncateds, _ = envs.step(actions)

# Compute returns for each environment
targets = []
for i in range(K):
    if terminateds[i] and not truncateds[i]:
        targets[i] = rewards[i]
    else:
        targets[i] = rewards[i] + gamma * V(next_states[i])

# Update using batch of K samples
advantages = targets - V(states)
actor_loss = -log_probs(actions) * advantages  # Mean over K
critic_loss = MSE(V(states), targets)  # Mean over K
```

**Expected behavior:**
- More stable learning (variance reduction from K samples)
- Same wall-clock time but K times more samples
- Better gradient estimates

#### Agent 3: n-step Returns (K=1, n=6)
```python
# Collect n steps
trajectory = []
for step in range(n):
    action = select_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    trajectory.append((state, action, reward, terminated, truncated))
    
    if terminated or truncated:
        break
    state = next_state

# Compute n-step returns
returns = []
R = V(next_state) if not (terminated and not truncated) else 0

for i in reversed(range(len(trajectory))):
    state, action, reward, terminated, truncated = trajectory[i]
    
    if terminated and not truncated:
        R = reward
    else:
        R = reward + gamma * R
    
    returns.insert(0, R)

# Update using all n samples
```

**Expected behavior:**
- Less biased returns (less reliance on V approximation)
- Potentially higher variance (longer rollouts)
- Trade-off between bias and variance

#### Agent 4: Combined (K=6, n=6)
```python
# Collect n steps from K environments
# This gives you K*n samples per update!

# Most stable learning
# Can use higher learning rates safely
```

### 5. Common Pitfalls

1. **Not handling truncation correctly**
   - Always check: `if terminated and not truncated` for true termination

2. **Mixing up gradient flows**
   - Use `.detach()` when computing advantages: `advantages = returns - values.detach()`
   - Don't backprop through the advantage baseline

3. **Forgetting to reset environments**
   - In parallel environments, each can finish at different times
   - Must handle resets independently

4. **Incorrect return computation**
   - Returns should be computed BACKWARDS from the end
   - Last step's return depends on whether it's terminal or truncated

5. **Not logging enough metrics**
   - Log: returns, losses, entropy, value predictions, gradient norms
   - You need these for debugging and analysis

### 6. Logging and Plotting

```python
# During training (every 1k steps)
log_metrics = {
    'step': total_steps,
    'episode_return': episode_returns,  # When episodes finish
    'actor_loss': actor_loss,
    'critic_loss': critic_loss,
    'entropy': policy_entropy,  # For debugging
    'value_mean': values.mean(),  # For debugging
}

# During evaluation (every 20k steps)
eval_metrics = {
    'eval_return_mean': mean_return,
    'eval_return_std': std_return,
    'value_trajectory': values_over_trajectory,
}
```

**Required plots:**
1. Training returns over time (with error bars from 3 seeds)
2. Evaluation returns over time (with error bars)
3. Actor/Critic losses over time
4. Value function evolution (mean over trajectory)

### 7. Questions You'll Need to Answer

From the assignment, be prepared to answer:

1. **Agent 0**: 
   - What value does V(s) converge to with correct bootstrapping?
   - What happens without correct bootstrapping?
   - Why? (Theoretical explanation)

2. **Agent 1**:
   - What value does V(s) converge to with stochastic rewards?
   - How does value loss differ from deterministic environment?
   - Why is learning more/less stable?

3. **Agent 2**:
   - Is learning faster/slower than K=1?
   - In terms of: (a) environment steps, (b) wall-clock time
   - Why is it more/less stable?

4. **Agent 3**:
   - What value does V(s) converge to?
   - Is learning faster/slower than n=1?
   - More/less stable? Why?
   - What does n>500 remind you of? (Hint: Monte Carlo)

5. **Agent 4**:
   - Can you increase learning rates? Why?
   - Effect of combining K and n?

### 8. Installation Requirements

```bash
pip install gymnasium torch matplotlib numpy
# Optional but recommended:
pip install wandb  # For logging
pip install seaborn  # For prettier plots
```

### 9. Recommended Code Structure

```
project/
├── notebook.ipynb          # Main notebook with results
├── agents/
│   ├── a2c.py             # A2C agent class
│   ├── networks.py        # Actor-Critic networks
│   └── utils.py           # Helper functions
├── training/
│   ├── train.py           # Training loop
│   └── evaluate.py        # Evaluation code
├── plotting/
│   └── visualize.py       # Plotting functions
└── experiments/
    └── run_experiment.py  # Script to run experiments
```

### 10. Next Steps

1. ✅ Fix the DQN error (if you want to see it work)
2. ✅ Understand DQN vs A2C difference
3. ⬜ Implement Agent 0 (basic A2C)
4. ⬜ Test with correct/incorrect bootstrapping
5. ⬜ Progressively add: stochastic rewards, K workers, n-steps
6. ⬜ Run experiments with 3 different seeds
7. ⬜ Create plots and analysis
8. ⬜ Prepare video presentation

Good luck with your project! The key is to understand each component deeply so you can explain it during the Q&A.
