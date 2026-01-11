"""
Agent 1: A2C with Stochastic Rewards (n = 1, K = 1)
Reinforcement Learning Mini-Project 2

Key features:
- Reward masking with probability p=0.9 (learning only)
- True episodic returns preserved for logging
- Correct truncation vs termination bootstrapping
- Same architecture & metrics as Agent 3
- Pickle + model saving for later comparison
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from collections import defaultdict


# ============================================================================
# NETWORKS
# ============================================================================

class Actor(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)

    def dist(self, x):
        return torch.distributions.Categorical(logits=self.forward(x))


class Critic(nn.Module):
    def __init__(self, state_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# STOCHASTIC REWARD WRAPPER
# ============================================================================

class StochasticRewardWrapper(gym.Wrapper):
    """
    Reward is zeroed with probability p_mask for learning.
    True reward is kept for logging.
    """
    def __init__(self, env, p_mask=0.9, seed=None):
        super().__init__(env)
        self.p_mask = p_mask
        self.rng = np.random.default_rng(seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["true_reward"] = reward

        if self.rng.random() < self.p_mask:
            reward = 0.0

        return obs, reward, terminated, truncated, info


# ============================================================================
# A2C AGENT (1-STEP)
# ============================================================================

class A2CAgent:
    def __init__(self, actor_lr=1e-5, critic_lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.actor = Actor()
        self.critic = Critic()
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state, greedy=False):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        dist = self.actor.dist(s)

        if greedy:
            action = dist.probs.argmax().item()
            logp = 0.0
        else:
            a = dist.sample()
            action = a.item()
            logp = dist.log_prob(a).item()

        value = self.critic(s).item()
        return action, logp, value

    def value(self, state):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.critic(s).item()

    def update(self, state, action, ret, adv):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor([action])
        ret = torch.tensor([ret])
        adv = torch.tensor([adv])

        # Critic
        v = self.critic(s)
        critic_loss = nn.MSELoss()(v, ret)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        # Actor
        dist = self.actor.dist(s)
        logp = dist.log_prob(a)
        entropy = dist.entropy().mean()

        actor_loss = -(logp * adv.detach()).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "mean_advantage": adv.item(),
            "mean_return": ret.item(),
            "mean_value": v.item()
        }

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, path)


# ============================================================================
# TRAINING
# ============================================================================

def train_agent1(seed, max_steps=500_000, eval_interval=20_000, save_dir="./agent1_results"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = StochasticRewardWrapper(
        gym.make("CartPole-v1"),
        p_mask=0.9,
        seed=seed
    )
    env.reset(seed=seed)

    agent = A2CAgent()
    history = defaultdict(list)

    total_steps = 0
    episode = 0

    print(f"\n===== Agent 1 | Seed {seed} =====\n")

    while total_steps < max_steps:
        state, _ = env.reset()
        done, truncated = False, False
        ep_reward = 0.0
        ep_steps = 0
        metrics = []

        while not (done or truncated):
            action, _, value = agent.select_action(state)
            next_state, masked_r, done, truncated, info = env.step(action)

            ep_reward += info["true_reward"]
            ep_steps += 1

            if done and not truncated:
                next_value = 0.0
            else:
                next_value = agent.value(next_state)

            ret = masked_r + agent.gamma * next_value
            adv = ret - value

            metrics.append(agent.update(state, action, ret, adv))
            state = next_state

        total_steps += ep_steps
        episode += 1

        history["train_steps"].append(total_steps)
        history["train_rewards"].append(ep_reward)
        history["train_episode_lengths"].append(ep_steps)

        for k in metrics[0]:
            history[k].append(np.mean([m[k] for m in metrics]))

        if episode % 10 == 0:
            print(f"Step {total_steps:7d} | Ep {episode:4d} | Reward {ep_reward:6.1f}")

    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/history_seed{seed}.pkl", "wb") as f:
        pickle.dump(dict(history), f)

    agent.save(f"{save_dir}/agent1_seed{seed}.pt")
    env.close()

    return dict(history)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    for seed in [0, 1, 2]:
        train_agent1(seed)
