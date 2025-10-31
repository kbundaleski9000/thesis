import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback



class GridWorldEnv(env.gym):
    """Custom GridWorld environment following gymnasium interface"""
    def __init__(self, size = size, targets = targets, obstacles = obstacles, num_agents = num_agents):
        super().__init__()
        self.size = size
        self.target = targets
        self.obstacles = obstacles if obstacles else []
        self.num_agents = num_agents
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(5)  # 5 possible movements
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=0, high=max(size), shape=(2,), dtype=int),
            "mean_field": spaces.Box(low=0, high=num_agents, shape=size, dtype=int)
        })
        
        self.actions = [(-1, 0), (1, 0), (0, 0), (0, 1), (0, -1)]  # Left, Right, Stay, Up, Down
        self.positions = []
        self.steps = 0
        self.reset()


    def is_valid_position(self, pos):
        """Check if a position is within bounds and not an obstacle."""
        x, y = pos
        return (0 <= x < self.size[0] and 0 <= y < self.size[1] and pos not in self.obstacles)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize agents at random positions (top-left concentrated)
        self.positions = []
        for _ in range(self.num_agents):
            x = int(self.size - self.target)
            y = int(self.size - self.target)
            self.positions.append((x, y))
            
        self.steps = 0
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """
        Execute one time step in the environment
        """
        # Move each agent
        new_positions = []
        rewards = []
        for i, pos in enumerate(self.positions):
            if pos == self.target:
                new_positions.append(pos)  # Stay at target
                rewards.append(0)  # No reward for staying at target
                continue

            # Get action (in full implementation, each agent would have its own policy)
            action_vec = self.actions[action]

            # Move agent with stochasticity
            if random.random() > 0.9:  # 10% chance action fails
                new_pos = pos
            else:
                new_pos = (pos[0] + action_vec[0], pos[1] + action_vec[1])
                if not self.is_valid_position(new_pos):
                    new_pos = pos

            new_positions.append(new_pos)

            # Calculate reward
            time_cost = abs(new_pos[0] - self.target[0]) + abs(new_pos[1] - self.target[1])
            congestion_cost = np.sum([1 for p in new_positions if p == new_pos])
            reward = - (time_cost + 1 * congestion_cost)  # Negative cost = reward
            rewards.append(reward)

        self.positions = new_positions
        self.steps += 1

        # Calculate termination conditions
        num_at_target = sum(1 for pos in self.positions if pos == self.target)
        terminated = num_at_target == self.num_agents
        truncated = self.steps >= 100

        return self._get_obs(), sum(rewards), terminated, truncated, self._get_info()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim + GRID_SIZE**2, 64)  
        self.fc2 = nn.Linear(64, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mu):
        x = torch.cat([x, mu], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.softmax(self.fc2(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + GRID_SIZE**2 + action_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, mu, a):
        x = torch.cat([x, mu, a], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def interact_with_env(env, actor, num_steps=10, device="cpu"):
    """
    Runs a rollout in the GridWorldEnv using the custom Actor.
    
    Args:
        env: your GridWorldEnv instance.
        actor: your Actor model.
        num_steps: how many steps to run.
        device: cpu or cuda.
    """
    obs, _ = env.reset()
    
    for step in range(num_steps):
        pos = torch.tensor(obs["position"], dtype=torch.float32).unsqueeze(0).to(device)  # [1, 2]
        mean_field = torch.tensor(obs["mean_field"], dtype=torch.float32).unsqueeze(0).to(device)  # [1, grid_size^2]

        # Actor outputs a prob distribution
        with torch.no_grad():
            action_probs = actor(pos, mean_field)  # [1, action_dim]
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step}: action={action}, reward={reward:.2f}, terminated={terminated}")

        if terminated or truncated:
            print("Episode done!")
            break


