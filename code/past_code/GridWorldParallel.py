import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
import matplotlib.pyplot as plt
import time

GRID_SIZE = 10

class GridWorldParallelEnv(ParallelEnv):
    """Custom GridWorld environment following PettingZoo Parallel interface with multiple moving agents"""
    metadata = {'render.modes': ['human'], "name": "gridworld_parallel_v0"}

    def __init__(self, size=(GRID_SIZE, GRID_SIZE), target=(0, 0), obstacles=None, num_moving_agents=50):
        super().__init__()
        self.size = size
        self.target = target
        self.obstacles = obstacles if obstacles else []
        self.num_moving_agents = num_moving_agents

        # Define agents
        self.agents = [f"movement_agent_{i}" for i in range(num_moving_agents)]
        self.possible_agents = self.agents[:]

        # Action spaces: discrete for movement agents
        self.action_spaces = {
            agent: spaces.Discrete(5) for agent in self.agents
        }

        # Observation spaces: position + flattened mean field
        grid_area = size[0] * size[1]
        movement_obs_size = 2 + grid_area
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(movement_obs_size,), dtype=np.float32)
            for agent in self.agents
        }

        # Movement directions: Left, Right, Stay, Up, Down
        self.actions = [(-1, 0), (1, 0), (0, 0), (0, 1), (0, -1)]

        # State variables
        self.positions = []
        self.mean_field = np.zeros(self.size, dtype=np.float32)
        self.steps = 0
        self.max_steps = 100
        self.congestion_coefficient = 1

    def reset(self, seed=42, options=None):
        random.seed(seed)
        self.agents = self.possible_agents[:]
        self.positions = []
        for _ in range(self.num_moving_agents):
            x = int(9 - random.randint(0, 2))
            y = int(9 - random.randint(0, 2))
            self.positions.append((x, y))
        self.steps = 0
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._update_mean_field()
        return self._get_obs(), self._get_info()

    def _update_mean_field(self):
        mean_field = np.zeros(self.size, dtype=np.float32)
        for x, y in self.positions:
            mean_field[x, y] += 1
        self.mean_field = mean_field / max(self.num_moving_agents, 1)

    def is_valid_position(self, pos):
        x, y = pos
        return (0 <= x < self.size[0] and
                0 <= y < self.size[1] and
                pos not in self.obstacles)

    def step(self, actions):
        """Execute a step for all agents in parallel"""
        rewards = {}
        new_positions = self.positions.copy()

        for agent, action in actions.items():
            agent_idx = int(agent.split("_")[2])
            pos = self.positions[agent_idx]

            if pos == self.target:
                rewards[agent] = 0
                continue

            dx, dy = self.actions[action]
            new_pos = (pos[0] + dx, pos[1] + dy)

            if self.is_valid_position(new_pos):
                new_positions[agent_idx] = new_pos

            # Reward: bonus for reaching target, penalty for time + congestion
            time_cost = 1
            x, y = new_positions[agent_idx]
            congestion_cost = self.mean_field[x, y] if self.is_valid_position(new_pos) else 10
            target_bonus = 15 if new_positions[agent_idx] == self.target else 0

            rewards[agent] = target_bonus - (time_cost + self.congestion_coefficient * congestion_cost)

        self.positions = new_positions
        self._update_mean_field()
        self.steps += 1

        # Check termination
        terminated = all(pos == self.target for pos in self.positions)
        truncated = self.steps >= self.max_steps
        self.terminations = {agent: terminated for agent in self.agents}
        self.truncations = {agent: truncated for agent in self.agents}

        self.rewards = rewards
        return self._get_obs(), rewards, self.terminations, self.truncations, self._get_info()

    def _get_obs(self):
        obs = {}
        flat_mean_field = self.mean_field.copy().flatten()
        for agent_idx, agent in enumerate(self.agents):
            position = np.array(self.positions[agent_idx], dtype=np.float32)
            position[0] /= self.size[0]
            position[1] /= self.size[1]
            obs[agent] = np.concatenate([position, flat_mean_field])
        return obs

    def _get_info(self):
        return {agent: {} for agent in self.agents}

    def render(self, mode="human"):
        grid_size = self.size[0]

        if not hasattr(self, "fig") or self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.colorbar = None

        self.ax.clear()
        density = self.mean_field.copy()
        masked_density = np.ma.array(density, mask=False)
        for ox, oy in self.obstacles:
            masked_density[ox, oy] = np.nan

        cmap = plt.cm.Reds
        cmap.set_bad(color="black")

        im = self.ax.imshow(masked_density, origin="upper", cmap=cmap, interpolation="nearest")

        # Mark target
        self.ax.scatter(
            self.target[1],
            self.target[0],
            marker="s",
            c="lime",
            edgecolor="black",
            s=200,
            label="Target"
        )

        self.ax.set_xticks(np.arange(-0.5, grid_size, 1))
        self.ax.set_yticks(np.arange(-0.5, grid_size, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(True, which="both", color="black", linewidth=0.5)
        self.ax.set_xlim(-0.5, grid_size - 0.5)
        self.ax.set_ylim(-0.5, grid_size - 0.5)

        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04, label="Agent density")
        else:
            self.colorbar.update_normal(im)

        plt.pause(0.01)

    def close(self):
        if hasattr(self, "fig") and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.colorbar = None


# Example usage
if __name__ == "__main__":
    env = GridWorldParallelEnv(num_moving_agents=50)
    obs, info = env.reset()
    env.render()
    for _ in range(3):
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        obs, rewards, terminations, truncations, info = env.step(actions)
        print("Rewards:", rewards)
        env.render()
        time.sleep(5)
    env.close()
