#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import copy
from typing import Callable, Dict, List, Optional

import random
import numpy as np
import heapq
import matplotlib.pyplot as plt

from benchmarl.environments.common import Task, TaskClass

from benchmarl.utils import DEVICE_TYPING

from gymnasium import spaces
from pettingzoo import ParallelEnv

from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase, PettingZooWrapper


class MyCustomEnv2(ParallelEnv):
    """Custom GridWorld environment following PettingZoo Parallel interface with multiple moving agents"""
    metadata = {'render.modes': ['human'], "name": "gridworld_parallel_v0"}

    def __init__(self, size=(10, 10), target=(0, 0), obstacles=None, num_moving_agents=50):
        super().__init__()
        self.size = size
        self.target = target
        obstacles_test = [(x, y) for x in range(1, 9) for y in range(1, 9)]
        self.obstacles = obstacles_test if obstacles_test else []
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
        self.red_lights = np.zeros(self.size, dtype=np.float32)
        self.steps = 0
        self.max_steps = 100
        self.congestion_coefficient = 10
        self.dist_map = self.comp_congested_dist_map()

    def reset(self, seed=42, options=None):
        random.seed(seed)
        self.agents = self.possible_agents[:]
        self.positions = []
        for _ in range(self.num_moving_agents):
            x = int(self.size[0] - 1 )
            y = int(self.size[1] - 1 )
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

        return (
            0 <= x < self.size[0] and
            0 <= y < self.size[1] and
            pos not in self.obstacles and
            random.uniform(0, 1) > self.red_lights[x, y]
        )
    

    def comp_congested_dist_map(self, base_cost = 1):
        """multi-source dijkstra with edge weights = base_cost + congestion_weight * mean_field[destination_cell]"""
        height, width = self.size
        dist_map = np.full((height, width), np.inf)
        obstacle_set = set(self.obstacles)  # again for perf, O(1) lookups
        # we use a priority queue by distance (smallest first). initialize:
        heap = [(0.0, self.target[0], self.target[1])]
        dist_map[self.target[0], self.target[1]] = 0.0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right as before
        visited = np.zeros((height, width), dtype=np.bool)  # could use dist_map < inf to check as well

        while heap:
            dist, x, y = heapq.heappop(heap)

            if visited[x, y]: 
                continue  # necessary for correctness
            visited[x, y] = True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # ✅ First check bounds
                if not (0 <= nx < height and 0 <= ny < width):
                    continue

                # ✅ Then check obstacle or visited
                if visited[nx, ny] or (nx, ny) in obstacle_set:
                    continue

                congestion = self.mean_field[nx, ny]
                edge_weight = base_cost 
                new_dist = dist + edge_weight

                if new_dist < dist_map[nx, ny]:
                    dist_map[nx, ny] = new_dist
                    heapq.heappush(heap, (new_dist, nx, ny))


        dist_map[dist_map == np.inf] = 100  # set unreachable cells to -1
        return dist_map


    def step(self, actions):
        """Execute a step for all agents in parallel"""
        rewards = {}
        new_positions = self.positions.copy()

        # 1️⃣ Move only active agents
        for agent, action in actions.items():
            if self.terminations.get(agent, False):  # already finished
                rewards[agent] = 0
                continue

            agent_idx = int(agent.split("_")[2])
            pos = self.positions[agent_idx]

            dx, dy = self.actions[action]
            new_pos = (pos[0] + dx, pos[1] + dy)

            if self.is_valid_position(new_pos):
                new_positions[agent_idx] = new_pos

        # Update state
        self.positions = new_positions
        self._update_mean_field()
        

        # 2️⃣ Compute rewards & update termination per agent
        for agent in self.agents:
            agent_idx = int(agent.split("_")[2])
            pos = self.positions[agent_idx]
            x, y = pos

            if pos == self.target:
                self.terminations[agent] = True
                rewards[agent] = 0  # stop penalizing at goal
            else:
                congestion_cost = self.mean_field[x, y]
                rewards[agent] = -(self.dist_map[x, y] + self.congestion_coefficient * congestion_cost)

        self.steps += 1

        # Episode truncation only
        self.truncations = {agent: (self.steps >= self.max_steps) for agent in self.agents}

        self.rewards = rewards
        return self._get_obs(), rewards, self.terminations, self.truncations, self._get_info()

    
    

    def _get_obs(self):
        obs = {}
        flat_mean_field = self.mean_field.copy().flatten()
        for agent_idx, agent in enumerate(self.agents):
            position = np.array(self.positions[agent_idx], dtype=np.float32)
            position[0] /= self.size[0]
            position[1] /= self.size[1]
            
            # -----------------------------------------------------------------
            # THE FIX: Force a deep copy of the final observation array
            # -----------------------------------------------------------------
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


class MyenvTask(Task):

    MY_TASK = None

    @staticmethod
    def associated_class():
        return MyEnvClass


class MyEnvClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        return lambda: PettingZooWrapper(
            MyCustomEnv2(),
            categorical_actions=True,
            device=device,
            seed=seed,
            return_state=False,
            **config,
        )

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_state(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return 100

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.full_action_spec

    @staticmethod
    def env_name() -> str:
        return "myenv"
