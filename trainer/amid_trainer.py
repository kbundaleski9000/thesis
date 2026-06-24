import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from leader.leadernet import GraphLeaderIncentiveNet, LeaderIncentiveNet
from solver.solver import PMFG_OMD_Solver_MultiGroup, solve_multigroup


class AMID_Trainer_MultiGroup:
    def __init__(self, env, solvers, leader_lr=0.01, leader_loss_objective = "social_optimum" ):
        self.env     = env
        self.solvers = solvers

        # One leader network per group (shared optimizer)
        self.leader_nets = LeaderIncentiveNet(env.rows, env.cols, env.K).to(env.device)
        self.optimizer = optim.Adam(self.leader_nets.parameters(), lr=leader_lr)
        self.leader_loss_objective = leader_loss_objective 

        # Precompute base theta per group (distance-based)
        self.base_thetas = self._make_base_theta()


    # ── helpers ───────────────────────────────────────────────
    def _make_base_theta(self):
        theta = torch.tensor(-self.env.dist_maps, dtype=torch.float32, device=self.env.device)
        return theta

    def _prepare_input(self):
        """3*K channel grid image for all groups."""
        env = self.env
        channels = []
        for k in range(env.K):
            group = env.groups[k]
            obs_ch  = env.obstacles.float()
            sink_ch = torch.zeros((env.rows, env.cols), device=env.device)
            sink_ch[group["sink"][0], group["sink"][1]] = 1.0
            src_ch  = torch.zeros((env.rows, env.cols), device=env.device)
            src_ch[group["source"][0], group["source"][1]] = 1.0
            channels.extend([obs_ch, sink_ch, src_ch])
        return torch.stack(channels).unsqueeze(0)  # (1, 3*K, rows, cols)

    # ── leader objective ───────────────────────────────────────
    def leader_objective(self, flows, final_flows, theta_leader):
        """
        Minimise total congestion + encourage all groups to reach their sinks.
        """

        total_social_reward = torch.tensor(0.0, dtype=torch.float32, device=self.env.device)

        for k in range(self.env.K):
            flow_k = flows[k]

            for h in range(self.solvers[k].H):
                # 1. Congestion cost: -alpha * L^2
                reward = -self.solvers[k].alpha * (final_flows[h])

                # include base distance theta and leader-provided incentive
                print("theta_leader in leader_objective", theta_leader.shape)
                print("base_thetas in leader_objective", self.base_thetas.shape)
                reward = reward + self.base_thetas[k] + theta_leader[0]
                reward[self.env.groups[k]["sink"]] = 0.0 # No congestion cost at the sink

                # Reward per cell: (congestion + signal + entropy)
                # We multiply by density (final_flow) to get total reward for the population
                step_reward = torch.sum(flow_k[h] * reward)
                total_social_reward = total_social_reward + step_reward

        # Leader minimizes the negative of total reward
        return -total_social_reward 
    
    def leader_objective_social_optimum(self, flows, theta_list, theta1_list):
        """
        Minimise total travel time .
        """
        L_total = torch.stack(flows).sum(dim=0)
        congestion = torch.sum(L_total ** 2)

        reg = sum(torch.sum(abs(t1)) for t1 in theta1_list)
        return congestion 
    
    def loss_follower(self, flows, final_flows, theta_leader):
        print("theta_leader in loss_follower", theta_leader.shape)

        total_social_reward = torch.tensor(0.0, dtype=torch.float32, device=self.env.device)

        for k in range(self.env.K):
            flow_k = flows[k]

            for h in range(self.solvers[k].H):
                # 1. Congestion cost: -alpha * L^2
                reward = -self.solvers[k].alpha * (final_flows[h])
                reward = reward + self.base_thetas[k] + theta_leader[0]

                # Reward per cell: (congestion + signal + entropy)
                # We multiply by density (final_flow) to get total reward for the population
                step_reward = torch.sum(flow_k[h] * reward)
                total_social_reward = total_social_reward + step_reward

        # Leader minimizes the negative of total reward
        return -total_social_reward 

    # ── single training step ───────────────────────────────────
    def train_step(self):
        self.optimizer.zero_grad()

        inp = self._prepare_input()
        
        theta_leader = self.leader_nets(inp)

        theta_final = theta_leader + self.base_thetas  # (K, rows, cols)

        print(theta_leader)

        _, flows, final_flows = solve_multigroup(self.solvers, theta_final)

        loss = self.leader_objective(flows, final_flows, theta_leader)

        loss.backward()
        self.optimizer.step()

        loss_follower = self.loss_follower(flows, final_flows, theta_leader)

        return loss.item(), loss_follower.item()

