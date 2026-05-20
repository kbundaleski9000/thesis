import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from leader.leadernet import LeaderIncentiveNet
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
        self.base_thetas = [
            self._make_base_theta(k) for k in range(env.K)
        ]

    # ── helpers ───────────────────────────────────────────────
    def _make_base_theta(self, k):
        dist_map = self.env.dist_maps[k]
        theta = -torch.tensor(dist_map, device=self.env.device)  # Base incentive: negative distance
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
    def leader_objective(self, flows, theta_list, theta1_list):
        """
        Minimise total congestion + encourage all groups to reach their sinks.
        """
        L_total = torch.stack(flows).sum(dim=0)
        congestion = torch.sum(L_total ** 2)

        reg = sum(torch.sum(abs(t1)) for t1 in theta1_list)
        return congestion 
    
    def leader_objective_social_optimum(self, flows, theta_list, theta1_list):
        """
        Minimise total travel time .
        """
        L_total = torch.stack(flows).sum(dim=0)
        congestion = torch.sum(L_total ** 2)

        reg = sum(torch.sum(abs(t1)) for t1 in theta1_list)
        return congestion 
    

    # ── single training step ───────────────────────────────────
    def train_step(self):
        self.optimizer.zero_grad()

        inp    = self._prepare_input()
        theta1_all = - self.leader_nets(inp)

        theta_list  = [self.base_thetas[k] + theta1_all for k in range(self.env.K)]
        theta1_list = [theta1_all for k in range(self.env.K)]


        _, flows, _ = solve_multigroup(self.solvers, theta_list)
        

        loss = self.leader_objective(flows, theta_list, theta1_list)

        loss.backward()
        self.optimizer.step()
        return loss.item()
