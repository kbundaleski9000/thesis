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
        self.base_thetas = self._make_base_theta()


    # ── helpers ───────────────────────────────────────────────
    def _make_base_theta(self):
        theta = torch.tensor(self.env.dist_maps - self.env.dist_maps, device=self.env.device)
        theta[0, 2, 1] = -1.125
        theta[0, 0, 3] = -1.125

        theta[0, 2, 4] = 25.0  # Stronger incentive at the sink
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
    def leader_objective(self, final_flows, theta_leader):
        """
        Minimise total congestion + encourage all groups to reach their sinks.
        """
        total_social_reward = 0.0

        for h in range(self.solvers[0].H):
            # 1. Congestion cost: -alpha * L^2
            reward1 = -self.solvers[0].alpha  * (final_flows[h,:,:])
            reward = reward1 - reward1
            reward[0,1] = reward1[0,1]  # Incentivize the gap cell
            reward[2,3] = reward1[2,3]  # Incentivize the gap cell
            reward += self.base_thetas[0]
            reward[2,4] = 0.0  
            
            # Reward per cell: (congestion + signal + entropy)
            # We multiply by density (final_flow) to get total reward for the population
            step_reward = torch.sum(final_flows[h,:,:] * (reward))
            total_social_reward += step_reward 


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
    

    # ── single training step ───────────────────────────────────
    def train_step(self):
        self.optimizer.zero_grad()

        inp  = self._prepare_input()
        theta_leader = self.leader_nets(inp)
        
        if self.env.K > 1:
            theta_leader = theta_leader.repeat(self.env.K, 1, 1)  # (K, 3, rows, cols)

        print("theta_leader", theta_leader.shape)

        theta_final = theta_leader + self.base_thetas  # (K, rows, cols)

        _, _, final_flows = solve_multigroup(self.solvers, theta_final)

        loss = self.leader_objective(final_flows, theta_leader)

        loss.backward()
        self.optimizer.step()
        return loss.item()
