import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from leader.leadernet import GraphLeaderIncentiveNet, LeaderIncentiveNet
from solver.solver import PMFG_OMD_Solver_MultiGroup, solve_multigroup


class GraphEdgeMFG_Trainer:
    def __init__(self, env, solvers, leader_lr=0.01):
        self.env = env
        self.solvers = solvers

        # Initialize the Edge Leader Network
        self.leader_nets = GraphLeaderIncentiveNet(env.N, env.K).to(env.device)
        self.optimizer = optim.Adam(self.leader_nets.parameters(), lr=leader_lr)
        
        # Precompute base theta as an edge matrix layout: (K, N, 1) -> expands to (K, N, N)
        self.base_thetas = - torch.zeros((self.env.K, self.env.N, self.env.N), dtype=torch.float32, device=env.device)
        self.base_thetas[0,4,2] = -1.1250
        self.base_thetas[0,5,3] = -1.1250

        print(self.env.K)


        for k in range(self.env.K):
            sink_idx = self.env.groups[k]["sink"]
            self.base_thetas[k, sink_idx, sink_idx] = env.N  # No congestion cost at the sink

    def _prepare_input(self, final_flows, flows_groups, policies):
        """Construct structural feature blocks for graph nodes."""
        N, K = self.env.N, self.env.K

        features = torch.zeros((self.env.K, 2, self.env.N, self.env.N), dtype=torch.float32, device=self.env.device)
        flows = torch.tensor(final_flows, dtype=torch.float32, device=self.env.device)
        # Compute total edge traffic across all groups for each time step.

        edge_flows = [flow.unsqueeze(-1) * policy for flow, policy in zip(flows, policies)]
        E_total_edges = torch.stack(edge_flows).sum(dim=0)  # (H, N, N)



        for k in range(K):
            adj = self.env.M
            norm_dist = self.base_thetas[k] / self.env.N
            features[k, 0, :,:] = norm_dist
            features[k, 1, :,:] = adj

        E_total_edges_flat = E_total_edges.view(-1)  # (H, N, N)
        features_flat = features.view(-1)  # Shape: (K * 2 * N * N)
        flows_flat = flows.view(-1)

        features_final = torch.cat([features_flat, flows_flat, E_total_edges_flat], dim=0)

        return features_final

    def compute_social_loss(self, flows, policies, theta_leader):
        """Calculates systemic social reward using link-level traffic contraction."""
        total_social_reward = 0.0

        # Compute total edge traffic across all groups for each time step.
        edge_flows = [flow.unsqueeze(-1) * policy for flow, policy in zip(flows, policies)]
        E_total_edges = torch.stack(edge_flows).sum(dim=0)  # (H, N, N)

        for k in range(self.env.K):
            flow_k = flows[k]        # (H, N)
            policy_k = policies[k]  # (H, N, N)

            for h in range(self.solvers[k].H):
                # 1. Compute this group's specific link utilization matrix
                E_kh = flow_k[h].unsqueeze(-1) * policy_k[h]  # Shape: (N, N)

                # 2. Use the true total edge utilization across all groups
                E_total_h = E_total_edges[h]

                E_total_final = torch.zeros_like(E_total_h)
                E_total_final[0,1] = E_total_h[0,1]
                E_total_final[2,3] = E_total_h[2,3]

                # 3. Aggregate Edge Reward Matrix
                reward_matrix = -self.solvers[k].alpha * E_total_final + self.base_thetas[k] + theta_leader[0]
                reward_matrix[self.env.groups[k]["sink"], self.env.groups[k]["sink"]] = 0.0  # No congestion cost at the sink

                # 4. Perform matrix dot product contraction
                step_reward = torch.sum(E_kh * reward_matrix)
                total_social_reward += step_reward

        return -total_social_reward

    def train_step(self, final_flows_previous, flows_previous, policies):
        self.optimizer.zero_grad()

        inp = self._prepare_input(final_flows_previous, flows_previous, policies)
        theta_leader = self.leader_nets(inp)  # Shape: (K, N, N)

        print("base_thetas in train_step", self.base_thetas.shape)

        # Broadcast alignment with the base static matrices
        theta_final = self.base_thetas + theta_leader  # Shape: (K, N, N)

        # solve_multigroup structure stays identical; handles N x N strategy spaces naturally
        policies, flows, final_flows = solve_multigroup(self.solvers, theta_final)

        loss = self.compute_social_loss(flows, policies, theta_leader)
        loss.backward()
        self.optimizer.step()

        # Detach if they are lists of tensors
        detached_flows = [f.detach() for f in flows] if isinstance(flows, list) else flows.detach()
        detached_policies = [p.detach() for p in policies] if isinstance(policies, list) else policies.detach()

        return loss.item(), final_flows.detach(), detached_flows, detached_policies