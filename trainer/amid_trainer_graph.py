import torch
import torch.nn as nn
import torch.optim as optim
from leader.leadernet import GraphLeaderIncentiveNet
from solver.solver import solve_multigroup

class GraphEdgeMFG_Trainer:
    def __init__(self, env, solvers, leader_lr=0.01):
        self.env = env
        self.solvers = solvers
        self.H = solvers[0].H

        # Initialize Leader Network mapping historical edge traffic flows
        self.leader_nets = GraphLeaderIncentiveNet(env.N, env.K).to(env.device)
        self.optimizer = optim.Adam(self.leader_nets.parameters(), lr=leader_lr)
        
        # Base static background biases per edge choice (K, N, N)
        self.base_thetas = - torch.ones((self.env.K, self.env.N, self.env.N), dtype=torch.float32, device=env.device)

    def compute_social_loss(self, flows, W_cong_history):
        """
        Calculates total travel times as social loss.
        
        Under the indicator reward structure, the loss is the integration of all 
        population mass that hasn't arrived at its destination sink yet.
        """
        K, H, N, _ = flows.shape
        total_social_loss = 0.0

        for h in range(H):
            for k in range(K):
                sink = self.env.groups[k]["sink"]
                # Sum over all waiting time tiers to get total mass at each node
                spatial_mass_h = flows[k, h].sum(dim=-1) 
                
                for u in range(N):
                    if u != sink:
                        # Add 1.0 unit of social cost for every unit of mass stuck traveling
                        total_social_loss += spatial_mass_h[u].item()

        return total_social_loss
    
    def _prepare_input(self, final_flows, E_total_edges):
        """Construct structural feature blocks for graph nodes."""
        N, K = self.env.N, self.env.K

        features = torch.zeros((self.env.K, 2, self.env.N, self.env.N), dtype=torch.float32, device=self.env.device)
        
        for k in range(K):
            adj = self.env.A  # <-- Updated from self.env.M to match self.A
            norm_dist = self.base_thetas[k] / N
            features[k, 0, :, :] = norm_dist
            features[k, 1, :, :] = adj

        features_channels = features.view(K * 2, N, N)
        features_final = torch.cat([features_channels, E_total_edges], dim=0)

        return features_final

    def train_step(self, flows_previous, W_cong_history):
            """
            Executes one policy optimization step for the leader infrastructure network.
            """
            self.optimizer.zero_grad()

            # 1. Sum out the waiting time dimension to get purely spatial flows (K, H, N)
            spatial_flows = flows_previous.sum(dim=-1)

            # 2. Use your original input feature preparation method!
            # This function stacks the flows and edge metrics together to reach the expected 267 features
            inp = self._prepare_input(spatial_flows, W_cong_history)
            
            # 3. Pass the correctly shaped tensor into the network
            theta_leader = self.leader_nets(inp)  # Shape: (K, N, N)

            # 4. Combine Network Incentives with baseline static constraints
            theta_final = self.base_thetas + theta_leader  # Shape: (K, N, N)

            # 5. Simulate Agent Response (Inner MFG loop)
            flows_new, final_spatial_flows, policies_new = solve_multigroup(
                self.env, self.solvers, theta_final, T=20, W_max=5
            )

            # 6. Compute Loss and Backpropagate
            social_loss = self.compute_social_loss(flows_new, W_cong_history)
            
            loss_tensor = torch.tensor(social_loss, requires_grad=True, device=self.env.device)
            loss_tensor.backward()
            self.optimizer.step()

            return social_loss, flows_new