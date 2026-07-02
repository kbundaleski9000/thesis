import torch
import numpy as np

class GraphWorldMFG_MultiGroup:
    """
    Graph environment that hosts K agent groups with routing paths.
    """
    def __init__(self, num_nodes, adjacency_matrix, groups, H,device="cpu"):
        self.device = device
        self.groups = groups
        self.K      = len(groups)
        self.num_nodes = num_nodes
        self.H = H
        
        # A is an adjacency matrix tensor of shape (N, N)
        self.A = torch.tensor(adjacency_matrix, dtype=torch.float32, device=device)
        self.N = self.A.shape[0]

    def get_neighbors(self, u):
        """Returns a list of integer node indices that node u connects to."""
        # Finds all columns 'v' where edge edge weight A[u, v] > 0
        neighbors = torch.where(self.A[u] > 0)[0].tolist()
        return neighbors
    
    def simulate_forward_with_policy(self, policy_sim, theta_leader, W_max=3):
        import torch

        if isinstance(policy_sim, list):
            policy_sim = torch.stack(policy_sim, dim=0)

        edge_cost = torch.zeros((self.N, self.N), device=self.device)
        edge_cost[0, 2] = 5.5
        edge_cost[1, 3] = 5.5

        # 1. Start with a clean list of individual timestep frames
        flows_list = []
        first_step = torch.zeros((self.K, self.N, W_max + 1), device=self.device)
        for k in range(self.K):
            source = self.groups[k]["source"]
            first_step[k, source, 0] = self.groups[k]["mass"]
        flows_list.append(first_step)

        W_cong_list = []

        # 2. Chronological timeline loop
        for h in range(self.H - 1):
            current_flows = flows_list[h]
            E_total_edges = torch.zeros((self.N, self.N), device=self.device)
            
            for k in range(self.K):
                active_mass = current_flows[k, :, 0]
                for u in range(self.N):
                    for v in self.get_neighbors(u):
                        E_total_edges[u, v] = E_total_edges[u, v] + (active_mass[u] * policy_sim[k, h, u, v])

            E_total_edges_final = torch.zeros((self.N, self.N), device=self.device)
            E_total_edges_final[0, 1] = E_total_edges[0, 1]
            E_total_edges_final[2, 3] = E_total_edges[2, 3]

            current_W_cong = torch.clamp(E_total_edges_final * 5.0 + edge_cost + theta_leader, min=1.0, max=float(W_max))
            W_cong_list.append(current_W_cong)

            # Build the next frame in isolation
            next_flow_step = torch.zeros((self.K, self.N, W_max + 1), device=self.device)
            for k in range(self.K):
                sink = self.groups[k]["sink"]
                
                # Rule A: Step down waiting tiers out-of-place
                for w in range(1, W_max + 1):
                    next_flow_step[k, :, w - 1] = next_flow_step[k, :, w - 1] + current_flows[k, :, w]

                # Rule B: Move active agents
                active_mass = current_flows[k, :, 0]
                for u in range(self.N):
                    if u == sink:
                        next_flow_step[k, sink, 0] = next_flow_step[k, sink, 0] + active_mass[sink]
                        continue

                    for v in self.get_neighbors(u):
                        moving_mass = active_mass[u] * policy_sim[k, h, u, v]
                        delay_continuous = current_W_cong[u, v]
                        
                        delay_floor = torch.clamp(torch.floor(delay_continuous).long(), 0, W_max)
                        delay_ceil = torch.clamp(torch.ceil(delay_continuous).long(), 0, W_max)
                        weight_ceil = delay_continuous - delay_floor.float()
                        weight_floor = 1.0 - weight_ceil

                        next_flow_step[k, v, delay_floor] = next_flow_step[k, v, delay_floor] + (moving_mass * weight_floor)
                        next_flow_step[k, v, delay_ceil] = next_flow_step[k, v, delay_ceil] + (moving_mass * weight_ceil)

            flows_list.append(next_flow_step)

        flows = torch.stack(flows_list, dim=1)
        W_cong_list.append(torch.zeros((self.N, self.N), device=self.device))
        W_cong_history = torch.stack(W_cong_list, dim=0)

        return flows, flows.sum(dim=0).sum(dim=-1), policy_sim, W_cong_history