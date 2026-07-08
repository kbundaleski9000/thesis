import torch
import numpy as np
import torch.nn.functional as F

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
    
    def get_policy(self, zeta):
        """Generates a valid routing probability distribution over valid outbound links across all h."""
        # policy shape: (H, N, N)
        policy = torch.zeros_like(zeta)
        for u in range(self.N):
            neighbors = self.get_neighbors(u)
            if len(neighbors) > 0:
                # 1. Initialize a 2D mask of shape (H, N) with -inf
                mask = torch.full((self.H, self.N), float('-inf'), device=self.device)

                mask[:, neighbors] = zeta[:, u, neighbors]
                policy[:, u, :] = F.softmax(mask, dim=-1)

                # 3. Softmax across the destination node dimension (dim=-1)
                # This ensures invalid nodes get 0 probability and valid ones sum to 1.0 for every timestep h
        return policy
    
    def simulate_forward_with_policy(self, zeta, W_max=15, theta_leader=None):
        """
        Main MFG execution loop using explicit congestion waiting-times.

        State representation (fixed):
        - node_mass[k, h, u]        : free mass at node u, ready to make a routing decision
        - edge_occ[k, h, u, v, w]   : mass currently IN TRANSIT on edge (u,v) with w steps left

        This replaces the old node-waiting-tier bookkeeping, which lost track of which
        edge in-transit mass was occupying, causing congestion to undercount mass that
        departed on earlier timesteps and was still mid-transit.

        Returns:
            node_mass: (K, H, N) - arrived/free mass per node over time
            edge_occ: (K, H, N, N, W_max+1) - in-transit mass per edge over time
            final_flows: (H, N) - aggregated spatial footprint (sum over K of node_mass + in-transit mass collapsed to origin/dest as needed)
            policies: converged agent policy tensors per group
            W_cong_history: (H, N, N)
            zeta_history: (T, K, H, N, N)
        """
        K = self.K
        H = self.H
        N = self.N
        edge_cost = torch.zeros((N, N), device=self.device)
        edge_cost[0, 2] = 5.5
        edge_cost[1, 3] = 5.5


        policies = [self.get_policy(zeta[k]) for k in range(self.K)]

        node_mass_list = []
        edge_occ_list = []

        # Seed initial source distribution at h=0 (all mass free at source)
        first_node_mass = torch.zeros((K, N), device=self.device)
        for k in range(K):
            source = self.groups[k]["source"]
            first_node_mass[k, source] = self.groups[k]["mass"]
        node_mass_list.append(first_node_mass)
        edge_occ_list.append(torch.zeros((K, N, N, W_max + 1), device=self.device))

        W_cong_history = torch.zeros((H, N, N), device=self.device)

        # Chronological progression step-by-step
        for h in range(H - 1):
            current_node_mass = node_mass_list[h]
            current_edge_occ = edge_occ_list[h]

            # tentative: mass that WOULD depart this step, under current policy
            tentative_edge_traffic = torch.zeros((N, N), device=self.device)
            for k in range(K):
                active_mass = current_node_mass[k]
                for u in range(N):
                    for v in self.get_neighbors(u):
                        tentative_edge_traffic[u, v] += active_mass[u] * policies[k][h, u, v]

            # congestion now includes BOTH already-in-transit occupancy AND this step's new departures
            E_total_edges = current_edge_occ.sum(dim=(0, 3)) + tentative_edge_traffic

            E_total_edges_final = torch.zeros((N, N), device=self.device)
            E_total_edges_final[0, 1] = E_total_edges[0, 1]
            E_total_edges_final[2, 3] = E_total_edges[2, 3]

            W_cong_history[h] = torch.clamp(
                E_total_edges_final * 5.0 + edge_cost + theta_leader,
                min=0.0, max=float(W_max)
            )
            # ... rest of the step (Rule A/B/C) proceeds exactly as before, using this W_cong_history[h]

            # Build next frame out-of-place
            next_node_mass = torch.zeros((K, N), device=self.device)
            next_edge_occ = torch.zeros((K, N, N, W_max + 1), device=self.device)

            for k in range(K):
                sink = self.groups[k]["sink"]

                # Rule A: in-transit mass steps down one waiting tier, staying on the same edge
                for w in range(1, W_max + 1):
                    next_edge_occ[k, :, :, w - 1] = next_edge_occ[k, :, :, w - 1] + current_edge_occ[k, :, :, w]

                # Rule B: mass with w=0 has arrived -> becomes free node_mass at destination v
                arriving_per_v = current_edge_occ[k, :, :, 0].sum(dim=0)  # sum over origin u -> (N,)
                next_node_mass[k] = next_node_mass[k] + arriving_per_v

                # Rule C: free mass at each node makes a new routing decision
                active_mass = current_node_mass[k]  # (N,)
                for u in range(N):
                    if u == sink:
                        next_node_mass[k, sink] = next_node_mass[k, sink] + active_mass[sink]
                        continue

                    for v in self.get_neighbors(u):
                        moving_mass = active_mass[u] * policies[k][h, u, v]

                        # Differentiable soft indexing (unchanged)
                        delay_continuous = W_cong_history[h, u, v]
                        delay_floor = torch.clamp(torch.floor(delay_continuous).long(), 0, W_max)
                        delay_ceil = torch.clamp(torch.ceil(delay_continuous).long(), 0, W_max)
                        weight_ceil = delay_continuous - delay_floor.float()
                        weight_floor = 1.0 - weight_ceil

                        # Mass enters the EDGE's occupancy, not the destination node's waiting tier
                        next_edge_occ[k, u, v, delay_floor] = next_edge_occ[k, u, v, delay_floor] + moving_mass * weight_floor
                        next_edge_occ[k, u, v, delay_ceil] = next_edge_occ[k, u, v, delay_ceil] + moving_mass * weight_ceil

            node_mass_list.append(next_node_mass)
            edge_occ_list.append(next_edge_occ)

        node_mass = torch.stack(node_mass_list, dim=1)   # (K, H, N)
        edge_occ = torch.stack(edge_occ_list, dim=1)      # (K, H, N, N, W_max+1)


        # Aggregated spatial footprint: free mass at each node, summed over groups
        final_flows = node_mass.sum(dim=0)  # (H, N)

        return node_mass, edge_occ, final_flows, policies, W_cong_history