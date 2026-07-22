import torch
import numpy as np
import torch.nn.functional as F

class GraphWorldMFG_MultiGroup:
    def __init__(self, num_nodes, adjacency_matrix, groups, H, device="cpu"):
        self.device = device
        self.groups = groups
        self.K = len(groups)
        self.num_nodes = num_nodes
        self.H = H
        self.A = torch.tensor(adjacency_matrix, dtype=torch.float32, device=device)
        self.N = self.A.shape[0]

    def get_neighbors(self, u):
        neighbors = torch.where(self.A[u] > 0)[0].tolist()
        return neighbors

    def get_policy(self, zeta):
        policy = torch.zeros_like(zeta)
        for u in range(self.N):
            neighbors = self.get_neighbors(u)
            if len(neighbors) > 0:
                mask = torch.full((self.H, self.N), float('-inf'), device=self.device)
                mask[:, neighbors] = zeta[:, u, neighbors]
                policy[:, u, :] = F.softmax(mask, dim=-1)
        return policy

    def simulate_forward_with_policy(self, zeta, W_max=15, theta_leader=None):
        K = self.K; H = self.H; N = self.N
        edge_cost = torch.zeros((N, N), device=self.device)
        edge_cost[0, 2] = 5.5
        edge_cost[1, 3] = 5.5

        policies = [self.get_policy(zeta[k]) for k in range(self.K)]

        node_mass_list = []
        edge_occ_list = []
        first_node_mass = torch.zeros((K, N), device=self.device)
        for k in range(K):
            source = self.groups[k]["source"]
            first_node_mass[k, source] = self.groups[k]["mass"]
        node_mass_list.append(first_node_mass)
        edge_occ_list.append(torch.zeros((K, N, N, W_max + 1), device=self.device))

        W_cong_history = torch.zeros((H, N, N), device=self.device)

        for h in range(H - 1):
            current_node_mass = node_mass_list[h]
            current_edge_occ = edge_occ_list[h]

            tentative_edge_traffic = torch.zeros((N, N), device=self.device)
            for k in range(K):
                active_mass = current_node_mass[k]
                for u in range(N):
                    for v in self.get_neighbors(u):
                        tentative_edge_traffic[u, v] += active_mass[u] * policies[k][h, u, v]

            E_total_edges = current_edge_occ.sum(dim=(0, 3)) + tentative_edge_traffic
            E_total_edges_final = torch.zeros((N, N), device=self.device)
            E_total_edges_final[0, 1] = E_total_edges[0, 1]
            E_total_edges_final[2, 3] = E_total_edges[2, 3]

            W_cong_history[h] = torch.clamp(
                E_total_edges_final * 5.0 + edge_cost + theta_leader,
                min=0.0, max=float(W_max)
            )

            next_node_mass = torch.zeros((K, N), device=self.device)
            next_edge_occ = torch.zeros((K, N, N, W_max + 1), device=self.device)

            for k in range(K):
                sink = self.groups[k]["sink"]
                # Rule A: shift existing tier w->w-1
                for w in range(1, W_max + 1):
                    next_edge_occ[k, :, :, w - 1] = next_edge_occ[k, :, :, w - 1] + current_edge_occ[k, :, :, w]

                # Rule C: new departures (MOVED before Rule B -- see solve_multigroup for why)
                active_mass = current_node_mass[k]
                for u in range(N):
                    if u == sink:
                        next_node_mass[k, sink] = next_node_mass[k, sink] + active_mass[sink]
                        continue
                    for v in self.get_neighbors(u):
                        moving_mass = active_mass[u] * policies[k][h, u, v]
                        delay_continuous = W_cong_history[h, u, v]
                        delay_floor = torch.clamp(torch.floor(delay_continuous).long(), 0, W_max)
                        delay_ceil = torch.clamp(torch.ceil(delay_continuous).long(), 0, W_max)
                        weight_ceil = delay_continuous - delay_floor.float()
                        weight_floor = 1.0 - weight_ceil
                        next_edge_occ[k, u, v, delay_floor] = next_edge_occ[k, u, v, delay_floor] + moving_mass * weight_floor
                        next_edge_occ[k, u, v, delay_ceil] = next_edge_occ[k, u, v, delay_ceil] + moving_mass * weight_ceil

                # Rule B: release whatever ended up at tier 0 -- catches BOTH mass that just
                # decremented down from tier 1 (Rule A) AND mass freshly dispatched at
                # delay==0 (Rule C), both becoming free THIS tick (uniform "delay+1" timing).
                arriving_per_v = next_edge_occ[k, :, :, 0].sum(dim=0)
                next_node_mass[k] = next_node_mass[k] + arriving_per_v
                next_edge_occ[k, :, :, 0] = 0.0

            node_mass_list.append(next_node_mass)
            edge_occ_list.append(next_edge_occ)

        node_mass = torch.stack(node_mass_list, dim=1)
        edge_occ = torch.stack(edge_occ_list, dim=1)
        final_flows = node_mass.sum(dim=0)

        return node_mass, edge_occ, final_flows, policies, W_cong_history
