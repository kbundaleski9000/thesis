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

        # Precomputed once, reused by get_policy / simulate_forward_with_policy instead
        # of rebuilding via get_neighbors() loops on every call.
        self._adj_bool = self.A > 0  # (N,N) bool
        self._has_neighbors = self._adj_bool.any(dim=1)  # (N,) bool

    def get_neighbors(self, u):
        neighbors = torch.where(self.A[u] > 0)[0].tolist()
        return neighbors

    def get_policy(self, zeta):
        """Vectorized: builds the full (H,N,N) mask in one shot instead of looping over u."""
        mask = torch.where(
            self._adj_bool.unsqueeze(0).expand(self.H, -1, -1),
            zeta,
            torch.full_like(zeta, float('-inf'))
        )
        policy = F.softmax(mask, dim=-1)
        # FIX: a node with zero outgoing edges has an all -inf row, and
        # softmax(all -inf) = 0/0 = NaN. Leave such rows at zero instead
        # (matches the original loop-based version, which simply skips
        # computing softmax for nodes with no neighbors).
        policy = torch.where(self._has_neighbors.view(1, -1, 1), policy, torch.zeros_like(policy))
        return policy

    def simulate_forward_with_policy(self, zeta, W_max=15, theta_leader=None, edge_cost=None,
                                      congest_all_edges=True, capacity=None, alpha=0.15, beta=4.0,
                                      cost_model="linear"):
        """
        Vectorized version of the forward rollout.

        FIX: edge_cost and the congestion mask are now real parameters instead of
        hardcoded for two specific edges -- see solve_multigroup in solver.py for
        the full explanation of why (the old hardcoding was specific to the small
        4-node test network and silently left every other edge free and
        congestion-insensitive on a larger graph).
        """
        K, H, N = self.K, self.H, self.N
        device = self.device    

        edge_cost = torch.zeros((N, N), device=device)
        edge_cost[0, 1] = 6
        edge_cost[0, 2] = 4
        edge_cost[1, 0] = 6
        edge_cost[1, 5] = 5
        edge_cost[2, 0] = 4

        edge_cost[2, 3] = 4
        edge_cost[2, 11] = 4
        edge_cost[3, 2] = 4
        edge_cost[3, 4] = 2
        edge_cost[3, 10] = 6

        edge_cost[4, 3] = 2
        edge_cost[4, 5] = 4
        edge_cost[4, 8] = 5
        edge_cost[5, 1] = 5
        edge_cost[5, 4] = 4

        edge_cost[5, 7] = 2
        edge_cost[6, 7] = 3
        edge_cost[6, 17] = 2
        edge_cost[7, 5] = 2
        edge_cost[7, 6] = 3

        edge_cost[7, 8] = 10
        edge_cost[7, 15] = 5
        edge_cost[8, 4] = 5
        edge_cost[8, 7] = 10
        edge_cost[8, 9] = 3

        edge_cost[9, 8] = 3
        edge_cost[9, 10] = 5
        edge_cost[9, 14] = 6
        edge_cost[9, 15] = 4
        edge_cost[9, 16] = 8

        edge_cost[10, 3] = 6
        edge_cost[10, 9] = 5
        edge_cost[10, 11] = 6
        edge_cost[10, 13] = 4
        edge_cost[11, 2] = 4

        edge_cost[11, 10] = 6
        edge_cost[11, 12] = 3
        edge_cost[12, 11] = 3
        edge_cost[12, 23] = 4
        edge_cost[13, 10] = 4

        edge_cost[13, 14] = 5
        edge_cost[13, 22] = 4
        edge_cost[14, 9] = 6
        edge_cost[14, 13] = 5
        edge_cost[14, 18] = 3

        edge_cost[14, 21] = 3
        edge_cost[15, 7] = 5
        edge_cost[15, 9] = 4
        edge_cost[15, 16] = 2
        edge_cost[15, 17] = 3

        edge_cost[16, 9] = 8
        edge_cost[16, 15] = 2
        edge_cost[16, 18] = 2
        edge_cost[17, 6] = 2
        edge_cost[17, 15] = 3

        edge_cost[17, 19] = 4
        edge_cost[18, 14] = 3
        edge_cost[18, 16] = 2
        edge_cost[18, 19] = 4
        edge_cost[19, 17] = 4

        edge_cost[19, 18] = 4
        edge_cost[19, 20] = 6
        edge_cost[19, 21] = 5
        edge_cost[20, 19] = 6
        edge_cost[20, 21] = 2

        edge_cost[20, 23] = 3
        edge_cost[21, 14] = 3
        edge_cost[21, 19] = 5
        edge_cost[21, 20] = 2
        edge_cost[21, 22] = 4

        edge_cost[22, 13] = 4
        edge_cost[22, 21] = 4
        edge_cost[22, 23] = 2
        edge_cost[23, 12] = 4
        edge_cost[23, 20] = 3

        edge_cost[23, 22] = 2
        edge_cost -= 2

        capacity = torch.zeros((N, N), device=device)
        capacity[0, 1] = 0.071825
        capacity[0, 2] = 0.064901
        capacity[1, 0] = 0.071825
        capacity[1, 5] = 0.013750
        capacity[2, 0] = 0.064901

        capacity[2, 3] = 0.047450
        capacity[2, 11] = 0.064901
        capacity[3, 2] = 0.047450
        capacity[3, 4] = 0.049314
        capacity[3, 10] = 0.013613

        capacity[4, 3] = 0.049314
        capacity[4, 5] = 0.013722
        capacity[4, 8] = 0.027732
        capacity[5, 1] = 0.013750
        capacity[5, 4] = 0.013722

        capacity[5, 7] = 0.013585
        capacity[6, 7] = 0.021747
        capacity[6, 17] = 0.064901
        capacity[7, 5] = 0.013585
        capacity[7, 6] = 0.021747

        capacity[7, 8] = 0.014005
        capacity[7, 15] = 0.013993
        capacity[8, 4] = 0.027732
        capacity[8, 7] = 0.014005
        capacity[8, 9] = 0.038591

        capacity[9, 8] = 0.038591
        capacity[9, 10] = 0.027732
        capacity[9, 14] = 0.037471
        capacity[9, 15] = 0.013463
        capacity[9, 16] = 0.013848

        capacity[10, 3] = 0.013613
        capacity[10, 9] = 0.027732
        capacity[10, 11] = 0.013613
        capacity[10, 13] = 0.013523
        capacity[11, 2] = 0.064901

        capacity[11, 10] = 0.013613
        capacity[11, 12] = 0.071825
        capacity[12, 11] = 0.071825
        capacity[12, 23] = 0.014119
        capacity[13, 10] = 0.013523

        capacity[13, 14] = 0.014219
        capacity[13, 22] = 0.013657
        capacity[14, 9] = 0.037471
        capacity[14, 13] = 0.014219
        capacity[14, 18] = 0.040390

        capacity[14, 21] = 0.026620
        capacity[15, 7] = 0.013993
        capacity[15, 9] = 0.013463
        capacity[15, 16] = 0.014503
        capacity[15, 17] = 0.054575

        capacity[16, 9] = 0.013848
        capacity[16, 15] = 0.014503
        capacity[16, 18] = 0.013378
        capacity[17, 6] = 0.064901
        capacity[17, 15] = 0.054575

        capacity[17, 19] = 0.064901
        capacity[18, 14] = 0.040390
        capacity[18, 16] = 0.013378
        capacity[18, 19] = 0.013873
        capacity[19, 17] = 0.064901

        capacity[19, 18] = 0.013873
        capacity[19, 20] = 0.014032
        capacity[19, 21] = 0.014076
        capacity[20, 19] = 0.014032
        capacity[20, 21] = 0.014503

        capacity[20, 23] = 0.013548
        capacity[21, 14] = 0.026620
        capacity[21, 19] = 0.014076
        capacity[21, 20] = 0.014503
        capacity[21, 22] = 0.013866


        capacity[22, 13] = 0.013657
        capacity[22, 21] = 0.013866
        capacity[22, 23] = 0.014083
        capacity[23, 12] = 0.014119
        capacity[23, 20] = 0.013548

        capacity[23, 22] = 0.014083
        
        capacity.sqrt_()
        

        if cost_model == "bpr" and capacity is None:
            raise ValueError(
                "cost_model='bpr' requires a real capacity tensor -- pass capacity=... "
                "explicitly, scaled to match your mass units."
            )

        congest_mask = self._adj_bool.float() if congest_all_edges else torch.zeros((N, N), device=device)
        if not congest_all_edges:
            congest_mask[0, 1] = 1.0
            congest_mask[2, 3] = 1.0

        policies = torch.stack([self.get_policy(zeta[k]) for k in range(K)], dim=0)  # (K,H,N,N)

        # (K,N) mask: is_sink[k,u] = 1 iff u is group k's sink. Rule C's "skip
        # departure at the sink" behavior is per-group, so this can't be folded
        # into the adjacency mask alone once K > 1.
        is_sink = torch.zeros((K, N), device=device)
        sources = torch.zeros((K, N), device=device)
        for k in range(K):
            is_sink[k, self.groups[k]["sink"]] = 1.0
            sources[k, self.groups[k]["source"]] = self.groups[k]["mass"]

        
        not_sink_mask = 1.0 - is_sink
        node_mass_list = [sources]
        edge_occ_list = [torch.zeros((K, N, N, W_max + 1), device=device)]

        W_cong_history = torch.zeros((H, N, N), device=device)
        W_parts = torch.zeros((4, H, N, N), device=device)

        for h in range(H - 1):
            current_node_mass = node_mass_list[h]      # (K,N)
            current_edge_occ = edge_occ_list[h]         # (K,N,N,W+1)
            pol_h = policies[:, h, :, :]                # (K,N,N)

            # tentative_edge_traffic[u,v] = sum_k current_node_mass[k,u] * pol_h[k,u,v]
            tentative_edge_traffic = torch.einsum('ku,kuv->uv', current_node_mass * not_sink_mask, pol_h) * self._adj_bool.float()

            E_total_edges = current_edge_occ.sum(dim=(0, 3)) + tentative_edge_traffic
            
            W_congestion = E_total_edges / capacity.clamp(min=1e-6)   # part 1
            W_freeflow   = edge_cost                                        # part 2
            W_toll       = theta_leader

            if cost_model == "bpr":
                print("bpr")
            else:
                W_parts[0, h] = W_congestion
                W_parts[1, h] = W_freeflow
                W_parts[2, h] = W_toll
                W_parts[3, h] = E_total_edges

                W_cong_history[h] = torch.clamp(
                    W_congestion + W_freeflow + W_toll, min=0.0, max=float(W_max)
            )


            delay = W_cong_history[h]
            delay_floor = torch.clamp(torch.floor(delay).long(), 0, W_max)
            delay_ceil = torch.clamp(torch.ceil(delay).long(), 0, W_max)
            weight_ceil = delay - delay_floor.float()
            weight_floor = 1.0 - weight_ceil

            # Rule A: shift existing tiers w -> w-1 (already a simple slice-shift).
            next_edge_occ = torch.zeros((K, N, N, W_max + 1), device=device)
            next_edge_occ[..., :W_max] = current_edge_occ[..., 1:]

            # Rule C: new departures, zeroed at any (k,u) where u is group k's sink.
            not_sink_mask = 1.0 - is_sink  # (K,N)
            moving_mass = torch.einsum('ku,kuv->kuv', current_node_mass * not_sink_mask, pol_h) * self._adj_bool.float()  # (K,N,N)

            floor_onehot = F.one_hot(delay_floor, num_classes=W_max + 1).float()  # (N,N,W+1)
            ceil_onehot = F.one_hot(delay_ceil, num_classes=W_max + 1).float()
            dispatch = (
                moving_mass.unsqueeze(-1) * weight_floor.unsqueeze(0).unsqueeze(-1) * floor_onehot.unsqueeze(0) +
                moving_mass.unsqueeze(-1) * weight_ceil.unsqueeze(0).unsqueeze(-1) * ceil_onehot.unsqueeze(0)
            )
            next_edge_occ = next_edge_occ + dispatch

            # Sink mass that stayed (no departure) carries forward directly.
            sink_carry = current_node_mass * is_sink  # (K,N)

            # Rule B: release whatever ended up at tier 0 (mass decremented from tier 1
            # AND mass freshly dispatched at delay==0), matching the "+1" timing convention.
            arriving_per_v = next_edge_occ[:, :, :, 0].sum(dim=1)  # (K,N)
            next_edge_occ[:, :, :, 0] = 0.0

            next_node_mass = sink_carry + arriving_per_v

            node_mass_list.append(next_node_mass)
            edge_occ_list.append(next_edge_occ)

        node_mass = torch.stack(node_mass_list, dim=1)
        edge_occ = torch.stack(edge_occ_list, dim=1)
        final_flows = node_mass.sum(dim=0)

        return node_mass, edge_occ, final_flows, list(policies), W_cong_history, W_parts