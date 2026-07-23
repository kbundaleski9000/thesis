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

    def simulate_forward_with_policy(self, zeta, W_max=15, theta_leader=None):
        """
        Vectorized version of the forward rollout. Numerically identical to the
        loop-based version -- inner Python loops over (k, u, v) replaced with
        masked/batched tensor operations, exactly mirroring the vectorization
        applied to solve_multigroup's forward pass (see solver_vectorized.py).
        The loop over h (time) remains a genuine Python loop -- it is inherently
        sequential and cannot be vectorized away.
        """
        K, H, N = self.K, self.H, self.N
        device = self.device

        edge_cost = torch.zeros((N, N), device=device)
        edge_cost[0, 2] = 5.5
        edge_cost[1, 3] = 5.5

        policies = torch.stack([self.get_policy(zeta[k]) for k in range(K)], dim=0)  # (K,H,N,N)

        # (K,N) mask: is_sink[k,u] = 1 iff u is group k's sink. Rule C's "skip
        # departure at the sink" behavior is per-group, so this can't be folded
        # into the adjacency mask alone once K > 1.
        is_sink = torch.zeros((K, N), device=device)
        sources = torch.zeros((K, N), device=device)
        for k in range(K):
            is_sink[k, self.groups[k]["sink"]] = 1.0
            sources[k, self.groups[k]["source"]] = self.groups[k]["mass"]

        node_mass_list = [sources]
        edge_occ_list = [torch.zeros((K, N, N, W_max + 1), device=device)]

        W_cong_history = torch.zeros((H, N, N), device=device)

        for h in range(H - 1):
            current_node_mass = node_mass_list[h]      # (K,N)
            current_edge_occ = edge_occ_list[h]         # (K,N,N,W+1)
            pol_h = policies[:, h, :, :]                # (K,N,N)

            # tentative_edge_traffic[u,v] = sum_k current_node_mass[k,u] * pol_h[k,u,v]
            tentative_edge_traffic = torch.einsum('ku,kuv->uv', current_node_mass, pol_h) * self._adj_bool.float()

            E_total_edges = current_edge_occ.sum(dim=(0, 3)) + tentative_edge_traffic
            E_total_edges_final = torch.zeros((N, N), device=device)
            E_total_edges_final[0, 1] = E_total_edges[0, 1]
            E_total_edges_final[2, 3] = E_total_edges[2, 3]

            W_cong_history[h] = torch.clamp(
                E_total_edges_final * 5.0 + edge_cost + theta_leader,
                min=0.0, max=float(W_max)
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

        return node_mass, edge_occ, final_flows, list(policies), W_cong_history