import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def solve_multigroup(env, solvers, T=200, W_max=100, theta_leader=None, edge_cost=None, capacity=None, alpha=0.15, beta=4.0,
                      cost_model="linear"):
    """
    Vectorized version of the MFG execution loop.

    FIX: edge_cost and the congestion mask are now real parameters instead of
    hardcoded for two specific edges. The original hardcoding (edge_cost[0,2]=5.5,
    edge_cost[1,3]=5.5, congestion tracked only on (0,1)/(2,3)) was specific to
    the small 4-node test network -- on a larger graph (e.g. Sioux Falls), it
    silently left every OTHER edge free and congestion-insensitive, which is not
    a hyperparameter issue, it's a structurally wrong cost model.

    Args:
        edge_cost: (N,N) tensor of free-flow/base cost per edge (this is t_e^0 when
                   cost_model="bpr"). Defaults to 0 everywhere if not provided.
        congest_all_edges: if True (default), every valid edge is congestion-sensitive.
        capacity: (N,N) tensor of per-edge capacity C_e, REQUIRED when cost_model="bpr".
                  Must be scaled consistently with your population mass units -- see
                  the mass/capacity scaling discussion (capacity_e_scaled = capacity_real / S,
                  where S is the same total-demand scale factor used to normalize mass).
        alpha, beta: standard BPR parameters (default 0.15, 4.0). Can be scalars or
                  (N,N) tensors if you have per-edge values from real data.
        cost_model: "linear" (original: edge_cost + E_total*5.0) or "bpr"
                  (t_e^0 * (1 + alpha*(x_e/C_e)^beta)).

    Returns: identical to before -- node_mass, edge_occ, final_flows, policies,
             W_cong_history, zeta_history.
    """
    K = len(solvers)
    H = solvers[0].H
    N = env.N
    device = env.device

    if edge_cost is None:
        edge_cost = torch.zeros((N, N), device=device) + 2.0

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
    capacity.sqrt_()

    if cost_model == "bpr" and capacity is None:
        raise ValueError(
            "cost_model='bpr' requires a real capacity tensor -- there's no safe "
            "default here, since capacity must be scaled to match your mass units "
            "(see the mass/capacity scaling discussion). Pass capacity=... explicitly."
        )

    adj_mask = env._adj_bool.float() if hasattr(env, "_adj_bool") else None
    if adj_mask is None:
        adj_mask = torch.zeros((N, N), device=device)
        for u in range(N):
            for v in env.get_neighbors(u):
                adj_mask[u, v] = 1.0

    # (K,N) mask: is_sink[k,u] = 1 iff u is group k's sink. Needed because Rule C's
    # "if u == sink: skip departure" behavior is per-GROUP, not global, so it can't
    # be folded into adj_mask alone once K > 1.
    is_sink = torch.zeros((K, N), device=device)
    sources = torch.zeros((K, N), device=device)
    for k in range(K):
        is_sink[k, env.groups[k]["sink"]] = 1.0
        sources[k, env.groups[k]["source"]] = env.groups[k]["mass"]

    not_sink_mask = 1.0 - is_sink
    zeta_history = torch.zeros((T, K, H, N, N), device=device)
    policies = torch.stack([solver.get_policy(solver.zeta) for solver in solvers], dim=0)  # (K,H,N,N)

    for t in range(T):
        node_mass_list = [sources]  # (K,N) at h=0
        edge_occ_list = [torch.zeros((K, N, N, W_max + 1), device=device)]

        W_cong_history = torch.zeros((H, N, N), device=device)
        # (3, H, N, N): [0] = congestion x_e/C_e, [1] = free-flow edge_cost, [2] = toll theta
        W_parts = torch.zeros((4, H, N, N), device=device)

        for h in range(H - 1):
            current_node_mass = node_mass_list[h]      # (K,N)
            current_edge_occ = edge_occ_list[h]         # (K,N,N,W+1)
            pol_h = policies[:, h, :, :]                # (K,N,N)

            # tentative_edge_traffic[u,v] = sum_k current_node_mass[k,u] * pol_h[k,u,v]
            # Replaces: for k: for u: for v in neighbors(u): tentative[u,v] += mass[u]*pol[k,u,v]
            tentative_edge_traffic = torch.einsum('ku,kuv->uv', current_node_mass * not_sink_mask, pol_h) * adj_mask

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

            delay = W_cong_history[h]  # (N,N)
            delay_floor = torch.clamp(torch.floor(delay).long(), 0, W_max)  # (N,N)
            delay_ceil = torch.clamp(torch.ceil(delay).long(), 0, W_max)
            weight_ceil = delay - delay_floor.float()
            weight_floor = 1.0 - weight_ceil

            # Rule A: shift existing tiers w -> w-1. This is just a slice-shift,
            # already fully vectorized in the original code -- unchanged here.
            next_edge_occ = torch.zeros((K, N, N, W_max + 1), device=device)
            next_edge_occ[..., :W_max] = current_edge_occ[..., 1:]

            # Rule C: new departures. moving_mass[k,u,v] = current_node_mass[k,u] * pol_h[k,u,v],
            # zeroed at any (k,u) where u is group k's sink (matching the original
            # "if u == sink: continue, no departure" branch).
            not_sink_mask = (1.0 - is_sink)  # (K,N), 1 where NOT the sink
            moving_mass = torch.einsum('ku,kuv->kuv', current_node_mass * not_sink_mask, pol_h) * adj_mask  # (K,N,N)

            # Scatter moving_mass*weight_floor into tier delay_floor[u,v], and
            # moving_mass*weight_ceil into tier delay_ceil[u,v], per edge (u,v).
            # Replaces: for u: for v in neighbors(u): edge_occ[...,delay_floor]+=...
            floor_onehot = F.one_hot(delay_floor, num_classes=W_max + 1).float()  # (N,N,W+1)
            ceil_onehot = F.one_hot(delay_ceil, num_classes=W_max + 1).float()    # (N,N,W+1)
            dispatch = (
                moving_mass.unsqueeze(-1) * weight_floor.unsqueeze(0).unsqueeze(-1) * floor_onehot.unsqueeze(0) +
                moving_mass.unsqueeze(-1) * weight_ceil.unsqueeze(0).unsqueeze(-1) * ceil_onehot.unsqueeze(0)
            )  # (K,N,N,W+1)
            next_edge_occ = next_edge_occ + dispatch

            # Sink mass that "stayed" (no departure) carries forward directly as free mass.
            sink_carry = current_node_mass * is_sink  # (K,N)

            # Rule B: release whatever ended up at tier 0 (mass that just decremented
            # from tier 1 via Rule A, AND mass freshly dispatched at delay==0 via Rule C),
            # matching the "+1" timing convention verified in the loop-based version.
            arriving_per_v = next_edge_occ[:, :, :, 0].sum(dim=1)  # sum over origin u -> (K,N)
            next_edge_occ[:, :, :, 0] = 0.0

            next_node_mass = sink_carry + arriving_per_v

            node_mass_list.append(next_node_mass)
            edge_occ_list.append(next_edge_occ)

        node_mass = torch.stack(node_mass_list, dim=1)   # (K, H, N)
        edge_occ = torch.stack(edge_occ_list, dim=1)      # (K, H, N, N, W_max+1)

        # --- THE BACKWARD PASS (OMD POLICY UPDATES) --- unchanged, one call per group
        new_policies = []
        for k, solver in enumerate(solvers):
            q = solver.compute_q_values_with_waiting_time(W_cong_history, W_max, theta_leader)
            solver.zeta = (1 - solver.eta * solver.tau) * solver.zeta + solver.eta * q
            new_policies.append(solver.get_policy(solver.zeta))
            zeta_history[t, k] = solver.zeta.clone()

        policies = torch.stack(new_policies, dim=0)

    final_flows = node_mass.sum(dim=0)  # (H, N)

    return node_mass, edge_occ, final_flows, list(policies), W_cong_history, zeta_history, W_parts


class GraphMFG_OMD_EdgeSolver_MultiGroup:
    """
    OMD Solver customized for Graph networks using dynamic delay-waiting times.
    Vectorized: the inner loops over nodes u and neighbors v have been replaced
    with masked tensor operations. The loop over h (backward induction) remains
    a genuine Python loop -- it is inherently sequential (each step depends on
    the result of the next), so it cannot be vectorized away.
    """
    def __init__(self, env, group_idx, eta=0.1, tau=0.01, T=200, alpha=1.0, H=30):
        self.env         = env
        self.k           = group_idx
        self.group       = env.groups[group_idx]
        self.eta         = eta
        self.tau         = tau
        self.T           = T
        self.H           = H
        self.alpha       = alpha
        self.device      = env.device
        self.N           = env.N

        self.zeta = torch.zeros((self.H, self.N, self.N), device=self.device)

        # Precompute the adjacency mask once (used by get_policy, compute_q_values,
        # and compute_exploitability) instead of calling env.get_neighbors in a loop
        # every time. adj_bool[u,v] = True iff v is a valid neighbor of u.
        self._adj_bool = torch.zeros((self.N, self.N), dtype=torch.bool, device=self.device)
        for u in range(self.N):
            for v in env.get_neighbors(u):
                self._adj_bool[u, v] = True

    def get_policy(self, zeta):
        """Generates a valid routing probability distribution over valid outbound links across all h."""
        # Vectorized: build the full (H,N,N) mask in one shot instead of looping over u.
        mask = torch.where(
            self._adj_bool.unsqueeze(0).expand(self.H, -1, -1),
            zeta,
            torch.full_like(zeta, float('-inf'))
        )
        policy = F.softmax(mask, dim=-1)
        # FIX: a node with ZERO outgoing edges (e.g. the sink) has an all -inf row,
        # and softmax(all -inf) = 0/0 = NaN. The original loop-based version simply
        # skips computing softmax for such nodes, leaving that row at its initial
        # all-zero value (harmless, since compute_q_values_with_waiting_time never
        # reads a sink's policy row anyway). Replicate that here.
        has_neighbors = self._adj_bool.any(dim=1)  # (N,) bool
        policy = torch.where(has_neighbors.view(1, -1, 1), policy, torch.zeros_like(policy))
        return policy

    def compute_q_values_with_waiting_time(self, W_cong_history, W_max=10, theta_leader=None):
        """
        Vectorized backward induction. The loop over h is kept (inherently
        sequential); the loops over u and over v in get_neighbors(u) have been
        replaced with masked (N,N) tensor operations computed once per h.
        """
        adj = self._adj_bool  # (N,N) bool
        N = self.N
        sink = self.env.groups[self.k]["sink"]

        transit_cost = -1.0
        # arrived_cost per node u: 0 at the sink, -1 everywhere else -- a (N,) vector.
        arrived_cost_vec = torch.full((N,), -1.0, device=self.env.device)
        arrived_cost_vec[sink] = 0.0

        V_list = [torch.zeros((N, W_max + 1), device=self.env.device),
                  torch.zeros((N, W_max + 1), device=self.env.device)]
        Q_list = []

        for h in reversed(range(self.H)):
            V_next = V_list[-1]

            # v_waiting[u,w] = transit_cost + V_next[u, w-1], for w=1..W_max.
            # Vectorized: this is just V_next shifted by one along the tier axis.
            v_waiting = transit_cost + V_next[:, :W_max]  # (N, W_max), columns are w=1..W_max

            # --- Choice-state (w=0) computation, vectorized over (u,v) ---
            delay = W_cong_history[h]  # (N,N)
            delay_floor = torch.clamp(torch.floor(delay).long(), 0, W_max)
            delay_ceil = torch.clamp(torch.ceil(delay).long(), 0, W_max)
            weight_ceil = delay - delay_floor.float()
            weight_floor = 1.0 - weight_ceil

            # future_val[u,v] = weight_floor[u,v]*V_next[v, delay_floor[u,v]]
            #                  + weight_ceil[u,v] *V_next[v, delay_ceil[u,v]]
            # Gather V_next at a DIFFERENT column per (u,v) pair: index with v along
            # dim 0 (broadcast across u) and the per-(u,v) delay index along dim 1.
            v_idx = torch.arange(N, device=self.env.device).view(1, N).expand(N, N)  # v_idx[u,v]=v
            future_floor = V_next[v_idx, delay_floor]  # (N,N)
            future_ceil = V_next[v_idx, delay_ceil]    # (N,N)
            future_val = weight_floor * future_floor + weight_ceil * future_ceil

            # q_actions[u,v] = arrived_cost[u] - theta_leader[u,v] + future_val[u,v],
            # masked to -inf wherever (u,v) isn't a real edge, AND the entire sink row
            # forced to -inf regardless of adjacency -- matching the original's
            # "if u == sink: skip the whole v-loop" behavior exactly (the original
            # never populates ANY q_actions[sink,v], even if sink has a self-loop edge).
            q_actions_full = arrived_cost_vec.unsqueeze(1) + future_val  # (N,N)
            valid_mask = adj.clone()
            valid_mask[sink, :] = False
            q_actions = torch.where(valid_mask, q_actions_full, torch.full_like(q_actions_full, float('-inf')))

            q_safe = torch.where(torch.isinf(q_actions), torch.zeros_like(q_actions), q_actions)

            # On-policy value per u: sum_v pi(v|u) * q_safe[u,v] + tau * entropy(pi(.|u))
            pi_full = self.get_policy(self.zeta)[h]  # (N,N)
            log_pi = torch.where(pi_full > 0, torch.log(pi_full), torch.zeros_like(pi_full))
            entropy = -(pi_full * log_pi).sum(dim=1)  # (N,)
            v_choice = (pi_full * q_safe).sum(dim=1) + self.tau * entropy  # (N,)
            v_choice[sink] = 0.0  # sink's choice-state value is always exactly 0

            V_current = torch.cat([v_choice.unsqueeze(1), v_waiting], dim=1)  # (N, W_max+1)
            q_actions_clean = torch.where(torch.isinf(q_actions), torch.zeros_like(q_actions), q_actions)

            V_list.append(V_current)
            Q_list.append(q_actions_clean)

        Q_list.reverse()
        Q = torch.stack(Q_list, dim=0)
        return Q

    def compute_exploitability(self, W_cong_history, W_max=10, theta_leader=None, h0=0, u0=None):
        """
        Vectorized exploitability check. Same semantics as the loop-based version:
        V_best (hard best-response, no entropy) vs V_pi (on-policy, no entropy bonus),
        evaluated at (h0, u0), defaulting to (0, this group's source).
        """
        N, H = self.N, self.H
        adj = self._adj_bool
        sink = self.env.groups[self.k]["sink"]
        if u0 is None:
            u0 = self.env.groups[self.k]["source"]

        arrived_cost_vec = torch.full((N,), -1.0, device=self.env.device)
        arrived_cost_vec[sink] = 0.0
        transit_cost = -1.0

        zero = lambda: torch.zeros((N, W_max + 1), device=self.env.device)
        Vb_list = [zero(), zero()]
        Vp_list = [zero(), zero()]

        v_idx = torch.arange(N, device=self.env.device).view(1, N).expand(N, N)

        for h in reversed(range(H)):
            Vb_next = Vb_list[-1]
            Vp_next = Vp_list[-1]

            vb_waiting = transit_cost + Vb_next[:, :W_max]
            vp_waiting = transit_cost + Vp_next[:, :W_max]

            delay = W_cong_history[h]
            df = torch.clamp(torch.floor(delay).long(), 0, W_max)
            dc = torch.clamp(torch.ceil(delay).long(), 0, W_max)
            wc = delay - df.float()
            wf = 1.0 - wc

            fb = wf * Vb_next[v_idx, df] + wc * Vb_next[v_idx, dc]
            fp = wf * Vp_next[v_idx, df] + wc * Vp_next[v_idx, dc]

            qb_full = arrived_cost_vec.unsqueeze(1)  + fb
            qp_full = arrived_cost_vec.unsqueeze(1)  + fp
            qb = torch.where(adj, qb_full, torch.full_like(qb_full, float('-inf')))
            qp = torch.where(adj, qp_full, torch.full_like(qp_full, float('-inf')))

            vb_choice = qb.max(dim=1).values  # (N,) hard best response
            vb_choice[sink] = 0.0

            pi_full = self.get_policy(self.zeta)[h]  # (N,N)
            qp_safe = torch.where(torch.isinf(qp), torch.zeros_like(qp), qp)
            vp_choice = (pi_full * qp_safe).sum(dim=1)  # (N,)
            vp_choice[sink] = 0.0

            Vb_list.append(torch.cat([vb_choice.unsqueeze(1), vb_waiting], dim=1))
            Vp_list.append(torch.cat([vp_choice.unsqueeze(1), vp_waiting], dim=1))

        Vb_list = Vb_list[2:]
        Vp_list = Vp_list[2:]
        Vb_list.reverse()
        Vp_list.reverse()
        V_best = torch.stack(Vb_list, dim=0)
        V_pi = torch.stack(Vp_list, dim=0)

        exploitability = (V_best[h0, u0, 0] - V_pi[h0, u0, 0]).item()
        return exploitability, V_best, V_pi