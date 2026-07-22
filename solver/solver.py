import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def solve_multigroup(env, solvers, T=200, W_max=10, theta_leader=None):
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
    K = len(solvers)
    H = solvers[0].H
    N = env.N
    edge_cost = torch.zeros((N, N), device=env.device)
    edge_cost[0, 2] = 5.5
    edge_cost[1, 3] = 5.5

    zeta_history = torch.zeros((T, K, H, N, N), device=env.device)

    # Initialize policies from the initial solver states
    policies = [solver.get_policy(solver.zeta) for solver in solvers]

    for t in range(T):
        # --- 1. THE FORWARD PASS ---
        node_mass_list = []
        edge_occ_list = []

        # Seed initial source distribution at h=0 (all mass free at source)
        first_node_mass = torch.zeros((K, N), device=env.device)
        for k in range(K):
            source = env.groups[k]["source"]
            first_node_mass[k, source] = env.groups[k]["mass"]
        node_mass_list.append(first_node_mass)
        edge_occ_list.append(torch.zeros((K, N, N, W_max + 1), device=env.device))

        W_cong_history = torch.zeros((H, N, N), device=env.device)

        # Chronological progression step-by-step
        for h in range(H - 1):
            current_node_mass = node_mass_list[h]
            current_edge_occ = edge_occ_list[h]

            # tentative: mass that WOULD depart this step, under current policy
            tentative_edge_traffic = torch.zeros((N, N), device=env.device)
            for k in range(K):
                active_mass = current_node_mass[k]
                for u in range(N):
                    for v in env.get_neighbors(u):
                        tentative_edge_traffic[u, v] += active_mass[u] * policies[k][h, u, v]

            # congestion now includes BOTH already-in-transit occupancy AND this step's new departures
            E_total_edges = current_edge_occ.sum(dim=(0, 3)) + tentative_edge_traffic

            E_total_edges_final = torch.zeros((N, N), device=env.device)
            E_total_edges_final[0, 1] = E_total_edges[0, 1]
            E_total_edges_final[2, 3] = E_total_edges[2, 3]

            W_cong_history[h] = torch.clamp(
                E_total_edges_final * 5.0 + edge_cost + theta_leader,
                min=0.0, max=float(W_max)
            )
            # ... rest of the step (Rule A/B/C) proceeds exactly as before, using this W_cong_history[h]

            # Build next frame out-of-place
            next_node_mass = torch.zeros((K, N), device=env.device)
            next_edge_occ = torch.zeros((K, N, N, W_max + 1), device=env.device)

            for k in range(K):
                sink = env.groups[k]["sink"]

                # Rule A: in-transit mass steps down one waiting tier, staying on the same edge
                for w in range(1, W_max + 1):
                    next_edge_occ[k, :, :, w - 1] = next_edge_occ[k, :, :, w - 1] + current_edge_occ[k, :, :, w]

                # Rule C: free mass at each node makes a new routing decision.
                # MOVED before Rule B (previously ran after): dispatching new departures
                # into next_edge_occ BEFORE releasing tier-0 means a fresh delay==0
                # dispatch lands in the same bucket Rule B is about to check, so it gets
                # released the same tick instead of waiting one extra tick.
                active_mass = current_node_mass[k]  # (N,)
                for u in range(N):
                    if u == sink:
                        next_node_mass[k, sink] = next_node_mass[k, sink] + active_mass[sink]
                        continue

                    for v in env.get_neighbors(u):
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

                # Rule B: mass with w=0 has arrived -> becomes free node_mass at destination v.
                # FIX: reads next_edge_occ (the FRESHLY updated tier-0 bucket, after both
                # Rule A's decrement and Rule C's new dispatches), not current_edge_occ (the
                # stale, one-tick-old snapshot). This catches BOTH mass that just decremented
                # down to tier 0 from tier 1, AND mass freshly dispatched at delay==0 -- both
                # become free THIS tick, matching a uniform "delay + 1" timing convention
                # instead of the previous "delay + 2" (which came from releasing only from
                # the stale snapshot, one tick behind what had actually just arrived at tier 0).
                arriving_per_v = next_edge_occ[k, :, :, 0].sum(dim=0)  # sum over origin u -> (N,)
                next_node_mass[k] = next_node_mass[k] + arriving_per_v
                next_edge_occ[k, :, :, 0] = 0.0  # already released -- don't let it linger and double-count next tick

            node_mass_list.append(next_node_mass)
            edge_occ_list.append(next_edge_occ)

        node_mass = torch.stack(node_mass_list, dim=1)   # (K, H, N)
        edge_occ = torch.stack(edge_occ_list, dim=1)      # (K, H, N, N, W_max+1)

        # --- 2. THE BACKWARD PASS (OMD POLICY UPDATES) ---
        new_policies = []
        for k, solver in enumerate(solvers):
            q = solver.compute_q_values_with_waiting_time(W_cong_history, W_max, theta_leader)

            solver.zeta = (1 - solver.eta * solver.tau) * solver.zeta + solver.eta * q
            new_policies.append(solver.get_policy(solver.zeta))
            zeta_history[t, k] = solver.zeta.clone()

        policies = new_policies

    # Aggregated spatial footprint: free mass at each node, summed over groups
    final_flows = node_mass.sum(dim=0)  # (H, N)

    return node_mass, edge_occ, final_flows, policies, W_cong_history, zeta_history


class GraphMFG_OMD_EdgeSolver_MultiGroup:
    """
    OMD Solver customized for Graph networks using dynamic delay-waiting times.
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

        # Strategy logits over state space: shape (H, N, N) representing edge maps from u -> v
        self.zeta = torch.zeros((self.H, self.N, self.N), device=self.device)

    def get_policy(self, zeta):
        """Generates a valid routing probability distribution over valid outbound links across all h."""
        # policy shape: (H, N, N)
        policy = torch.zeros_like(zeta)
        for u in range(self.N):
            neighbors = self.env.get_neighbors(u)
            if len(neighbors) > 0:
                # 1. Initialize a 2D mask of shape (H, N) with -inf
                mask = torch.full((self.H, self.N), float('-inf'), device=self.device)

                # 2. Assign the 2D slice of logits [H, len(neighbors)] to the valid columns.
                # FIX: no extra temperature division here. The paper's Fomd already folds
                # all entropy regularization into the (1 - eta*tau) decay applied to zeta
                # in the OMD update (see solve_multigroup). softmax(zeta) is defined at
                # temperature 1 -- dividing by an extra ad hoc constant here double-counts
                # (and, at /0.5, actively sharpens) the regularization, which is what was
                # collapsing the policy to a near-deterministic single path.
                mask[:, neighbors] = zeta[:, u, neighbors]
                policy[:, u, :] = F.softmax(mask, dim=-1)

                # 3. Softmax across the destination node dimension (dim=-1)
                # This ensures invalid nodes get 0 probability and valid ones sum to 1.0 for every timestep h
        return policy

    def compute_q_values_with_waiting_time(self, W_cong_history, W_max=10, theta_leader=None):
        import torch

        Q_list = []
        # 1. Start with the terminal boundary conditions at horizon H (all zeros)
        # Shape: (N, W_max + 1)
        # FIX: seed with TWO terminal frames, not one. The w==1 boundary case below needs
        # to look two frames ahead (h+2), not one, and both "H" and "H+1" are equally
        # past the end of the horizon -- there's nothing there but the same all-zero
        # boundary condition, so padding with two identical terminal frames keeps the
        # V_list[-2] lookup valid even for the first two iterations of the loop.
        V_list = [torch.zeros((self.N, W_max + 1), device=self.env.device),
                  torch.zeros((self.N, W_max + 1), device=self.env.device)]

        # 2. Backward induction loop over time steps
        for h in reversed(range(self.H)):
            V_next = V_list[-1]       # value matrix from step h + 1
            V_next_next = V_list[-2]  # value matrix from step h + 2

            # Build the tensors for the current timestep 'h' completely out-of-place
            V_current_nodes = []
            Q_current_nodes = []

            for u in range(self.N):
                # FIX: use this solver's own group sink, not group 0's. Harmless when
                # K=1, but silently wrong the moment groups have different sinks.
                sink = self.env.groups[self.k]["sink"]

                # FIX: "still traveling" (w > 0) and "already arrived, free to act" (w == 0)
                # are different situations and must use different costs. Previously both
                # used the same `running_cost = 0 if u == sink else -1`, which meant ANY
                # edge landing on the sink -- e.g. a long, congested edge like 1->3 -- was
                transit_cost = -1.0
                arrived_cost = 0.0 if u == sink else -1.0

                # --- Rule A: Handle locked waiting states (w > 0) out-of-place ---
                # REVERTED to the simple, uniform +1 formula (no w==1 special case).
                # Instead of patching the backward induction to match a "+2" timing
                # convention, the forward simulation itself (solve_multigroup /
                # simulate_forward_with_policy) was changed so a delay==0 dispatch
                # becomes free the SAME tick, not one tick later. With that forward-sim
                # fix in place, this uniform +1 formula is what's actually consistent.
                v_waiting = [transit_cost + V_next[u, w - 1] for w in range(1, W_max + 1)]

                # --- Rule B: Handle choice states (w == 0) ---
                # FIX: initialize to -inf, not 0. Previously, non-neighbor "actions" were
                # left at exactly 0.0 and then fed into logsumexp alongside real, typically
                # very negative, path costs. Since 0 > any real cumulative cost, those
                # phantom invalid actions dominated the soft-max and pinned V near 0
                # everywhere -- washing out the true cost differences between routes and
                # removing the OMD update's incentive to ever deviate off the first path
                # it locked onto. Masking with -inf restricts the soft-max to only the
                # actual available actions, exactly as get_policy already does.
                q_actions = torch.full((self.N,), float('-inf'), device=self.env.device)

                if u == sink:
                    v_choice = torch.tensor([0.0], device=self.env.device)
                else:
                    for v in self.env.get_neighbors(u):
                        delay_continuous = W_cong_history[h, u, v]
                        delay_floor = torch.clamp(torch.floor(delay_continuous).long(), 0, W_max)
                        delay_ceil = torch.clamp(torch.ceil(delay_continuous).long(), 0, W_max)

                        weight_ceil = delay_continuous - delay_floor.float()
                        weight_floor = 1.0 - weight_ceil

                        # REVERTED to the simple, uniform lookup (no delay==0 special
                        # case) -- consistent with the forward-sim fix described above.
                        future_val = (weight_floor * V_next[v, delay_floor] +
                                      weight_ceil * V_next[v, delay_ceil])

                        q_actions[v] = arrived_cost - theta_leader[u, v] + future_val

                    # FIX (approach 2, not approach 1): use the ON-POLICY value of the
                    # actual current policy pi = softmax(zeta), not the idealized
                    # instantaneously-optimal soft-Bellman value via logsumexp. These two
                    # only coincide once pi has converged to exactly softmax(Q/tau); zeta
                    # is an EMA of past Q so pi lags behind that during essentially all of
                    # training. Using the on-policy value here matches the paper's own
                    # Vtau_h definition (expectation under the actual pi, plus tau*H(pi))
                    # and keeps this backward pass consistent with the same pi that
                    # generated W_cong_history in the forward pass -- which is what the
                    # AMID adjoint method (Lemma 2) requires for its gradient to be exact.
                    pi_u = self.get_policy(self.zeta)[h, u, :]

                    q_safe = torch.where(torch.isinf(q_actions), torch.zeros_like(q_actions), q_actions)
                    log_pi = torch.where(pi_u > 0, torch.log(pi_u), torch.zeros_like(pi_u))
                    entropy = -(pi_u * log_pi).sum()

                    v_choice = ((pi_u * q_safe).sum() + self.tau * entropy).unsqueeze(0)

                # Combine choice (w=0) and waiting tiers (w > 0) out-of-place for node 'u'
                v_node = torch.cat([v_choice, torch.stack(v_waiting, dim=0)], dim=0)

                V_current_nodes.append(v_node)

                # Replace -inf placeholders with 0 before storing Q. The -inf mask is only
                # needed to get a correct V; get_policy already masks non-neighbor zeta
                # entries itself, so feeding -inf/nan into zeta via the OMD update
                # (zeta = (1-eta*tau)*zeta + eta*Q) is unnecessary risk for no benefit.
                q_actions_clean = torch.where(
                    torch.isinf(q_actions), torch.zeros_like(q_actions), q_actions
                )
                Q_current_nodes.append(q_actions_clean)

            # Stack the nodes together to form the full spatial frame for time 'h'
            V_list.append(torch.stack(V_current_nodes, dim=0))
            Q_list.append(torch.stack(Q_current_nodes, dim=0))

        # 3. Assemble full sequence tensors cleanly out-of-place
        Q_list.reverse()
        Q = torch.stack(Q_list, dim=0)

        return Q

    def compute_exploitability(self, W_cong_history, W_max=10, theta_leader=None, h0=0, u0=None):
        """
        Standard mean-field exploitability check (matches the paper's Fig 2 diagnostic).

        Holds the given W_cong_history FIXED -- i.e. assumes every other agent keeps
        behaving exactly as observed -- and asks: how much better could a single
        deviating unit of mass do by best-responding instead of following the
        current policy self.zeta?

        Runs the SAME corrected backward induction as compute_q_values_with_waiting_time
        (same transit/arrived cost split, same -inf action masking, same two-frame
        w==1 boundary fix) but bootstraps two parallel value functions instead of one:
          - V_best: hard best-response value (max over actions, no entropy)
          - V_pi:   actual on-policy value of the current policy (weighted by pi,
                    no entropy bonus -- entropy is a regularizer artifact, not real cost)

        Args:
            W_cong_history: (H, N, N) congestion/delay field to hold fixed
            W_max: same wait-tier discretization used to generate W_cong_history
            theta_leader: (N, N) leader incentives in effect when W_cong_history was produced
            h0, u0: which (time, node) to report exploitability at. Defaults to
                    (0, this group's source) -- the value relevant to a mass unit
                    about to start its journey under the observed equilibrium.

        Returns:
            exploitability: scalar >= 0. ~0 means self.zeta is (near) a Nash/Wardrop
                             equilibrium under this congestion field; larger means more
                             room for a unilateral deviation to do better.
            V_best, V_pi: (H, N, W_max+1) full value tables, in case you want to
                          inspect exploitability at other states too.
        """
        import torch
        N, H = self.N, self.H
        sink = self.env.groups[self.k]["sink"]
        if u0 is None:
            u0 = self.env.groups[self.k]["source"]

        zero = lambda: torch.zeros((N, W_max + 1), device=self.env.device)
        Vb_list = [zero(), zero()]
        Vp_list = [zero(), zero()]

        def wait_chain(V_next, V_next2, u, transit_cost):
            # REVERTED to the simple, uniform +1 formula -- see compute_q_values_with_waiting_time
            # for why (the forward simulation was fixed instead of the backward induction).
            return [transit_cost + V_next[u, w - 1] for w in range(1, W_max + 1)]

        for h in reversed(range(H)):
            Vb_next, Vb_next2 = Vb_list[-1], Vb_list[-2]
            Vp_next, Vp_next2 = Vp_list[-1], Vp_list[-2]

            Vb_nodes, Vp_nodes = [], []
            for u in range(N):
                transit_cost = -1.0
                arrived_cost = 0.0 if u == sink else -1.0

                vb_waiting = wait_chain(Vb_next, Vb_next2, u, transit_cost)
                vp_waiting = wait_chain(Vp_next, Vp_next2, u, transit_cost)

                if u == sink:
                    vb_choice = torch.tensor([0.0], device=self.env.device)
                    vp_choice = torch.tensor([0.0], device=self.env.device)
                else:
                    qb = torch.full((N,), float('-inf'), device=self.env.device)
                    qp = torch.full((N,), float('-inf'), device=self.env.device)
                    for v in self.env.get_neighbors(u):
                        delay = W_cong_history[h, u, v]
                        df = torch.clamp(torch.floor(delay).long(), 0, W_max)
                        dc = torch.clamp(torch.ceil(delay).long(), 0, W_max)
                        wc = delay - df.float()
                        wf = 1.0 - wc

                        # REVERTED to the simple, uniform lookup -- see
                        # compute_q_values_with_waiting_time for why.
                        fb = wf * Vb_next[v, df] + wc * Vb_next[v, dc]
                        fp = wf * Vp_next[v, df] + wc * Vp_next[v, dc]

                        qb[v] = arrived_cost - theta_leader[u, v] + fb
                        qp[v] = arrived_cost - theta_leader[u, v] + fp

                    # Best response: hard max over available actions, no entropy --
                    # this is the real quantity a rational deviator would chase.
                    vb_choice = qb.max().unsqueeze(0)

                    # On-policy: actual expected value under the CURRENT policy,
                    # again no entropy bonus (we want real achieved cost here, not
                    # the regularized surrogate used inside the OMD update itself).
                    pi_u = self.get_policy(self.zeta)[h, u, :]
                    qp_safe = torch.where(torch.isinf(qp), torch.zeros_like(qp), qp)
                    vp_choice = (pi_u * qp_safe).sum().unsqueeze(0)

                Vb_nodes.append(torch.cat([vb_choice, torch.stack(vb_waiting)]))
                Vp_nodes.append(torch.cat([vp_choice, torch.stack(vp_waiting)]))

            Vb_list.append(torch.stack(Vb_nodes, dim=0))
            Vp_list.append(torch.stack(Vp_nodes, dim=0))

        Vb_list = Vb_list[2:]  # drop the two padding terminal frames
        Vp_list = Vp_list[2:]
        Vb_list.reverse()
        Vp_list.reverse()
        V_best = torch.stack(Vb_list, dim=0)  # (H, N, W_max+1)
        V_pi = torch.stack(Vp_list, dim=0)

        exploitability = (V_best[h0, u0, 0] - V_pi[h0, u0, 0]).item()
        return exploitability, V_best, V_pi
