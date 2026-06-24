import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import heapq


class PMFG_OMD_Solver_MultiGroup:
    """
    One solver instance per group.
    compute_q_values now takes L_total (all groups combined).
    """
    def __init__(self, env, group_idx, eta=0.1, tau=0.01, T=100,
                 alpha=1.0, H=8):
        self.env        = env
        self.k          = group_idx
        self.group      = env.groups[group_idx]
        self.eta        = eta
        self.tau        = tau
        self.T          = T
        self.H          = H
        self.alpha      = alpha
        self.num_actions = 5
        self.device     = env.device

        # Logits: (H, rows, cols, actions)
        self.zeta = torch.zeros(
            (H, env.rows, env.cols, self.num_actions), device=self.device
        )

    def get_policy(self, zeta):
        return F.softmax(zeta, dim=-1)

    # ── Forward pass ──────────────────────────────────────────
    def compute_population_flow(self, policy):
        """Propagate this group's mass forward under policy."""
        src_r, src_c = self.group["source"]
        mass = self.group.get("mass", 1.0)

        L_h = torch.zeros((self.env.rows, self.env.cols), device=self.device)
        L_h[src_r, src_c] = mass
        all_L = [L_h]

        rows_idx = torch.arange(self.env.rows, device=self.device).view(-1, 1)
        cols_idx = torch.arange(self.env.cols, device=self.device).view( 1,-1)

        for h in range(self.H - 1):
            next_L = torch.zeros_like(L_h)
            for a in range(self.num_actions):
                nr = (rows_idx + self.env.dr[a]).clamp(0, self.env.rows - 1)
                nc = (cols_idx + self.env.dc[a]).clamp(0, self.env.cols - 1)
                mask = self.env.obstacles[nr, nc]
                nr = torch.where(mask, rows_idx, nr)
                nc = torch.where(mask, cols_idx, nc)

                mass_move = all_L[h] * policy[h, :, :, a]
                next_L = next_L.index_put(
                    (nr.flatten(), nc.flatten()), mass_move.flatten(), accumulate=True
                )
            all_L.append(next_L)

        return torch.stack(all_L)   # (H, rows, cols)

    # ── Backward pass ─────────────────────────────────────────
    def compute_q_values(self, L_self, L_total, policy, theta):
        """
        Backward pass.
        L_self  : this group's own flow  (H, rows, cols)
        L_total : sum over all groups    (H, rows, cols)  ← coupling term
        """
        q_list = [None] * self.H
        V_next = torch.zeros((self.env.rows, self.env.cols), device=self.device)

        rows_idx = torch.arange(self.env.rows, device=self.device).view(-1, 1)
        cols_idx = torch.arange(self.env.cols, device=self.device).view( 1,-1)

        for h in reversed(range(self.H)):
            # Congestion felt from TOTAL density (all groups)
            theta = torch.reshape(theta, (self.env.rows, self.env.cols))
            reward = -self.alpha * (L_total[h])
            reward += theta  # Add incentive signal from leader
            

            current_q = torch.zeros(
                (self.env.rows, self.env.cols, self.num_actions), device=self.device
            )
            for a in range(self.num_actions):
                nr = (rows_idx + self.env.dr[a]).clamp(0, self.env.rows - 1)
                nc = (cols_idx + self.env.dc[a]).clamp(0, self.env.cols - 1)
                mask = self.env.obstacles[nr, nc]
                tr = torch.where(mask, rows_idx, nr)
                tc = torch.where(mask, cols_idx, nc)
                current_q[:, :, a] = reward + V_next[tr, tc]

            q_list[h] = current_q
            pi_h    = policy[h]
            entropy = -torch.sum(pi_h * torch.log(pi_h + 1e-9), dim=-1)
            V_next  = torch.sum(pi_h * current_q, dim=-1) + self.tau * entropy

        return torch.stack(q_list)   # (H, rows, cols, actions)


class GraphMFG_OMD_EdgeSolver_MultiGroup:
    def __init__(self, env, group_idx, eta=0.1, tau=0.01, T=100, alpha=1.0, H=8):
        self.env = env
        self.k = group_idx
        self.group = env.groups[group_idx]
        self.eta = eta
        self.tau = tau
        self.T = T
        self.H = H
        self.alpha = alpha
        self.device = env.device
        self.N = env.N

        # Precompute base theta as an edge matrix layout: (K, N, 1) -> expands to (K, N, N)
        self.base_thetas = - torch.zeros((self.env.K, self.env.N, self.env.N), dtype=torch.float32, device=env.device)

        for k in range(self.env.K):
            sink_idx = self.env.groups[k]["sink"]
            self.base_thetas[k, sink_idx, sink_idx] = 0.0  # No congestion cost at the sink

        # Strategy matrix: (H, Nodes_From, Nodes_To)
        self.zeta = torch.zeros((H, self.N, self.N), device=self.device)

    def get_policy(self, zeta):
        return F.softmax(zeta + self.env.log_M, dim=-1)

    def compute_population_flow(self, policy):
        """Propagate group mass forward (Remains identical to node-level setup)."""
        src_node = self.group["source"]
        mass = self.group.get("mass", 1.0)

        L_h = torch.zeros(self.N, device=self.device)
        L_h[src_node] = mass
        all_L = [L_h]

        for h in range(self.H - 1):
            next_L = torch.matmul(policy[h].t(), all_L[h])
            all_L.append(next_L)

        return torch.stack(all_L)  # (H, N)

    def compute_edge_traffic(self, policy, flow):
        """Return edge utilization matrices E_h[i,j] = flow_h[i] * policy_h[i,j]."""
        return flow.unsqueeze(-1) * policy

    def compute_q_values(self, L_self, E_total_all_groups, policy, theta):
        """
        Backward pass across edge topologies.

        E_total_all_groups : (H, N, N) matrix of collective edge traffic across ALL groups
        theta              : (N, N) matrix of link edge rewards proposed by the Leader
        """
        q_list = [None] * self.H
        V_next = torch.zeros(self.N, device=self.device)
        ones_N = torch.ones(self.N, device=self.device)

        for h in reversed(range(self.H)):
            # 1. Isolate global edge traffic for this specific time step h
            E_total_h = E_total_all_groups[h]  # Shape: (N, N)

            E_total_final = torch.zeros_like(E_total_h)
            E_total_final[0,1] = E_total_h[0,1]
            E_total_final[2,3] = E_total_h[2,3]

            R_edge = self.base_thetas[self.k] - self.alpha * E_total_final + theta 

            # 3. Dynamic programming addition: Q_ij = R_ij + V_j
            current_q = R_edge + torch.outer(ones_N, V_next)

            # Enforce graph constraints by zeroing illegal edges
            current_q = current_q * self.env.M
            q_list[h] = current_q

            # Shannon Entropy calculation per node row
            pi_h = policy[h]
            entropy = -torch.sum(pi_h * torch.log(pi_h + 1e-9), dim=-1)

            # Contract expected Q values along columns and add exploratory entropy
            expected_q = torch.diagonal(torch.matmul(pi_h, current_q.t()))
            V_next = expected_q + self.tau * entropy
            

        return torch.stack(q_list)  # Dimensions: (H, N, N)

    # Multi-Group Coordinate Solve function remains structurally the same, 
    # but it operates on optimized node matrices without flattening loops.


# ─────────────────────────────────────────────────────────────
# 3.  MULTI-GROUP COORDINATE SOLVE
# ─────────────────────────────────────────────────────────────

def solve_multigroup(solvers, theta_list, number_epochs = 200):
    """
    Run T steps of joint OMD across all K groups.
    Returns:
        policies : list of (H, rows, cols, actions) tensors
        flows    : list of (H, rows, cols) tensors
        L_total  : (H, rows, cols) sum of all flows
    """
    K = len(solvers)

    # Detach zetas to break old graph
    for solver in solvers:
        solver.zeta = solver.zeta.detach().requires_grad_(True)

    # Initialise policies from current zetas
    policies = [s.get_policy(s.zeta) for s in solvers]

    T = number_epochs
    print(f"Running multi-group OMD for T={T} steps...")
    for t in range(T):
        # Forward: each group computes its own flow
        flows = [s.compute_population_flow(policies[k])
                 for k, s in enumerate(solvers)]
        L_total = torch.stack(flows).sum(dim=0)

        # Compute edge-level traffic matrices for solvers that use edge rewards
        edge_flows = [None] * K
        for k, solver in enumerate(solvers):
            if hasattr(solver, "compute_edge_traffic"):
                edge_flows[k] = solver.compute_edge_traffic(policies[k], flows[k])

        E_total_edges = None
        if any(ef is not None for ef in edge_flows):
            E_total_edges = torch.stack([ef for ef in edge_flows if ef is not None]).sum(dim=0)

        # Backward + OMD update for each group
        new_policies = []
        for k, solver in enumerate(solvers):
            if edge_flows[k] is not None:
                q = solver.compute_q_values(
                    flows[k], E_total_edges, policies[k], theta_list[k]
                )
            else:
                q = solver.compute_q_values(
                    flows[k], L_total, policies[k], theta_list[k]
                )
            solver.zeta = (
                (1 - solver.eta * solver.tau) * solver.zeta + solver.eta * q
            )
            new_policies.append(solver.get_policy(solver.zeta))
        policies = new_policies

        if t == T - 1:
            print(f"Step {t+1}/{T} completed. Final policies and flows computed.")

    # Final flows with converged policies
    flows   = [s.compute_population_flow(policies[k])
               for k, s in enumerate(solvers)]
    L_total = torch.stack(flows).sum(dim=0)
    print("Multi-group OMD completed.")
    return policies, flows, L_total