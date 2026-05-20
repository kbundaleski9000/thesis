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
            reward1 = -self.alpha * (L_total[h])

            reward = reward1 - reward1
            reward[0,1] = reward1[0,1]  # Incentivize the gap cell
            reward[2,3] = reward1[2,3]  # Incentivize the gap cell
            reward += theta
            

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


# ─────────────────────────────────────────────────────────────
# 3.  MULTI-GROUP COORDINATE SOLVE
# ─────────────────────────────────────────────────────────────

def solve_multigroup(solvers, theta_list):
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

    T = solvers[0].T
    for t in range(T):
        # Forward: each group computes its own flow
        flows   = [s.compute_population_flow(policies[k])
                   for k, s in enumerate(solvers)]
        L_total = torch.stack(flows).sum(dim=0)   # (H, rows, cols)

        # Backward + OMD update for each group
        new_policies = []
        for k, solver in enumerate(solvers):
            q = solver.compute_q_values(
                flows[k], L_total, policies[k], theta_list[k]
            )
            solver.zeta = (
                (1 - solver.eta * solver.tau) * solver.zeta + solver.eta * q
            )
            new_policies.append(solver.get_policy(solver.zeta))
        policies = new_policies

    # Final flows with converged policies
    flows   = [s.compute_population_flow(policies[k])
               for k, s in enumerate(solvers)]
    L_total = torch.stack(flows).sum(dim=0)
    return policies, flows, L_total