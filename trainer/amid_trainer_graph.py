import torch
import torch.nn as nn
import torch.optim as optim
from leader.leadernet import GraphLeaderIncentiveNet, GraphLeaderIncentiveNetCNN
from solver.solver import solve_multigroup

class GraphEdgeMFG_Trainer:
    def __init__(self, env, solvers, leader_lr=0.01):
        self.env = env
        self.solvers = solvers
        self.H = solvers[0].H
        self.initial_zetas = [s.zeta.clone() for s in solvers]

        # Initialize Leader Network mapping historical edge traffic flows
        self.leader_nets = GraphLeaderIncentiveNetCNN(env.N, env.K, solvers[0].H).to(env.device)
        self.optimizer = optim.Adam(self.leader_nets.parameters(), lr=leader_lr)
        self.OMDsteps = 50

        self.edge_cost = torch.zeros((self.env.N, self.env.N), device=self.env.device) + 2.0
        # adj is already {0, 1} -- no normalization needed.
        capacity = torch.zeros((self.env.N, self.env.N))
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
        capacity.sqrt_()  # Apply sqrt twice to match the original code's behavior

        self.capacity = capacity

        
    def compute_social_loss(self, final_flows, W_cong_history):
        """
        Calculates total travel times as social loss.
        
        Under the indicator reward structure, the loss is the integration of all 
        population mass that hasn't arrived at its destination sink yet.
        """
        H, N = final_flows.shape
        total_social_loss = 0.0

        for h in range(H):
            for k in range(self.env.K):
                cost = self.env.groups[k]["mass"] - final_flows[h, self.env.groups[k]["sink"]]
                total_social_loss += cost

        return total_social_loss
    
    def compute_congestion_penalty_loss(self, edge_occ, normalize_by_capacity=True):
        """
        Load-balancing / peak-congestion penalty: sum_h sum_{u,v} (x_e[h,u,v])^2,
        or (x_e[h,u,v] / capacity[u,v])^2 if normalize_by_capacity=True.
    
        NOTE: this requires edge_occ (K, H, N, N, W_max+1), NOT W_cong_history --
        W_cong_history is DELAY, not flow. edge_occ is what solve_multigroup
        already returns but train_step currently discards (`_, _, final_flows_new,
        ...`). You'll need to keep it: change that line to capture edge_occ_new,
        and pass it into this function instead of/alongside W_cong_history.
    
        Args:
            edge_occ: (K, H, N, N, W_max+1) tensor from solve_multigroup's return.
            normalize_by_capacity: if True, divides by self.capacity (per-edge)
                before squaring -- requires self.capacity to be set (an (N,N)
                tensor), the same way self.edge_cost is already set in __init__.
                If False, computes raw flow^2 (no capacity normalization).
    
        Returns: scalar tensor, differentiable w.r.t. whatever edge_occ traces
            back to (e.g. theta_leader, if this is called on a fresh, live
            edge_occ inside compute_loss_derivative_wrt_theta_direct-style code --
            NOT on a .detach()'d one from train_step's no_grad() forward pass).
        """
        # x_e[h,u,v] = total mass occupying edge (u,v) at time h, summed over
        # groups (K) and wait-tiers (last dim) -- same reduction used throughout
        # this project's diagnostics (e.g. "occ_per_edge = edge_occ.sum(dim=(0,-1))").
        edge_flows = edge_occ.sum(dim=(0, -1))  # (H, N, N)
    
        if normalize_by_capacity:
            if not hasattr(self, "capacity"):
                raise AttributeError(
                    "compute_congestion_penalty_loss(normalize_by_capacity=True) "
                    "requires self.capacity (an (N,N) tensor) to be set in __init__, "
                    "the same way self.edge_cost already is."
                )
            # clamp(min=1e-6) avoids division by zero on non-edges / zero-capacity entries
            ratio = edge_flows / self.capacity.clamp(min=1e-6)
        else:
            ratio = edge_flows
    
        # Mask to valid edges only -- non-edges have edge_flows=0 anyway (nothing
        # can occupy a nonexistent edge), so squaring zero contributes zero; this
        # mask is mostly a safety net in case capacity has stray nonzero entries
        # on non-edges.
        adj_mask = self.env.A.detach()  # (N,N), 0/1
        penalty = (ratio.pow(2) * adj_mask.unsqueeze(0)).sum()
    
        return penalty
    
    
    def congestion_loss(self, edge_occ, lam=1.0, normalize_by_capacity=True):
        """
        total_travel_time + lam * congestion_penalty, letting you trade off
        between minimizing total travel time (the verified-correct existing
        objective) and discouraging any single edge from becoming severely
        overloaded relative to its capacity.
    
        lam=0 recovers the existing compute_social_loss exactly. Start small
        (e.g. 0.01-0.1) and increase only if you actually want load-balancing
        to visibly trade off against raw travel time -- a large lam can pull
        the solution away from the true travel-time-minimizing optimum, which
        is the whole point of using this as a deliberate, tunable regularizer
        rather than always defaulting to it.
        """
        congestion_penalty = self.compute_congestion_penalty_loss(edge_occ, normalize_by_capacity)
        return lam * congestion_penalty
    
    def _prepare_input(self, final_flows, W_cong_history):
        """
        Normalized version of _prepare_input. Each feature group is scaled to a
        consistent, well-behaved range using KNOWN theoretical bounds (not on-the-fly
        batch statistics, which would shift around as training progresses and make
        the leader's input distribution non-stationary on top of everything else
        already changing during training).
    
        Ranges used:
        - final_flows:      [0, total_mass]  -> normalize by total mass (usually 1.0)
        - W_cong_history:    [0, W_max]        -> normalize by W_max
        - edge_cost:         [0, edge_cost.max()] -> normalize by its own max (computed once)
        - adj:                already {0,1}     -> left as-is, no normalization needed
        """
        adj = self.env.A.detach()  # (N, N)
    
        # Total mass across all groups -- the natural upper bound for final_flows.
        total_mass = sum(g["mass"] for g in self.env.groups)
        final_flows_norm = final_flows 
    
        # W_max isn't currently stored on the trainer -- pass it in or store it at
        # __init__ time. Using self.W_max here; add `self.W_max = W_max` in __init__.
        W_cong_norm = W_cong_history   # -> [0, 1]
    
        # edge_cost is fixed for the whole training run -- normalize by its own max,
        # computed once (e.g. in __init__) rather than recomputed every call.
        edge_cost_norm = self.edge_cost   # -> [0, 1]

        capacity_ = self.capacity
    
    
        features_final = torch.cat([
            W_cong_norm.flatten(),
            final_flows_norm.flatten(),
            edge_cost_norm.flatten(),
            adj.flatten(),
            capacity_.flatten()
        ], dim=0)
    
        return features_final

    def train_step_ds(self, flows_previous, W_cong_history, trainstep):
            """
            Executes one policy optimization step for the leader infrastructure network.
            """
            self.optimizer.zero_grad()

            if trainstep == 1:
            # OPTION A: If you want the network to train on iteration 1, 
                # feed dummy/zero placeholder features through your network:
                spatial_flows = torch.zeros((self.env.K, self.env.H, self.env.N), device=self.env.device)
                base_cong = torch.zeros((self.env.H, self.env.N, self.env.N), device=self.env.device)
                
                inp = self._prepare_input(spatial_flows, base_cong)
                theta_leader = self.leader_nets(inp)

            # OPTION B: If you strictly want a hardcoded flat zero matrix for step 1,
            # you MUST explicitly tell PyTorch to track its gradients:
            # theta_leader = torch.zeros((self.env.N, self.env.N), device=self.env.device, requires_grad=True)

            else:
                # 1. Prepare the input features for the leader network
                spatial_flows = flows_previous.sum(dim=-1)  # Sum over waiting time dimension
                inp = self._prepare_input(spatial_flows, W_cong_history)
                theta_leader = self.leader_nets(inp)

            # 5. Simulate Agent Response (Inner MFG loop)
            flows_new, final_spatial_flows, policies_new, W_cong_history_new, zeta_history_new = solve_multigroup(
                self.env, self.solvers, T=50, W_max=100, theta_leader=theta_leader
            )

            # 6. Compute Loss and Backpropagate
            social_loss = self.compute_social_loss(flows_new, W_cong_history)
            
            social_loss.backward()
            self.optimizer.step()

            return social_loss, flows_new, W_cong_history_new
    
    def compute_vector_jacobian_product_zeta(self, a_t, zeta_t, k_group, theta_leader):
        """
        Computes the exact Vector-Jacobian Product (VJP): a_t * (dF / dzeta_t)
        where F(\theta, \zeta_t) is the Q-value output operator.
        """
        # 1. Ensure the history snapshot requires a gradient for this specific VJP tracking operation
        zeta_target = zeta_t[k_group].detach().clone().requires_grad_(True)
        policy_target = torch.zeros(self.env.K, self.env.H, self.env.N, self.env.N, device=self.env.device)
        
        # 2. Map zeta to policies via the correct group's Softmax operation
        for k in range(self.env.K):
            if k == k_group:
                policy_target[k] = self.solvers[k].get_policy(zeta_target)
            else:
                # For other groups, we can use their final zeta without gradient tracking
                policy_target[k] = self.solvers[k].get_policy(zeta_t[k].detach())
        
        # 3. Simulate forward passing the localized target policy and current leader rules
        flows_sim, final_flows_sim, _, W_cong_sim = self.env.simulate_forward_with_policy(
            policy_sim=policy_target, 
            theta_leader=theta_leader
        )
        
        # 4. Compute the Q-values (this represents F(\theta, \zeta_t) in your algorithm)
        # W_max is available via self.solvers[0].W_max or passed in
        q_out = self.solvers[k_group].compute_q_values_with_waiting_time(
            W_cong_sim, W_max=100, theta_leader=theta_leader
        )
        
        # 5. Compute the Vector-Jacobian Product directly using PyTorch's autograd tool
        # grad_outputs=a_t tells PyTorch to calculate a_t * (dq_out / dzeta_input)
        vjp = torch.autograd.grad(
            outputs=q_out,
            inputs=zeta_target,
            grad_outputs=a_t,
            retain_graph=True,
            create_graph=False
        )[0]
        
        return vjp
    
    def compute_loss_derivative_wrt_zeta(self, final_zeta, theta_leader_val):
        """
        Line 4 of Algorithm 1: dG/dzeta_T, for the full (K,H,N,N) tensor (K=1 for now).
        """
        zeta_target = final_zeta.detach().clone().requires_grad_(True)

        node_mass, edge_occ, final_flows, policies, W_cong_history = self.env.simulate_forward_with_policy(
            zeta=zeta_target,
            theta_leader=theta_leader_val,
            W_max=100   # fix: was silently defaulting to 3
        )

        loss_G = self.compute_social_loss(final_flows, W_cong_history)  # use freshly computed W_cong, not stale arg
        grad_zeta = torch.autograd.grad(outputs=loss_G, inputs=zeta_target)[0]

        return grad_zeta.detach()   # shape (K,H,N,N)
    
    def compute_loss_derivative_wrt_theta_direct(self, final_zeta, theta_leader_val):
        """
        Line 4 of Algorithm 1 (the OTHER half): s_{T+1} = d_theta G(theta, zeta_{T+1}),
        holding zeta_{T+1} FIXED.

        This is NOT zero. compute_social_loss() itself doesn't take theta as an argument,
        but the final_flows fed into it comes from a fresh simulate_forward_with_policy
        call that USES theta_leader directly, inside its congestion formula. So theta has
        a genuine direct effect on G, separate from its indirect effect through the zeta
        trajectory (which the rest of the adjoint recursion below already accounts for).
        Previously this term was silently dropped by initializing s_adjoint to zeros.
        """
        zeta_fixed = final_zeta.detach()
        theta_target = theta_leader_val.detach().clone().requires_grad_(True)

        node_mass, edge_occ, final_flows, policies, W_cong_history = self.env.simulate_forward_with_policy(
            zeta=zeta_fixed,
            theta_leader=theta_target,
            W_max=100
        )

        loss_G = self.compute_social_loss(final_flows, W_cong_history)
        grad_theta = torch.autograd.grad(outputs=loss_G, inputs=theta_target)[0]

        return grad_theta.detach()   # shape (N, N)
    
    def compute_exact_vjp_zeta(self, a_t, zeta_t, theta_leader_val):
        """
        Line 6 of Algorithm 1 (AMID): Exact Vector-Jacobian Product a_t * (dQ / dzeta_t)
        """
        # Create an isolated leaf tensor for this specific OMD time-step slice
        zeta_target = zeta_t.detach().clone().requires_grad_(True)
        
        # 2. Map those local policies to the corresponding congestion layout
        _, _, _, _, W_cong_sim = self.env.simulate_forward_with_policy(
            zeta=zeta_target,
            theta_leader=theta_leader_val
        )
        
        # 3. Reconstruct the local Q-value state evaluations across all groups
        q_outputs = []
        for k in range(self.env.K):
            q_k = self.solvers[k].compute_q_values_with_waiting_time(
                W_cong_sim, W_max=100, theta_leader=theta_leader_val
            )
            q_outputs.append(q_k)
        q_outputs = torch.stack(q_outputs, dim=0)
        
        # 4. Compute the exact isolated Vector-Jacobian Product
        # FIX: We add [0] at the end to unpack the single Tensor out of the autograd tuple!
        vjp = torch.autograd.grad(
            outputs=q_outputs,
            inputs=zeta_target,
            grad_outputs=a_t,
            retain_graph=False
        )[0]
        
        return vjp.detach()
    
    def compute_exact_vjp_theta(self, a_t, zeta_t, theta_leader):
        """
        Computes the exact VJP a_t · (dF/dtheta), capturing both the direct
        dependence of Q on theta and the indirect dependence through congestion.
        """
        # 1. theta is now the leaf requiring grad; zeta_t stays fixed/detached
        theta_target = theta_leader.detach().clone().requires_grad_(True)
        zeta_fixed = zeta_t.detach()  # stored, frozen — not differentiated

        # 3. Run S: policies + theta_target -> congestion, with matching W_max
        _, _, _, _, W_cong_sim = self.env.simulate_forward_with_policy(
            zeta=zeta_fixed,
            theta_leader=theta_target,
            W_max=100   # must match the true rollout's W_max
        )

        # 4. Run B: congestion + theta_target -> Q-values, per group (own sink, not group 0)
        q_outputs = []
        for k in range(self.env.K):
            q_k = self.solvers[k].compute_q_values_with_waiting_time(
                W_cong_sim, W_max=100, theta_leader=theta_target
            )
            q_outputs.append(q_k)
        q_outputs = torch.stack(q_outputs, dim=0)

        # 5. VJP: a_t · (dQ/dtheta), both channels included automatically
        vjp = torch.autograd.grad(
            outputs=q_outputs,
            inputs=theta_target,
            grad_outputs=a_t,
            retain_graph=False
        )[0]

        return vjp.detach()
        

    def train_step(self, final_flows, W_cong_history, iteration):
        self.optimizer.zero_grad()

        if iteration == 1:
            final_flows = torch.zeros((self.env.H, self.env.N), device=self.env.device)
            base_cong = torch.zeros((self.env.H, self.env.N, self.env.N), device=self.env.device)
            inp = self._prepare_input(final_flows, base_cong)
            theta_leader = self.leader_nets(final_flows, W_cong_history, self.edge_cost, self.env.A)
        else:
            inp = self._prepare_input(final_flows, W_cong_history)
            theta_leader = self.leader_nets(final_flows, W_cong_history, self.edge_cost, self.env.A)

        for solver, z0 in zip(self.solvers, self.initial_zetas):
            solver.zeta = z0.clone()

        T_steps = self.OMDsteps
        
        with torch.no_grad():
            _, edge_occ_new, final_flows_new, policies_new, W_cong_history_new, zeta_history = solve_multigroup(
                self.env, self.solvers, T=T_steps, W_max=100, theta_leader=theta_leader.detach().clone(),
                edge_cost=self.edge_cost
            )

        print(zeta_history.shape)

        exploitability, V_best, V_pi = self.solvers[0].compute_exploitability(W_cong_history_new, W_max=100, theta_leader=theta_leader.detach().clone())
        print(f"Exploitability: {exploitability}")

        social_loss = self.compute_social_loss(final_flows_new, W_cong_history_new)
        congestion_loss = self.congestion_loss(edge_occ=edge_occ_new)

        s_adjoint = self.compute_loss_derivative_wrt_theta_direct(
            zeta_history[T_steps - 1], theta_leader
        )

        a_adjoint = self.compute_loss_derivative_wrt_zeta(
            zeta_history[T_steps - 1], theta_leader
        )   # fixed: direct (K,H,N,N) assignment, no per-group loop, no positional-arg bug

        eta = self.solvers[0].eta
        tau = self.solvers[0].tau

        for t in reversed(range(T_steps)):
            zetas_t = zeta_history[t]

            vjp_zeta  = self.compute_exact_vjp_zeta(a_adjoint, zetas_t, theta_leader.detach())
            vjp_theta = self.compute_exact_vjp_theta(a_adjoint, zetas_t, theta_leader)

            s_adjoint += eta * vjp_theta

            a_adjoint = (1 - eta * tau) * a_adjoint + eta * vjp_zeta

        loss_surrogate = torch.sum(theta_leader * s_adjoint.detach())
        loss_surrogate.backward()

        torch.nn.utils.clip_grad_norm_(self.leader_nets.parameters(), max_norm=1.0)
        self.optimizer.step()

        theta_leader_out = theta_leader.detach()

        return social_loss, final_flows_new, W_cong_history_new, theta_leader_out, congestion_loss