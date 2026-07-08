import torch
import torch.nn as nn
import torch.optim as optim
from leader.leadernet import GraphLeaderIncentiveNet
from solver.solver import solve_multigroup

class GraphEdgeMFG_Trainer:
    def __init__(self, env, solvers, leader_lr=0.01):
        self.env = env
        self.solvers = solvers
        self.H = solvers[0].H
        self.initial_zetas = [s.zeta.clone() for s in solvers]

        # Initialize Leader Network mapping historical edge traffic flows
        self.leader_nets = GraphLeaderIncentiveNet(env.N, env.K, solvers[0].H).to(env.device)
        self.optimizer = optim.Adam(self.leader_nets.parameters(), lr=leader_lr)

        self.edge_cost = torch.zeros((self.env.N, self.env.N), device=self.env.device)
        self.edge_cost[0, 2] = 5.5
        self.edge_cost[1, 3] = 5.5

        
    def compute_social_loss(self, final_flows, W_cong_history):
        """
        Calculates total travel times as social loss.
        
        Under the indicator reward structure, the loss is the integration of all 
        population mass that hasn't arrived at its destination sink yet.
        """
        H, N = final_flows.shape
        total_social_loss = 0.0

        for h in range(H):
            cost = 1 - final_flows[h, self.env.groups[0]["sink"]]
            total_social_loss += cost

        return total_social_loss
    
    def _prepare_input(self, final_flows, W_cong_history):
        """Construct structural feature blocks for graph nodes."""

        adj = self.env.A.detach()  # Adjacency matrix (N, N)

        features_final = torch.cat([final_flows.flatten(), W_cong_history.flatten(), 
                                    self.edge_cost.flatten(), adj.flatten()], dim=0)

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
                self.env, self.solvers, T=50, W_max=15, theta_leader=theta_leader
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
            W_cong_sim, W_max=15, theta_leader=theta_leader
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
            W_max=15   # fix: was silently defaulting to 3
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
            W_max=15
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
                W_cong_sim, W_max=15, theta_leader=theta_leader_val
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
            W_max=15   # must match the true rollout's W_max
        )

        # 4. Run B: congestion + theta_target -> Q-values, per group (own sink, not group 0)
        q_outputs = []
        for k in range(self.env.K):
            q_k = self.solvers[k].compute_q_values_with_waiting_time(
                W_cong_sim, W_max=15, theta_leader=theta_target
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
            theta_leader = self.leader_nets(inp)
        else:
            inp = self._prepare_input(final_flows, W_cong_history)
            theta_leader = self.leader_nets(inp)

        for solver, z0 in zip(self.solvers, self.initial_zetas):
            solver.zeta = z0.clone()

        print(f"Iteration {iteration}: Leader theta_leader =\n{theta_leader.detach()}")

        T_steps = 60

        with torch.no_grad():
            _, _, final_flows_new, policies_new, W_cong_history_new, zeta_history = solve_multigroup(
                self.env, self.solvers, T=T_steps, W_max=15, theta_leader=theta_leader.detach().clone()
            )

        exploitability, V_best, V_pi = self.solvers[0].compute_exploitability(W_cong_history, W_max=15, theta_leader=theta_leader)
        print(f"Exploitability: {exploitability}")

        social_loss = self.compute_social_loss(final_flows_new, W_cong_history_new)

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

        return social_loss, final_flows_new, W_cong_history_new, theta_leader_out