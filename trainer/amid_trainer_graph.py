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

        # Initialize Leader Network mapping historical edge traffic flows
        self.leader_nets = GraphLeaderIncentiveNet(env.N, env.K, solvers[0].H).to(env.device)
        self.optimizer = optim.Adam(self.leader_nets.parameters(), lr=leader_lr)
        
        # Base static background biases per edge choice (K, N, N)
        self.base_thetas = - torch.ones((self.env.K, self.env.N), dtype=torch.float32, device=env.device)

        for i in range(self.env.K):
            self.base_thetas[i, self.env.groups[i]["sink"]] = 0.0  # No penalty for target

    def compute_social_loss(self, flows, W_cong_history):
        """
        Calculates total travel times as social loss.
        
        Under the indicator reward structure, the loss is the integration of all 
        population mass that hasn't arrived at its destination sink yet.
        """
        K, H, N, _ = flows.shape
        total_social_loss = 0.0

        for h in range(H):
            cost = 1 - flows[0, h, self.env.groups[0]["sink"], 0]
            total_social_loss += cost

        return total_social_loss
    
    def _prepare_input(self, spatial_flows, W_cong_history):
        """Construct structural feature blocks for graph nodes."""

        adj = self.env.A.detach()  # Adjacency matrix (N, N)

        features_final = torch.cat([spatial_flows.flatten(), W_cong_history.flatten(), 
                                    self.base_thetas.flatten(), adj.flatten()], dim=0)

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
    
    def compute_loss_derivative_wrt_zeta(self, final_zeta, k_group, theta_leader):
        """
        Computes the exact terminal adjoint vector a_T = \partial G / \partial \zeta_T
        """
        # 1. Attach the gradient tracker to a clone of the actual logit tensor shape (H, N, N)
        zeta_target = final_zeta[k_group].detach().clone().requires_grad_(True)
        policy_target = torch.zeros(self.env.K, self.env.H, self.env.N, self.env.N, device=self.env.device)
        
        # 2. Map zeta to policies via the correct group's Softmax operation
        for k in range(self.env.K):
            if k == k_group:
                policy_target[k] = self.solvers[k].get_policy(zeta_target)
            else:
                # For other groups, we can use their final zeta without gradient tracking
                policy_target[k] = self.solvers[k].get_policy(final_zeta[k].detach())
        
        # 3. Simulate forward passing the localized target policy and current leader rules
        flows_sim, final_flows_sim, _, _ = self.env.simulate_forward_with_policy(
            policy_sim=policy_target, 
            theta_leader=theta_leader
        )
        
        # 4. Compute your Social Loss objective G from the simulation outputs
        # Example: Social loss minimizes total traveling mass over non-sink nodes
        sink = self.env.groups[k_group]["sink"]
        loss_G = torch.tensor(0.0, device=self.env.device)
        for h in range(self.env.H):
            cost = 1 - flows_sim[k_group, h, sink, 0]
            loss_G = loss_G + cost
        # 5. Extract the derivative of the loss with respect to the input logits
        loss_G.backward(retain_graph=True)
        a_T = zeta_target.grad
        
        return a_T
    
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
        

    def train_step(self, flows_previous, W_cong_history, iteration):
        self.optimizer.zero_grad()

        if iteration == 1:
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

        # 2. Forward Pass of the Game: Run OMD for T steps
        T_steps = 40
        flows_new, _, _, W_cong_history_new, zeta_history = solve_multigroup(
            self.env, self.solvers, T=T_steps, W_max=15, theta_leader=theta_leader
        )

        # 3. Calculate Terminal Adjoint States (Line 4 of Alg 1)
        # s_T+1 = dG/dtheta, a_T = -dG/dzeta
        # For a total travel time minimization objective:
        social_loss = self.compute_social_loss(flows_new, W_cong_history_new)
        
        # Initialize your adjoint vectors with the same shape as zeta
        # Assuming K groups, and each group has a zeta matrix
        a_adjoint = [torch.zeros_like(s.zeta) for s in self.solvers]
        s_adjoint = torch.zeros(self.H, self.env.N, self.env.N, device=self.env.device) # This accumulates our grad_theta

        # Calculate initial terminal values based on your G objective
        # (Using manual derivatives of social loss with respect to final converged states)
        for k in range(self.env.K):
            a_adjoint[k] = self.compute_loss_derivative_wrt_zeta(zeta_history[T_steps-1], k, theta_leader)
            print(f"Adjoint vector a_T for group {k} has shape: {a_adjoint[k].shape}")

        # =========================================================================
        # 4. ALGORITHM 1 BACKWARD PASS: Loops exactly T times in reverse
        # =========================================================================
        eta = self.solvers[0].eta
        tau = self.solvers[0].tau

        for t in reversed(range(T_steps)):
            # Fetch the zetas recorded at step t of the forward loop
            zetas_t = zeta_history[t]
            
            for k in range(self.env.K):
                # Line 6: at-1 = (1 - ητ)at + η * at * ∂ζF
                # F(θ, ζ) = (1-ητ)ζ + η*Q. The derivative ∂ζF involves how Q changes with ζ
                # Since Q depends on congestion from flows, and flows depend on policies (softmax of ζ)
                # You compute the Jacobian matrix multiplication or its vector-Jacobian product approximation:
                vjp_zeta = self.compute_vector_jacobian_product_zeta(a_adjoint[k], zetas_t, k, theta_leader)
                a_adjoint[k] = (1 - eta * tau) * a_adjoint[k] + eta * vjp_zeta
                
                # Line 7: st-1 = st - η * at * ∂θF
                # Since Q = running_cost + theta_leader + V, the partial derivative ∂θQ is identity (1.0)
                # Thus, ∂θF with respect to theta_leader simplifies cleanly to a direct mapping of the adjoint weight:
                s_adjoint -= eta * a_adjoint[k]
        
        s_adjoint_final = s_adjoint.sum(dim=0)
        # 5. Execute backpropagation through the leader network using the final s_0 gradient
        theta_leader.backward(s_adjoint_final)
        self.optimizer.step()

        theta_leader_out = theta_leader.detach()

        print(f"theta _ leader out is {theta_leader_out}")

        return social_loss, flows_new, W_cong_history_new, theta_leader_out