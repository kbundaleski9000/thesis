import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def solve_multigroup(env, solvers, T=200, W_max=10, theta_leader=None):
    """
    Main MFG execution loop using explicit congestion waiting-times.
    
    Returns:
        flows: Granular macro-distributions per group of shape (K, H, N, W_max + 1)
        final_flows: Total combined population density over spatial nodes of shape (H, N)
        policies: Converged agent policy tensors per group
    """
    K = len(solvers)
    H = solvers[0].H
    N = env.N
    edge_cost = torch.zeros((N, N), device=env.device)
    edge_cost[0,2] = 5.5
    edge_cost[1,3] = 5.5

    zeta_history = torch.zeros((T, K, H, N, N), device=env.device)
    
    # Initialize policies from the initial solver states
    policies = [solver.get_policy(solver.zeta) for solver in solvers]

    for t in range(T):
        # --- 1. THE FORWARD PASS ---
        # State tracking: (Group, Time, Location Node, Remaining Waiting Time)
        flows = torch.zeros((K, H, N, W_max + 1), device=env.device)
        
        # Seed initial source distribution at h=0, w=0 (completely free)
        for k in range(K):
            source = env.groups[k]["source"]
            flows[k, 0, source, 0] = env.groups[k]["mass"]
            
        # Tracking variables for edge usage maps and dynamic traffic delays
        E_total_edges = torch.zeros((H, N, N), device=env.device)
        W_cong_history = torch.zeros((H, N, N), device=env.device)

        # Chronological progression step-by-step
        for h in range(H - 1):
            # Accumulate traffic leaving nodes from agents whose wait time has expired (w=0)
            for k in range(K):
                active_mass = flows[k, h, :, 0]
                for u in range(N):
                    for v in env.get_neighbors(u):
                        edge_traffic = active_mass[u] * policies[k][h, u, v]
                        E_total_edges[h, u, v] += edge_traffic


            E_total_edges_final = torch.zeros((H, N, N), device=env.device)
            E_total_edges_final[h, 0, 1] = E_total_edges[h, 0, 1]
            E_total_edges_final[h, 2, 3] = E_total_edges[h, 2, 3]

             
            # Calculate link delays (Using pure continuous tensor arithmetic)
            W_cong_history[h] = torch.clamp(
                E_total_edges_final[h] * 5.0 + edge_cost + theta_leader, 
                min=1.0, 
                max=float(W_max)
            )


            # Distribute tracking flows into the next timestep h+1
            for k in range(K):
                sink = env.groups[k]["sink"]
                
                # Rule A: Trapped / Locked agents (w > 0) step down their timers, staying at node
                for w in range(1, W_max + 1):
                    flows[k, h + 1, :, w - 1] = flows[k, h + 1, :, w - 1] + flows[k, h, :, w]
                    
                # Rule B: Free agents (w == 0) make strategic transitions
                active_mass = flows[k, h, :, 0]
                for u in range(N):
                    if u == sink:
                        # Agents at their goal sink absorb there at w=0 permanently
                        flows[k, h + 1, sink, 0] = flows[k, h + 1, sink, 0] + active_mass[sink]
                        continue
                        
                    for v in env.get_neighbors(u):
                        moving_mass = active_mass[u] * policies[k][h, u, v]
                        
                        # --- FIXED: DIFFERENTIABLE SOFT INDEXING ---
                        # Fetch the continuous tensor value directly (NO .item() or int())
                        delay_continuous = W_cong_history[h, u, v]
                        
                        # Find the surrounding integer indices using PyTorch tensor math
                        delay_floor = torch.floor(delay_continuous).long()
                        delay_ceil = torch.ceil(delay_continuous).long()
                        
                        # Clamp indices to ensure they stay within bounds [0, W_max]
                        delay_floor = torch.clamp(delay_floor, min=0, max=W_max)
                        delay_ceil = torch.clamp(delay_ceil, min=0, max=W_max)
                        
                        # Calculate how close the delay is to each neighbor
                        weight_ceil = delay_continuous - delay_floor.float()
                        weight_floor = 1.0 - weight_ceil
                        
                        # Distribute the moving mass proportionally across both slots
                        # This keeps the math smooth and fully differentiable!
                        flows[k, h + 1, v, delay_floor] = flows[k, h + 1, v, delay_floor] + (moving_mass * weight_floor)
                        flows[k, h + 1, v, delay_ceil] = flows[k, h + 1, v, delay_ceil] + (moving_mass * weight_ceil)

        # --- 2. THE BACKWARD PASS (OMD POLICY UPDATES) ---
        new_policies = []
        for k, solver in enumerate(solvers):
            # Use backward induction value maps factoring ahead for link delays

            q = solver.compute_q_values_with_waiting_time(W_cong_history, W_max, theta_leader)
            
            # Update regularized Online Mirror Descent logits
            solver.zeta = (1 - solver.eta * solver.tau) * solver.zeta + solver.eta * q
            new_policies.append(solver.get_policy(solver.zeta))
            zeta_history[t, k] = solver.zeta.clone()  # Store for analysis
            
        policies = new_policies

    # Sum out groups and waiting-time tiers to provide a aggregated spatial footprint tensor
    final_flows = flows.sum(dim=0).sum(dim=-1)

    return flows, final_flows, policies, W_cong_history, zeta_history


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
                
                # 2. Assign the 2D slice of logits [H, len(neighbors)] to the valid columns
                # Inside get_policy:
                mask[:, neighbors] = zeta[:, u, neighbors] / 3.0  # Dividing by a temperature softens the step
                policy[:, u, :] = F.softmax(mask, dim=-1)
                
                # 3. Softmax across the destination node dimension (dim=-1) 
                # This ensures invalid nodes get 0 probability and valid ones sum to 1.0 for every timestep h
        return policy

    def compute_q_values_with_waiting_time(self, W_cong_history, W_max=10, theta_leader=None):
        import torch
        
        Q_list = []
        # 1. Start with the terminal boundary conditions at horizon H (all zeros)
        # Shape: (N, W_max + 1)
        V_list = [torch.zeros((self.N, W_max + 1), device=self.env.device)]

        # 2. Backward induction loop over time steps
        for h in reversed(range(self.H)):
            V_next = V_list[-1]  # This is the value matrix from step h + 1
            
            # Build the tensors for the current timestep 'h' completely out-of-place
            V_current_nodes = []
            Q_current_nodes = []
            
            for u in range(self.N):
                sink = self.env.groups[0]["sink"]
                running_cost = 0.0 if u == sink else -1.0
                
                # --- Rule A: Handle locked waiting states (w > 0) out-of-place ---
                # Gather all waiting values into a list instead of modifying a tensor slice
                v_waiting = [running_cost + V_next[u, w - 1] for w in range(1, W_max + 1)]
                
                # --- Rule B: Handle choice states (w == 0) ---
                q_actions = torch.zeros(self.N, device=self.env.device)
                if u == sink:
                    v_choice = torch.tensor([0.0], device=self.env.device)
                else:
                    for v in self.env.get_neighbors(u):
                        delay_continuous = W_cong_history[h, u, v]
                        delay_floor = torch.clamp(torch.floor(delay_continuous).long(), 0, W_max)
                        delay_ceil = torch.clamp(torch.ceil(delay_continuous).long(), 0, W_max)
                        
                        weight_ceil = delay_continuous - delay_floor.float()
                        weight_floor = 1.0 - weight_ceil

                        # Differentiable linear interpolation lookup
                        future_val = (weight_floor * V_next[v, delay_floor] + 
                                    weight_ceil * V_next[v, delay_ceil])
                        
                        q_actions[v] = running_cost - theta_leader[u, v] + future_val
                    
                    v_choice = torch.max(q_actions).unsqueeze(0)
                
                # Combine choice (w=0) and waiting tiers (w > 0) out-of-place for node 'u'
                v_node = torch.cat([v_choice, torch.tensor(v_waiting, device=self.env.device)], dim=0)
                
                V_current_nodes.append(v_node)
                Q_current_nodes.append(q_actions)
                
            # Stack the nodes together to form the full spatial frame for time 'h'
            V_list.append(torch.stack(V_current_nodes, dim=0))
            Q_list.append(torch.stack(Q_current_nodes, dim=0))
            
        # 3. Assemble full sequence tensors cleanly out-of-place
        Q_list.reverse()
        Q = torch.stack(Q_list, dim=0)
        
        return Q