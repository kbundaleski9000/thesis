import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def solve_multigroup(env, solvers, theta_list, T=20, W_max=5):
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
    edge_cost[0,2] = -5.5
    edge_cost[1,3] = -5.5
    
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

             
            # Map physical edge traffic volumes directly into integer delays
            # Scale multiplier determines traffic sensitivity (Clamped between 0 and W_max)
            W_cong_history[h] = torch.clamp(torch.floor(E_total_edges_final[h] * 5.0 + edge_cost), 0, W_max)

            # Distribute tracking flows into the next timestep h+1
            for k in range(K):
                sink = env.groups[k]["sink"]
                
                # Rule A: Trapped / Locked agents (w > 0) step down their timers, staying at node
                for w in range(1, W_max + 1):
                    flows[k, h + 1, :, w - 1] += flows[k, h, :, w]
                    
                # Rule B: Free agents (w == 0) make strategic transitions
                active_mass = flows[k, h, :, 0]
                for u in range(N):
                    if u == sink:
                        # Agents at their goal sink absorb there at w=0 permanently
                        flows[k, h + 1, sink, 0] += active_mass[sink]
                        continue
                        
                    for v in env.get_neighbors(u):
                        moving_mass = active_mass[u] * policies[k][h, u, v]
                        delay = int(W_cong_history[h, u, v].item())
                        
                        # Injected into the target node at the designated delay level
                        flows[k, h + 1, v, delay] += moving_mass

        # --- 2. THE BACKWARD PASS (OMD POLICY UPDATES) ---
        new_policies = []
        for k, solver in enumerate(solvers):
            # Use backward induction value maps factoring ahead for link delays
            q = solver.compute_q_values_with_waiting_time(W_cong_history, W_max, theta_list[k])
            
            # Update regularized Online Mirror Descent logits
            solver.zeta = (1 - solver.eta * solver.tau) * solver.zeta + solver.eta * q
            new_policies.append(solver.get_policy(solver.zeta))
            
        policies = new_policies

    # Sum out groups and waiting-time tiers to provide a aggregated spatial footprint tensor
    final_flows = flows.sum(dim=0).sum(dim=-1)

    return flows, final_flows, policies


class GraphMFG_OMD_EdgeSolver_MultiGroup:
    """
    OMD Solver customized for Graph networks using dynamic delay-waiting times.
    """
    def __init__(self, env, group_idx, eta=0.1, tau=0.01, alpha=1.0, H=30):
        self.env         = env
        self.k           = group_idx
        self.group       = env.groups[group_idx]
        self.eta         = eta
        self.tau         = tau
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
                mask[:, neighbors] = zeta[:, u, neighbors]
                
                # 3. Softmax across the destination node dimension (dim=-1) 
                # This ensures invalid nodes get 0 probability and valid ones sum to 1.0 for every timestep h
                policy[:, u, :] = F.softmax(mask, dim=-1)
        return policy

    def compute_q_values_with_waiting_time(self, W_cong_history, W_max, theta_leader_k):
        """
        Performs backwards value-iteration induction factoring dynamic delays.
        
        A non-sink node incurs a penalty of -1.0 for every step spent traveling.
        Leader incentives are added directly as bonuses on specific edge choices.
        """
        V = torch.zeros((self.H + 1, self.N, W_max + 1), device=self.device)
        Q = torch.zeros((self.H, self.N, self.N), device=self.device)
        sink = self.group["sink"]
        
        for h in reversed(range(self.H)):
            for u in range(self.N):
                # Standard binary running penalty indicator structure
                running_cost = 0.0 if u == sink else -1.0
                
                # 1. Evaluate value transitions for locked waiting states (w > 0)
                for w in range(1, W_max + 1):
                    V[h, u, w] = running_cost + V[h + 1, u, w - 1]
                    
                # 2. Evaluate free states (w == 0) where routing policies can actively choose links
                if u == sink:
                    V[h, sink, 0] = 0.0 # Absorbing sink costs nothing
                else:
                    for v in self.env.get_neighbors(u):
                        delay = int(W_cong_history[h, u, v].item())
                        delay = min(delay, W_max)
                        
                        # Temporal skip: landing index capped cleanly at max horizon bounds
                        next_h = min(h + 1, self.H)
                        
                        # Q value integrates leader incentives as an edge reward adjustment
                        leader_bonus = theta_leader_k[u, v] if theta_leader_k is not None else 0.0
                        Q[h, u, v] = running_cost + leader_bonus + V[next_h, v, delay]
                    
                    # For Best-Response OMD updates, V maps the highest possible edge choice
                    # Non-neighboring nodes will remain -inf or filtered via masking 
                    V[h, u, 0] = torch.max(Q[h, u, :])
                    
        return Q