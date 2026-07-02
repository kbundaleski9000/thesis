import torch
import torch.nn as nn

class LeaderIncentiveNet(nn.Module):
    """
    Single shared CNN that produces theta maps for ALL K groups simultaneously.
    Input: (1, 3*K, rows, cols) — [obstacles, sink_k, source_k] for each k stacked
    Output: (K, rows, cols) — one incentive map per group with its spatial mean injected
    """
    def __init__(self, rows, cols, K):
        super().__init__()
        self.K = K
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3 * K, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16,    16, kernel_size=3, padding=1), nn.ReLU()
        )
        
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):   # x: (1, 3*K, rows, cols)
        features = self.feature_extractor(x)
        out = self.final_conv(features) # Shape: (1, K, rows, cols)
        
        # 1. Apply Sigmoid first
        out = self.activation(out)
        
        # 2. Calculate the mean across rows (dim=2) and cols (dim=3)
        # keepdim=True leaves the shape as (1, K, 1, 1)
        spatial_mean = torch.mean(out, dim=(2, 3), keepdim=True)
        
        # 3. Add the spatial mean to every grid cell within its respective group
        out = out - spatial_mean  # Broadcasting will add the mean to each cell in the group
        
        # Return negated output flattened to (K, rows, cols)
        return -out.squeeze(0)


class GraphLeaderIncentiveNet(nn.Module):
    """
    MLP-based Network that generates edge-level theta maps for all K groups.
    Input: (N, 3 * K) — Features describing node structural profiles
    Output: (K, N, N) — One incentive matrix per group
    """
    def __init__(self, num_nodes, K, H):
        super().__init__()
        self.N = num_nodes
        self.K = K
        self.H = H
        
        # 1. Total features for the WHOLE graph = N nodes * (3 * K features per node)
        # For N=4, K=1, this equals 4 * 3 = 12 input features
        global_input_dim = self.K * self.N * self.H + self.H * self.N * self.N  + self.K * self.N + self.N * self.N
        
        # 2. Total output elements needed = K groups * N from_nodes * N to_nodes
        # For N=4, K=1, this equals 1 * 4 * 4 = 16 output elements
        global_output_dim = 1 * self.N * self.N
        
        self.mlp = nn.Sequential(
            nn.Linear(global_input_dim, 32), nn.ReLU(),
            nn.Linear(32, 32),                nn.ReLU(),
            nn.Linear(32, global_output_dim)  # Outputs exactly 16 values globally
        )
        self.activation = nn.Sigmoid()

    def forward(self, node_features):   
        # Expected input shape: (N, 3 * K) -> (4, 3)
        
        # 1. Flatten the node matrix into a single 1D vector of size (12)
        flat_input = node_features.view(-1) 
        
        # 2. Pass through the MLP to get an output vector of size (16)
        out = self.mlp(flat_input)       
        
        # 3. Reshape safely because 16 elements maps perfectly to (1, 4, 4)
        out = out.view(self.N, self.N)   
        
        # 4. Turn into a negative penalty map
        out = self.activation(out)
        return -out