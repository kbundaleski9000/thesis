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
    MLP-based Network that generates theta maps for all K groups.
    Input: (N, Feature_Dim) — Features describing node roles
    Output: (K, N) — One incentive signal per graph node per group
    """
    def __init__(self, num_nodes, K):
        super().__init__()
        self.N = num_nodes
        self.K = K
        
        # Process node structures directly using Dense Linear layers
        self.mlp = nn.Sequential(
            nn.Linear(3 * self.K, 32), nn.ReLU(),
            nn.Linear(32, 32),          nn.ReLU(),
            nn.Linear(32, 1) # Output a logit for each group per node
        )
        self.activation = nn.Sigmoid()

    def forward(self, node_features):   
        # node_features shape: (N, feature_dim)
        out = self.mlp(node_features)   # Shape: (N, K)
        out = out.t()                   # Transpose to group format: (K, N)
        out = self.activation(out)
        
        
        return -out