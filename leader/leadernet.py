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
        
        self.final_conv = nn.Conv2d(16, K, kernel_size=1)
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
        out = out - spatial_mean
        
        # Return negated output flattened to (K, rows, cols)
        return -out.squeeze(0)