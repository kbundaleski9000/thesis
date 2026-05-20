import torch
import torch.nn as nn


class LeaderIncentiveNet(nn.Module):
    """
    Single shared CNN that produces theta maps for ALL K groups simultaneously.
    Input: (1, 3*K, rows, cols) — [obstacles, sink_k, source_k] for each k stacked
    Output: (K, rows, cols) — one incentive map per group
    """
    def __init__(self, rows, cols, K):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Conv2d(3 * K, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16,    16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16,     1, kernel_size=1), 
            nn.Tanh()
        )

    def forward(self, x):   # x: (1, 3*K, rows, cols)
        return self.net(x).squeeze(0)  # (K, rows, cols)
