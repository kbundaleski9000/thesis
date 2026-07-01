import torch
import numpy as np

class GraphWorldMFG_MultiGroup:
    """
    Graph environment that hosts K agent groups with routing paths.
    """
    def __init__(self, num_nodes, adjacency_matrix, groups, device="cpu"):
        self.device = device
        self.groups = groups
        self.K      = len(groups)
        self.num_nodes = num_nodes
        
        # A is an adjacency matrix tensor of shape (N, N)
        self.A = torch.tensor(adjacency_matrix, dtype=torch.float32, device=device)
        self.N = self.A.shape[0]

    def get_neighbors(self, u):
        """Returns a list of integer node indices that node u connects to."""
        # Finds all columns 'v' where edge edge weight A[u, v] > 0
        neighbors = torch.where(self.A[u] > 0)[0].tolist()
        return neighbors