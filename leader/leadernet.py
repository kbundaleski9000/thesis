import torch
import torch.nn as nn

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
        global_input_dim =   3 * self.N * self.N         +          self.H * self.N       + self.H * self.N * self.N         
        
        # 2. Total output elements needed = K groups * N from_nodes * N to_nodes
        # For N=4, K=1, this equals 1 * 4 * 4 = 16 output elements
        global_output_dim = 1 * self.N * self.N
        
        self.mlp = nn.Sequential(
            nn.Linear(global_input_dim, 2048), nn.ReLU(),
            nn.Linear(2048, 1024),                nn.ReLU(),
            nn.Linear(1024, global_output_dim) # Outputs exactly 16 values globally
        )

        final = self.mlp[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.constant_(final.bias, -4.0)
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
        out = self.activation(out) * 5.0
        return out


class GraphLeaderIncentiveNetCNN(nn.Module):
    """
    CNN-based version of GraphLeaderIncentiveNet.
 
    Rather than flattening everything into one giant vector fed to a massive
    Linear layer, this treats the (H, N, N) congestion history as H channels
    over an N x N grid, and the (N, N) edge_cost / adj matrices as 2 more
    channels -- letting 2D convolutions process them with shared weights
    instead of one independent weight per (feature, hidden-unit) pair.
 
    final_flows (H, N) doesn't fit the (N,N) spatial grid naturally, so it's
    kept as a separate flat vector and concatenated in after the conv trunk.
 
    This does NOT respect true graph adjacency (see caveat in the reply) --
    it's a parameter-efficient improvement over the flatten-everything MLP,
    not a full graph-neural-network solution.
    """
    def __init__(self, num_nodes, K, H):
        super().__init__()
        self.N = num_nodes
        self.K = K
        self.H = H
 
        # Channels: H timesteps of congestion + edge_cost + adj = H + 2 channels,
        # each an (N, N) "image".
        in_channels = H + 2
 
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        # After the conv trunk: (32, N, N) -> flatten -> combine with final_flows (H*N)
        conv_out_dim = 32 * self.N * self.N
        flows_dim = self.H * self.N
 
        self.head = nn.Sequential(
            nn.Linear(conv_out_dim + flows_dim, 1024), nn.LayerNorm(1024), nn.ReLU(),
        )
        self.out_layer = nn.Linear(1024, self.N * self.N)
        nn.init.normal_(self.out_layer.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.out_layer.bias, -4.0)
        self.activation = nn.Sigmoid()
 
    def forward(self, final_flows, W_cong_history, edge_cost, adj):
        """
        Args (unflattened, unlike the MLP version's single concatenated vector):
            final_flows:    (H, N)
            W_cong_history: (H, N, N)
            edge_cost:      (N, N)
            adj:            (N, N)
        """
        # Stack into (H+2, N, N) channel-first image, add batch dim for Conv2d.
        img = torch.cat([
            W_cong_history,               # (H, N, N)
            edge_cost.unsqueeze(0),       # (1, N, N)
            adj.unsqueeze(0),             # (1, N, N)
        ], dim=0).unsqueeze(0)            # (1, H+2, N, N)
 
        conv_out = self.conv(img).flatten()          # (32*N*N,)
        combined = torch.cat([conv_out, final_flows.flatten()], dim=0)
        features = self.head(combined)
        out = self.out_layer(features).view(self.N, self.N)
        return self.activation(out) * 5.0