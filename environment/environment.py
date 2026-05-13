import torch
import numpy as np
import heapq


class GridWorldMFG_MultiGroup:
    """
    Grid world that hosts K agent groups.

    groups: list of dicts, each with keys
        "source" : (row, col)
        "sink"   : (row, col)
        "mass"   : float  (total probability mass, default 1.0)
    """
    def __init__(self, rows, cols, groups, obstacles=None, device="cpu"):
        self.rows    = rows
        self.cols    = cols
        self.device  = device
        self.groups  = groups          # list of group dicts
        self.K       = len(groups)

        # Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
        self.dr = torch.tensor([-1, 1, 0, 0, 0], device=device)
        self.dc = torch.tensor([ 0, 0,-1, 1, 0], device=device)

        self.obstacles = torch.zeros((rows, cols), dtype=torch.bool, device=device)
        if obstacles:
            for r, c in obstacles:
                self.obstacles[r, c] = True

        self.dist_maps = self.compute_multi_group_dist_maps(base_cost=1.0)

    # Convenience properties kept for backward-compat with helpers
    @property
    def source(self):
        return self.groups[0]["source"]

    @property
    def sink(self):
        return self.groups[0]["sink"]
    


    def compute_multi_group_dist_maps(self, base_cost=1.0, congestion_weight=1.0, target_sink_cost= -10.0):
        """
        Computes a distance map for each group in the MFG.
        Each group travels toward its own 'sink' while avoiding congestion.
        
        Args:
            mean_field: (rows, cols) tensor or array representing agent density.
            base_cost: Flat cost to move to an adjacent cell.
            congestion_weight: Scaling factor for the mean_field penalty.
            target_sink_reward: Reward to reach the target sink.
        
        Returns:
            K_dist_maps: (K, rows, cols) numpy array of distances.
        """
        K = self.K
        rows, cols = self.rows, self.cols
        # Initialize distance maps for all K groups
        all_dist_maps = np.full((K, rows, cols), 100.0) # Using 100 as your 'max' default

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for k in range(K):
            sink = self.groups[k]["sink"]
            target_r, target_c = sink
            
            dist_map = np.full((rows, cols), np.inf)
            visited = np.zeros((rows, cols), dtype=bool)
            heap = [(0.0, target_r, target_c)]
            dist_map[target_r, target_c] = 0.0

            while heap:
                dist, r, c = heapq.heappop(heap)

                if visited[r, c]:
                    continue
                visited[r, c] = True

                for dr, dc in directions:
                    nr, nc = r + dr, c + dc

                    # 1. Bounds Check
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    
                    # 2. Obstacle Check (using the class tensor)
                    if visited[nr, nc] or self.obstacles[nr, nc]:
                        continue

                    # 3. Cost Calculation: base + penalty based on density at destination
                    # Note: We calculate distance BACKWARDS from sink to source
                    edge_weight = base_cost 
                    new_dist = dist + edge_weight

                    if new_dist < dist_map[nr, nc]:
                        dist_map[nr, nc] = new_dist
                        heapq.heappush(heap, (new_dist, nr, nc))

            # Replace unreachable areas and store
            dist_map[target_r, target_c] = target_sink_cost  # Ensure sink is zero
            dist_map[dist_map == np.inf] = 100.0
            all_dist_maps[k] = dist_map

        return all_dist_maps