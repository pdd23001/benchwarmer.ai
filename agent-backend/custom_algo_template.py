from benchwarmer.algorithms.base import AlgorithmWrapper
import random

class MyCustomAlgorithm(AlgorithmWrapper):
    # Give your algorithm a unique name for the leaderboard
    name = "my_custom_algo"

    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        """
        Implement your algorithm logic here.
        
        Parameters:
        - instance: {"nodes": [0, 1, ...], "edges": [{"source": 0, "target": 1, ...}, ...]}
        - timeout: time limit in seconds
        
        Returns:
        - {"solution": ..., "metadata": ...}
        """
        
        nodes = instance["nodes"]
        edges = instance["edges"]
        
        # Build adjacency list for fast lookup
        # weight defaults to 1.0 if not present
        adj = {n: [] for n in nodes}
        for e in edges:
            u, v = e["source"], e["target"]
            w = e.get("weight", 1.0)
            adj[u].append((v, w))
            adj[v].append((u, w))
            
        # --- Greedy Constructive Algorithm ---
        # 1. Shuffle nodes to add randomness (avoid bias from node IDs)
        # 2. Iterate nodes and place in set 0 or 1 to maximize cut edges to existing neighbors
        
        sorted_nodes = list(nodes)
        random.shuffle(sorted_nodes)
        
        partition = {}  # node_id -> 0 or 1
        
        for u in sorted_nodes:
            # Calculate cut weight if we place u in set 0 vs set 1
            # considering only neighbors already placed
            weight_if_0 = 0.0
            weight_if_1 = 0.0
            
            for v, w in adj[u]:
                if v in partition:
                    if partition[v] == 1:
                        weight_if_0 += w  # u=0, v=1 -> edge cut
                    else:
                        weight_if_1 += w  # u=1, v=0 -> edge cut
            
            # Greedily choose the assignment that adds more weight to the cut
            if weight_if_0 >= weight_if_1:
                partition[u] = 0
            else:
                partition[u] = 1
        
        # Convert dict back to list in node order [0, 1, 2, ...]
        final_partition = [partition[n] for n in nodes]
        
        return {
            "solution": {
                "partition": final_partition
            },
            "metadata": {
                "algorithm": "constructive_greedy",
                "heuristic": "maximize immediate cut gain"
            }
        }
