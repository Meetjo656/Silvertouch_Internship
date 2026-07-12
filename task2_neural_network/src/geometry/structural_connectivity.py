import math
from pathlib import Path
from src.utils.logger import get_logger
from collections import defaultdict

logger = get_logger(__name__)

class StructuralConnectivity:
    """Task 3: Connectivity Analysis."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def _endpoints_match(self, p1, p2, thresh=5):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < thresh
        
    def analyze_connectivity(self, image_id: str, lines: list) -> dict:
        """
        Builds a graph from lines and computes connected component sizes.
        """
        logger.info(f"[{image_id}] Running Connectivity Analysis...")
        
        # Build adjacency list
        n = len(lines)
        adj = defaultdict(list)
        
        for i in range(n):
            l1 = lines[i]
            pts1 = [(l1["x1"], l1["y1"]), (l1["x2"], l1["y2"])]
            for j in range(i + 1, n):
                l2 = lines[j]
                pts2 = [(l2["x1"], l2["y1"]), (l2["x2"], l2["y2"])]
                
                connected = False
                for p1 in pts1:
                    for p2 in pts2:
                        if self._endpoints_match(p1, p2):
                            connected = True
                            break
                    if connected: break
                        
                if connected:
                    adj[i].append(j)
                    adj[j].append(i)
                    
        # Compute node degrees
        degrees = [len(adj[i]) for i in range(n)]
        
        # Compute Connected Components
        visited = [False] * n
        components = []
        
        for i in range(n):
            if not visited[i]:
                comp = []
                queue = [i]
                visited[i] = True
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                components.append(comp)
                
        # Map line idx to its component size and rank
        components.sort(key=len, reverse=True)
        comp_sizes = [0] * n
        comp_ranks = [0] * n
        
        for rank, comp in enumerate(components):
            size = len(comp)
            for idx in comp:
                comp_sizes[idx] = size
                comp_ranks[idx] = rank
                
        return {
            "degrees": degrees,
            "component_sizes": comp_sizes,
            "component_ranks": comp_ranks
        }
