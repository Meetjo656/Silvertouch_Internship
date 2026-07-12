import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Polygon
import networkx as nx

from src.data.metadata_mgr import MaskMetadata
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RegionNode:
    """Represents a single structural region node in the blueprint graph."""
    def __init__(self, mask_id: str, polygon: Polygon, area: float, structural_class: str):
        self.mask_id = mask_id
        self.polygon = polygon
        self.area = area
        self.structural_class = structural_class
        
        # Will be populated during graph resolution
        self.parent_region: Optional[str] = None
        
    def __repr__(self):
        return f"<RegionNode {self.mask_id} ({self.structural_class}) Area: {self.area:.1f}>"

class RegionGraphBuilder:
    """Builds a planar partition graph from a set of overlapping SAM masks."""
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes: Dict[str, RegionNode] = {}
        
    def add_mask(self, mask_id: str, mask_img: np.ndarray, structural_class: str):
        """Converts a binary mask image into a Shapely polygon and adds it as a node."""
        # Find contours
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
            
        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Need at least 3 points for a valid polygon
        if len(largest_contour) < 3:
            return
            
        # Convert OpenCV contour shape (N, 1, 2) to list of tuples (x, y)
        points = [tuple(point[0]) for point in largest_contour]
        
        try:
            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0) # Fix invalid self-intersecting polygons
                
            area = poly.area
            node = RegionNode(mask_id, poly, area, structural_class)
            self.nodes[mask_id] = node
            self.graph.add_node(mask_id, data=node)
            
        except Exception as e:
            logger.error(f"Failed to create polygon for {mask_id}: {e}")
            
    def compute_overlaps(self) -> List[Dict]:
        """Calculates exact overlaps using Shapely geometry and adds overlap edges."""
        overlap_edges = []
        mask_ids = list(self.nodes.keys())
        n = len(mask_ids)
        
        for i in range(n):
            for j in range(i + 1, n):
                id1 = mask_ids[i]
                id2 = mask_ids[j]
                poly1 = self.nodes[id1].polygon
                poly2 = self.nodes[id2].polygon
                
                if poly1.intersects(poly2):
                    intersection = poly1.intersection(poly2)
                    inter_area = intersection.area
                    if inter_area > 0:
                        self.graph.add_edge(id1, id2, type='overlap', weight=inter_area)
                        overlap_edges.append({
                            "source": id1,
                            "target": id2,
                            "intersection_area": inter_area,
                            "pct_1": inter_area / poly1.area,
                            "pct_2": inter_area / poly2.area
                        })
                        
        return overlap_edges
        
    def compute_adjacencies(self, distance_threshold: float = 10.0) -> List[Dict]:
        """Finds adjacent nodes that share a boundary (within threshold) but do not overlap heavily."""
        adjacency_edges = []
        mask_ids = list(self.nodes.keys())
        n = len(mask_ids)
        
        for i in range(n):
            for j in range(i + 1, n):
                id1 = mask_ids[i]
                id2 = mask_ids[j]
                
                # Skip if they already have a heavy overlap edge
                if self.graph.has_edge(id1, id2) and self.graph[id1][id2].get('type') == 'overlap':
                    continue
                    
                poly1 = self.nodes[id1].polygon
                poly2 = self.nodes[id2].polygon
                
                distance = poly1.distance(poly2)
                if distance <= distance_threshold:
                    self.graph.add_edge(id1, id2, type='adjacent', weight=distance)
                    adjacency_edges.append({
                        "source": id1,
                        "target": id2,
                        "distance": distance
                    })
                    
        return adjacency_edges
        
    def export_graph_data(self) -> Dict:
        """Exports the current state of the graph for reconstruction analysis."""
        return {
            "nodes": [
                {
                    "id": n_id,
                    "structural_class": n.structural_class,
                    "area": n.area
                } for n_id, n in self.nodes.items()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "type": data['type'],
                    "weight": data['weight']
                } for u, v, data in self.graph.edges(data=True)
            ]
        }
