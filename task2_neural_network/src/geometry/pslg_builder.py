import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union, linemerge
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PSLGBuilder:
    """Stage 2.8: Planar Straight Line Graph Recovery.
    Converts fragmented lines into a mathematically valid noded graph.
    """
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.intersection_stats = []
        self.topology_stats = []
        
    def _extend_line(self, line: LineString, distance: float) -> LineString:
        """Extends a line string at both ends by a given distance to force intersections at corners."""
        coords = list(line.coords)
        if len(coords) < 2: return line
        
        p1, p2 = np.array(coords[0]), np.array(coords[-1])
        vec = p2 - p1
        length = np.linalg.norm(vec)
        
        if length == 0: return line
        
        direction = vec / length
        
        # Extend p1 backwards, p2 forwards
        new_p1 = p1 - direction * distance
        new_p2 = p2 + direction * distance
        
        return LineString([new_p1, new_p2])
        
    def build_graph(self, image_id: str, regularized_lines: list, original_img: np.ndarray) -> list:
        logger.info(f"[{image_id}] Building PSLG Graph...")
        
        h, w = original_img.shape[:2] if original_img is not None else (1000, 1000)
        diagonal = np.sqrt(h**2 + w**2)
        snap_dist = max(10, diagonal * 0.01) # Extend by 1% of diagonal to cross gaps
        
        # 1. Convert dictionary lines to Shapely and Extend
        shapely_lines = []
        for line in regularized_lines:
            p1 = (line["x1"], line["y1"])
            p2 = (line["x2"], line["y2"])
            sl = LineString([p1, p2])
            extended = self._extend_line(sl, snap_dist)
            shapely_lines.append(extended)
            
        # 2. INTERSECTION SWEEP
        # unary_union on LineStrings automatically computes all geometric intersections
        # and splits the lines exactly at the intersection points! This forms a perfect noded graph.
        try:
            noded_graph = unary_union(shapely_lines)
        except Exception as e:
            logger.error(f"Failed to union lines for {image_id}: {e}")
            return []
            
        if noded_graph.is_empty:
            return []
            
        if isinstance(noded_graph, LineString):
            pslg_edges = [noded_graph]
        else:
            pslg_edges = list(noded_graph.geoms)
            
        # 3. Clean up dangling segments (artifacts of line extension)
        # We only keep edges that are part of a closed cycle, OR edges that are long enough.
        # Actually, for the PSLG to feed perfectly into polygonize, dangling edges don't hurt (they are just ignored by polygonize).
        # But we can remove extremely tiny edges to keep the graph clean.
        cleaned_edges = [edge for edge in pslg_edges if edge.length > snap_dist * 0.5]
        
        # Calculate intersection points (just endpoints of noded lines)
        intersections = set()
        for edge in cleaned_edges:
            intersections.add(edge.coords[0])
            intersections.add(edge.coords[-1])
            
        # Record Stats
        self.intersection_stats.append({
            "image_id": image_id,
            "raw_line_count": len(regularized_lines),
            "extended_line_count": len(shapely_lines),
            "noded_segment_count": len(pslg_edges),
            "cleaned_edge_count": len(cleaned_edges),
            "intersection_nodes": len(intersections)
        })
        
        # Draw overlay
        overlay = original_img.copy() if original_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw Edges
        for edge in cleaned_edges:
            ext_coords = np.array(edge.coords, dtype=np.int32)
            cv2.polylines(overlay, [ext_coords], isClosed=False, color=(255, 0, 255), thickness=2)
            
        # Draw Intersections (Nodes)
        for pt in intersections:
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)
            
        out_png = self.out_dir / f"{image_id}_junction_map.png"
        cv2.imwrite(str(out_png), overlay)
        
        return cleaned_edges
        
    def write_reports(self):
        report_dir = self.out_dir.parent
        
        # Intersection Report
        inter_path = report_dir / "intersection_report.md"
        with open(inter_path, "w", encoding="utf-8") as f:
            f.write("# Intersection Sweep Report\n\n")
            f.write("| Image ID | Raw Lines | Extended Lines | Noded Segments | Final Graph Edges | Intersection Nodes |\n")
            f.write("|---|---|---|---|---|---|\n")
            for stat in self.intersection_stats:
                f.write(f"| {stat['image_id']} | {stat['raw_line_count']} | {stat['extended_line_count']} | {stat['noded_segment_count']} | {stat['cleaned_edge_count']} | {stat['intersection_nodes']} |\n")
                
        # Topology Report
        topo_path = report_dir / "topology_report.md"
        with open(topo_path, "w", encoding="utf-8") as f:
            f.write("# Topology Audit Report\n\n")
            f.write("| Image ID | Graph Nodes | Graph Edges |\n")
            f.write("|---|---|---|\n")
            for stat in self.intersection_stats:
                f.write(f"| {stat['image_id']} | {stat['intersection_nodes']} | {stat['cleaned_edge_count']} |\n")
