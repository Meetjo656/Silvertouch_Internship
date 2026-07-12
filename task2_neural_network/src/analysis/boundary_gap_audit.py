import cv2
import numpy as np
import json
from pathlib import Path
from shapely.geometry import Point, LineString
import math

from src.data.metadata_mgr import MetadataManager
from src.geometry.artwork_suppressor import ArtworkSuppressor
from src.geometry.edge_detector import EdgeDetector
from src.geometry.line_detector import LineDetector
from src.geometry.line_regularizer import LineRegularizer
from src.geometry.pslg_builder import PSLGBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def run_boundary_gap_audit():
    project_root = Path(".")
    out_dir = project_root / "data" / "interim" / "gap_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    suppressor = ArtworkSuppressor(out_dir)
    edge_detector = EdgeDetector(out_dir)
    line_detector = LineDetector(out_dir)
    regularizer = LineRegularizer(out_dir)
    pslg_builder = PSLGBuilder(out_dir)
    
    audit_results = []
    
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        logger.info(f"[{image_id}] Running Boundary Gap Audit...")
        
        image_path = project_root / img_meta.image_path
        original_img = cv2.imread(str(image_path))
        if original_img is None: continue
        
        # Pipeline Execution up to PSLG
        bilateral, opened = suppressor.process(image_id, original_img)
        edges = edge_detector.detect_edges(image_id, bilateral, opened)
        raw_lines = line_detector.detect_lines(image_id, edges, original_img)
        reg_lines = regularizer.regularize(image_id, raw_lines, original_img)
        pslg_edges = pslg_builder.build_graph(image_id, reg_lines, original_img)
        
        # Analyze Endpoints
        endpoint_degrees = {}
        
        # Count connections at each endpoint
        # Use rounding to handle floating point noise from Shapely intersections
        def pt_key(coord):
            return (round(coord[0], 2), round(coord[1], 2))
            
        for edge in pslg_edges:
            coords = list(edge.coords)
            if len(coords) < 2: continue
            
            p1 = pt_key(coords[0])
            p2 = pt_key(coords[-1])
            
            endpoint_degrees[p1] = endpoint_degrees.get(p1, 0) + 1
            endpoint_degrees[p2] = endpoint_degrees.get(p2, 0) + 1
            
        dangling_endpoints = [pt for pt, degree in endpoint_degrees.items() if degree == 1]
        
        # Calculate Potential Gaps (nearest neighbor for each dangling endpoint)
        potential_gaps = []
        max_gap = 0
        total_gap = 0
        
        if len(dangling_endpoints) > 1:
            for i, p1 in enumerate(dangling_endpoints):
                min_dist = float('inf')
                best_pair = None
                for j, p2 in enumerate(dangling_endpoints):
                    if i == j: continue
                    d = compute_distance(p1, p2)
                    if d < min_dist:
                        min_dist = d
                        best_pair = p2
                if best_pair is not None and min_dist < 200: # Threshold for considered gap
                    potential_gaps.append((p1, best_pair))
                    max_gap = max(max_gap, min_dist)
                    total_gap += min_dist
                    
        mean_gap = total_gap / len(potential_gaps) if potential_gaps else 0
        
        audit_results.append({
            "image_id": image_id,
            "total_lines": len(pslg_edges),
            "endpoint_count": len(endpoint_degrees),
            "dangling_endpoints": len(dangling_endpoints),
            "potential_gaps": len(potential_gaps),
            "mean_gap": mean_gap,
            "max_gap": max_gap
        })
        
        # Visualize Endpoints
        h, w = original_img.shape[:2]
        overlay = original_img.copy()
        
        # Draw edges
        for edge in pslg_edges:
            ext_coords = np.array(edge.coords, dtype=np.int32)
            cv2.polylines(overlay, [ext_coords], isClosed=False, color=(255, 255, 255), thickness=1)
            
        # Draw nodes
        for pt, degree in endpoint_degrees.items():
            cx, cy = int(pt[0]), int(pt[1])
            if degree == 1:
                cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1) # Red for Dangling
            else:
                cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1) # Green for Closed Junction
                
        # Draw potential gaps
        for g1, g2 in potential_gaps:
            pt1 = (int(g1[0]), int(g1[1]))
            pt2 = (int(g2[0]), int(g2[1]))
            cv2.line(overlay, pt1, pt2, (0, 255, 255), 2) # Yellow for Gaps
            
        out_overlay = out_dir / f"{image_id}_boundary_gap_overlay.png"
        cv2.imwrite(str(out_overlay), overlay)
        
    # Generate Report
    report_path = project_root / "data" / "interim" / "boundary_gap_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Boundary Gap Audit Report\n\n")
        f.write("| Image ID | Total Lines | Endpoints | Dangling Endpoints | Potential Gaps | Mean Gap Distance | Max Gap Distance |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for res in audit_results:
            f.write(f"| {res['image_id']} | {res['total_lines']} | {res['endpoint_count']} | {res['dangling_endpoints']} | {res['potential_gaps']} | {res['mean_gap']:.1f}px | {res['max_gap']:.1f}px |\n")
            
    logger.info("Boundary Gap Audit Complete.")

if __name__ == "__main__":
    run_boundary_gap_audit()
