import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union

from src.data.metadata_mgr import MetadataManager
from src.geometry.artwork_suppressor import ArtworkSuppressor
from src.geometry.edge_detector import EdgeDetector
from src.geometry.line_detector import LineDetector
from src.geometry.line_regularizer import LineRegularizer
from src.geometry.pslg_builder import PSLGBuilder
from src.geometry.polygonizer import Polygonizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_outer_boundary_audit():
    project_root = Path(".")
    out_dir = project_root / "data" / "interim" / "outer_boundary_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    suppressor = ArtworkSuppressor(out_dir)
    edge_detector = EdgeDetector(out_dir)
    line_detector = LineDetector(out_dir)
    regularizer = LineRegularizer(out_dir)
    pslg_builder = PSLGBuilder(out_dir)
    polygonizer = Polygonizer(out_dir)
    
    boundary_results = []
    coverage_results = []
    
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        logger.info(f"[{image_id}] Running Outer Boundary Audit...")
        
        image_path = project_root / img_meta.image_path
        original_img = cv2.imread(str(image_path))
        if original_img is None: continue
        
        h, w = original_img.shape[:2]
        carton_area = h * w
        
        # Pipeline Execution up to Polygonizer
        bilateral, opened = suppressor.process(image_id, original_img)
        edges = edge_detector.detect_edges(image_id, bilateral, opened)
        raw_lines = line_detector.detect_lines(image_id, edges, original_img)
        reg_lines = regularizer.regularize(image_id, raw_lines, original_img)
        pslg_edges = pslg_builder.build_graph(image_id, reg_lines, original_img)
        faces = polygonizer.polygonize_lines(image_id, pslg_edges, original_img)
        
        # Filter structural faces
        min_face_area = carton_area * 0.005
        structural_faces = [f for f in faces if f.area >= min_face_area]
        
        recovered_union = None
        recovered_area = 0.0
        if structural_faces:
            recovered_union = unary_union(structural_faces)
            recovered_area = recovered_union.area
            
        # Compute Convex Hull of all PSLG lines (since lines represent the carton boundary)
        hull_poly = None
        hull_area = 0.0
        
        all_points = []
        for edge in pslg_edges:
            all_points.extend(list(edge.coords))
            
        if all_points:
            multipoint = MultiPoint(all_points)
            hull_poly = multipoint.convex_hull
            hull_area = hull_poly.area
            
        coverage_ratio = recovered_area / hull_area if hull_area > 0 else 0
        missing_area_pct = max(0, 1.0 - coverage_ratio) * 100
        
        boundary_results.append({
            "image_id": image_id,
            "recovered_area": recovered_area,
            "hull_area": hull_area,
            "coverage_ratio": coverage_ratio
        })
        
        coverage_results.append({
            "image_id": image_id,
            "faces_extracted": len(structural_faces),
            "face_area": recovered_area,
            "coverage_pct": (recovered_area / carton_area) * 100,
            "hull_area": hull_area,
            "missing_area_pct": missing_area_pct
        })
        
        # Visualize
        overlay = original_img.copy()
        
        # Draw Convex Hull
        if hull_poly and isinstance(hull_poly, Polygon):
            hull_coords = np.array(hull_poly.exterior.coords, dtype=np.int32)
            cv2.polylines(overlay, [hull_coords], isClosed=True, color=(0, 255, 255), thickness=3) # Yellow hull
            
        # Draw Recovered Faces
        for face in structural_faces:
            ext_coords = np.array(face.exterior.coords, dtype=np.int32)
            # Fill faces green with alpha
            overlay_copy = overlay.copy()
            cv2.fillPoly(overlay_copy, [ext_coords], (0, 255, 0))
            cv2.addWeighted(overlay_copy, 0.4, overlay, 0.6, 0, overlay)
            
        out_overlay = out_dir / f"{image_id}_face_coverage_overlay.png"
        cv2.imwrite(str(out_overlay), overlay)
        
    # Generate Outer Boundary Report
    report_path1 = project_root / "data" / "interim" / "outer_boundary_report.md"
    with open(report_path1, "w", encoding="utf-8") as f:
        f.write("# Outer Boundary Audit Report\n\n")
        f.write("| Image ID | Recovered Face Area | Convex Hull Area | Coverage Ratio |\n")
        f.write("|---|---|---|---|\n")
        for res in boundary_results:
            f.write(f"| {res['image_id']} | {res['recovered_area']:.1f} | {res['hull_area']:.1f} | {res['coverage_ratio']:.3f} |\n")
            
    # Generate Face Coverage Forensics Report
    report_path2 = project_root / "data" / "interim" / "face_coverage_report.md"
    with open(report_path2, "w", encoding="utf-8") as f:
        f.write("# Face Coverage Forensics Report\n\n")
        f.write("| Image ID | Faces Extracted | Face Area | Coverage % (of Image) | Convex Hull Area | Missing Area % (of Hull) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for res in coverage_results:
            f.write(f"| {res['image_id']} | {res['faces_extracted']} | {res['face_area']:.1f} | {res['coverage_pct']:.1f}% | {res['hull_area']:.1f} | {res['missing_area_pct']:.1f}% |\n")
            
    logger.info("Outer Boundary Audit Complete.")

if __name__ == "__main__":
    run_outer_boundary_audit()
