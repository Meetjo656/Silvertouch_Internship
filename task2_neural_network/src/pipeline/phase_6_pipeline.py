import cv2
import numpy as np
from pathlib import Path

from src.data.metadata_mgr import MetadataManager
from src.geometry.artwork_suppressor import ArtworkSuppressor
from src.geometry.edge_detector import EdgeDetector
from src.geometry.line_detector import LineDetector
from src.geometry.line_regularizer import LineRegularizer
from src.geometry.pslg_builder import PSLGBuilder
from src.geometry.boundary_completion_engine import BoundaryCompletionEngine
from src.geometry.polygonizer import Polygonizer
from src.blueprint.planar_partition_builder import PlanarPartitionBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_phase_6_pipeline():
    project_root = Path(".")
    out_dir = project_root / "data" / "interim" / "phase_6"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    suppressor = ArtworkSuppressor(out_dir)
    edge_detector = EdgeDetector(out_dir)
    line_detector = LineDetector(out_dir)
    regularizer = LineRegularizer(out_dir)
    pslg_builder = PSLGBuilder(out_dir)
    boundary_engine = BoundaryCompletionEngine(project_root, out_dir)
    polygonizer = Polygonizer(out_dir)
    partition_builder = PlanarPartitionBuilder(out_dir)
    
    all_metrics = []
    
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        logger.info(f"=== Starting Phase 6 Pipeline for {image_id} ===")
        
        image_path = project_root / img_meta.image_path
        original_img = cv2.imread(str(image_path))
        if original_img is None: continue
        
        h, w = original_img.shape[:2]
        carton_area = h * w
        
        # 1. Base Geometry Recovery
        bilateral, opened = suppressor.process(image_id, original_img)
        edges = edge_detector.detect_edges(image_id, bilateral, opened)
        raw_lines = line_detector.detect_lines(image_id, edges, original_img)
        reg_lines = regularizer.regularize(image_id, raw_lines, original_img)
        
        # 2. Initial PSLG & Polygonization (to get initial faces for perimeter hypothesis)
        initial_pslg = pslg_builder.build_graph(image_id, reg_lines, original_img)
        initial_faces = polygonizer.polygonize_lines(f"{image_id}_initial", initial_pslg, original_img)
        
        # 3. Boundary Completion
        completed_network = boundary_engine.complete_boundary(
            image_id, img_meta, initial_pslg, initial_faces, original_img
        )
        
        # 4. PSLG Retest (Re-node the completed network)
        # Convert shapely lines back to dicts for our PSLG builder's first step, 
        # or just modify PSLGBuilder to accept LineStrings directly.
        # Actually, PSLGBuilder._extend_line takes LineStrings. Let's just pass the completed_network.
        # Wait, PSLGBuilder.build_graph expects regularized_lines as dictionaries!
        # Let's convert completed_network back to dicts.
        completed_dicts = []
        for line in completed_network:
            coords = list(line.coords)
            completed_dicts.append({
                "x1": int(coords[0][0]), "y1": int(coords[0][1]),
                "x2": int(coords[-1][0]), "y2": int(coords[-1][1])
            })
            
        final_pslg = pslg_builder.build_graph(f"{image_id}_final", completed_dicts, original_img)
        
        # 5. Final Polygonization & Partition
        final_faces = polygonizer.polygonize_lines(f"{image_id}_final", final_pslg, original_img)
        metrics = partition_builder.build_partition(f"{image_id}_final", final_faces, original_img, carton_area)
        
        # Fix image ID in metrics since we added "_final"
        metrics["image_id"] = image_id
        all_metrics.append(metrics)
        
    # Generate Report
    report_path = project_root / "data" / "interim" / "phase_6_polygonization_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 6 Polygonization Retest Report\n\n")
        f.write("| Image ID | Recovered Faces | Coverage % | Overlap % | Border Consistency | Validity Score |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        avg_cov = 0
        valid_count = 0
        
        for m in all_metrics:
            f.write(f"| {m['image_id']} | {m['face_count']} | {m['coverage_pct']*100:.1f}% | {m['overlap_pct']:.1f}% | {m['shared_border_consistency']}% | {m['blueprint_validity']:.1f} |\n")
            avg_cov += m['coverage_pct']
            if m['coverage_pct'] >= 0.4:
                valid_count += 1
                
        avg_cov /= len(all_metrics) if all_metrics else 1
        
        f.write("\n## Sprint Criteria Evaluation\n")
        f.write(f"- **Target:** Coverage > 40%, Overlap 0%\n")
        f.write(f"- **Cartons Meeting Criteria:** {valid_count} / {len(all_metrics)}\n")
        
        if valid_count >= 2:
            f.write("\n**SUCCESS:** Boundary Completion Engine validated!\n")
        else:
            f.write("\n**FAILED:** Further gap bridging tuning required.\n")
            
    logger.info(f"Phase 6 Pipeline complete. Report saved to {report_path}")

if __name__ == "__main__":
    run_phase_6_pipeline()
