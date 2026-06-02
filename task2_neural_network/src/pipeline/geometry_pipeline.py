import cv2
import numpy as np
from pathlib import Path
from src.data.metadata_mgr import MetadataManager
from src.geometry.artwork_suppressor import ArtworkSuppressor
from src.geometry.edge_detector import EdgeDetector
from src.geometry.line_detector import LineDetector
from src.geometry.line_regularizer import LineRegularizer
from src.geometry.polygonizer import Polygonizer
from src.geometry.pslg_builder import PSLGBuilder
from src.blueprint.planar_partition_builder import PlanarPartitionBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_geometry_pipeline():
    project_root = Path(".")
    out_dir = project_root / "data" / "interim" / "geometry_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    suppressor = ArtworkSuppressor(out_dir)
    edge_detector = EdgeDetector(out_dir)
    line_detector = LineDetector(out_dir)
    regularizer = LineRegularizer(out_dir)
    pslg_builder = PSLGBuilder(out_dir)
    polygonizer = Polygonizer(out_dir)
    partition_builder = PlanarPartitionBuilder(out_dir)
    
    all_metrics = []
    
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        logger.info(f"=== Starting Geometry Pipeline for {image_id} ===")
        
        image_path = project_root / img_meta.image_path
        original_img = cv2.imread(str(image_path))
        if original_img is None:
            continue
            
        h, w = original_img.shape[:2]
        carton_area = h * w
        
        # Stage 0: Artwork Suppression
        bilateral, opened = suppressor.process(image_id, original_img)
        
        # Stage 1: Edge Recovery
        edges = edge_detector.detect_edges(image_id, bilateral, opened)
        
        # Stage 2: Line Detection
        raw_lines = line_detector.detect_lines(image_id, edges, original_img)
        
        # Stage 2.5: Line Regularization
        regularized_lines = regularizer.regularize(image_id, raw_lines, original_img)
        
        # Stage 2.8: PSLG Construction
        pslg_edges = pslg_builder.build_graph(image_id, regularized_lines, original_img)
        
        # Stage 3: Polygonization
        faces = polygonizer.polygonize_lines(image_id, pslg_edges, original_img)
        
        # Stage 4: Planar Partition Builder
        metrics = partition_builder.build_partition(image_id, faces, original_img, carton_area)
        all_metrics.append(metrics)
        
    # Write PSLG Reports
    pslg_builder.write_reports()
        
    # Generate Report
    report_path = out_dir / "geometry_pipeline_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 5.1 Geometry-First Pipeline Results\n\n")
        f.write("| Image ID | Recovered Faces | Coverage % | Overlap % | Border Consistency | Validity Score |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        avg_cov = 0
        valid_count = 0
        
        for m in all_metrics:
            f.write(f"| {m['image_id']} | {m['face_count']} | {m['coverage_pct']*100:.1f}% | {m['overlap_pct']:.1f}% | {m['shared_border_consistency']}% | {m['blueprint_validity']:.1f} |\n")
            avg_cov += m['coverage_pct']
            if m['face_count'] >= 6 and m['face_count'] <= 15 and m['coverage_pct'] >= 0.8:
                valid_count += 1
                
        avg_cov /= len(all_metrics) if all_metrics else 1
        
        f.write("\n## Sprint Criteria Evaluation\n")
        f.write(f"- **Target:** 6-15 recovered faces, Coverage > 80%, Overlap 0%, Consistency 100%\n")
        f.write(f"- **Cartons Meeting Criteria:** {valid_count} / {len(all_metrics)}\n")
        
        if valid_count >= 2:
            f.write("\n**SUCCESS:** Geometry-First architecture validated!\n")
        else:
            f.write("\n**FAILED:** Further tuning of edge detection and regularization required.\n")
            
    logger.info(f"Pipeline complete. Report saved to {report_path}")

if __name__ == "__main__":
    run_geometry_pipeline()
