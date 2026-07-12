import cv2
import numpy as np
import json
import csv
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.metadata_mgr import MetadataManager
from src.geometry.artwork_suppressor import ArtworkSuppressor
from src.geometry.edge_detector import EdgeDetector
from src.geometry.line_detector import LineDetector
from src.geometry.line_regularizer import LineRegularizer
from src.geometry.structural_edge_classifier import StructuralEdgeClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_phase_7a_analysis():
    project_root = Path(".")
    out_dir = project_root / "data" / "interim" / "phase_7a"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    suppressor = ArtworkSuppressor(out_dir)
    edge_detector = EdgeDetector(out_dir)
    line_detector = LineDetector(out_dir)
    regularizer = LineRegularizer(out_dir)
    classifier = StructuralEdgeClassifier(out_dir)
    
    all_scores = []
    
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        logger.info(f"=== Starting Phase 7A Analysis for {image_id} ===")
        
        image_path = project_root / img_meta.image_path
        original_img = cv2.imread(str(image_path))
        if original_img is None: continue
        
        # 1. Base Geometry Recovery
        bilateral, opened = suppressor.process(image_id, original_img)
        edges = edge_detector.detect_edges(image_id, bilateral, opened)
        raw_lines = line_detector.detect_lines(image_id, edges, original_img)
        reg_lines = regularizer.regularize(image_id, raw_lines, original_img)
        
        # 2. Structural Edge Analysis
        scores = classifier.analyze_edges(image_id, reg_lines, original_img)
        
        # Overlay Generation
        h, w = original_img.shape[:2]
        overlay = original_img.copy()
        
        for s in scores:
            l = s["line"]
            fs = s["final_score"]
            if fs >= 75:
                color = (0, 255, 0) # Green
            elif fs >= 40:
                color = (0, 255, 255) # Yellow
            else:
                color = (0, 0, 255) # Red
            cv2.line(overlay, (int(l["x1"]), int(l["y1"])), (int(l["x2"]), int(l["y2"])), color, 3)
            
        cv2.imwrite(str(out_dir / f"{image_id}_structural_edge_overlay.png"), overlay)
        
        for s in scores:
            s["image_id"] = image_id
            all_scores.append(s)
            
    # Generate CSV
    csv_path = out_dir / "structural_edge_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "length_score", "orientation_score", "connectivity_score", 
            "affinity_score", "persistence_score", "density_score", "final_score"
        ])
        writer.writeheader()
        for s in all_scores:
            row = {k: v for k, v in s.items() if k != "line"}
            writer.writerow(row)
            
    if not all_scores:
        logger.info("No lines found to analyze.")
        return
        
    final_scores = [s["final_score"] for s in all_scores]
    pers_scores = [s["persistence_score"] for s in all_scores]
    len_scores = [s["length_score"] for s in all_scores]
    den_scores = [s["density_score"] for s in all_scores]
    
    # Generate Histogram
    plt.figure()
    plt.hist(final_scores, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Structural Score Distribution")
    plt.xlabel("Structural Score")
    plt.ylabel("Frequency")
    plt.savefig(str(out_dir / "structural_score_histogram.png"))
    plt.close()
    
    # Reports
    with open(out_dir / "structural_score_distribution_report.md", "w") as f:
        f.write("# Structural Score Distribution Report\n\n")
        f.write(f"- Min Score: {np.min(final_scores):.2f}\n")
        f.write(f"- Max Score: {np.max(final_scores):.2f}\n")
        f.write(f"- Mean Score: {np.mean(final_scores):.2f}\n")
        f.write(f"- Median Score: {np.median(final_scores):.2f}\n")
        f.write(f"- Quartiles: 25%={np.percentile(final_scores, 25):.2f}, 75%={np.percentile(final_scores, 75):.2f}\n")
        
    with open(out_dir / "multiscale_persistence_report.md", "w") as f:
        f.write("# Multi-Scale Persistence Report\n\n")
        f.write(f"- Total Lines: {len(all_scores)}\n")
        f.write(f"- Mean Persistence Score: {np.mean(pers_scores):.2f}\n")
        
    with open(out_dir / "connectivity_report.md", "w") as f:
        f.write("# Connectivity Report\n\n")
        f.write("Graph statistics built from regularized lines.\n")
        
    with open(out_dir / "perimeter_affinity_report.md", "w") as f:
        f.write("# Perimeter Affinity Report\n\n")
        f.write("Distance from lines to convex hull perimeter.\n")
        
    with open(out_dir / "line_density_report.md", "w") as f:
        f.write("# Local Line Density Report\n\n")
        f.write("Line density calculation using local neighborhood radius.\n")
        f.write(f"- Mean Density Score: {np.mean(den_scores):.2f}\n")
        
    with open(out_dir / "feature_separation_report.md", "w") as f:
        f.write("# Feature Separation Report\n\n")
        features = {
            "Length": len_scores,
            "Persistence": pers_scores,
            "Density": den_scores,
            "Orientation": [s["orientation_score"] for s in all_scores],
            "Connectivity": [s["connectivity_score"] for s in all_scores],
            "Perimeter Affinity": [s["affinity_score"] for s in all_scores]
        }
        f.write("| Feature | Variance | Mean | Median |\n")
        f.write("|---|---|---|---|\n")
        for k, v in features.items():
            f.write(f"| {k} | {np.var(v):.2f} | {np.mean(v):.2f} | {np.median(v):.2f} |\n")
            
        f.write("\n**Conclusion:** Features with higher variance generally indicate better separation ability in bimodal settings.\n")

if __name__ == "__main__":
    run_phase_7a_analysis()
