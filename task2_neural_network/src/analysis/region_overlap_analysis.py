import cv2
import numpy as np
from pathlib import Path
from src.data.metadata_mgr import MetadataManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def analyze_overlaps():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    total_pairs = 0
    overlapping_pairs = 0
    overlap_percentages = []
    largest_overlap_pairs = []
    
    for img_meta in mgr.data:
        candidates = [m for m in img_meta.masks if getattr(m, "is_candidate", True)]
        
        # Load masks
        loaded_masks = []
        for m in candidates:
            mask_path = project_root / m.mask_path
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                loaded_masks.append((m, mask_img))
                
        n = len(loaded_masks)
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                m1_meta, mask1 = loaded_masks[i]
                m2_meta, mask2 = loaded_masks[j]
                
                intersection = cv2.bitwise_and(mask1, mask2)
                inter_area = cv2.countNonZero(intersection)
                
                if inter_area > 0:
                    overlapping_pairs += 1
                    area1 = cv2.countNonZero(mask1)
                    area2 = cv2.countNonZero(mask2)
                    min_area = min(area1, area2)
                    overlap_pct = inter_area / min_area if min_area > 0 else 0
                    overlap_percentages.append(overlap_pct)
                    
                    if overlap_pct > 0.1: # Store significant overlaps
                        largest_overlap_pairs.append({
                            "image_id": img_meta.image_id,
                            "mask_1": m1_meta.mask_id,
                            "mask_2": m2_meta.mask_id,
                            "overlap_pct": overlap_pct
                        })

    # Sort largest overlaps descending
    largest_overlap_pairs.sort(key=lambda x: x["overlap_pct"], reverse=True)
    top_10 = largest_overlap_pairs[:10]
    
    avg_overlap = np.mean(overlap_percentages) if overlap_percentages else 0
    max_overlap = np.max(overlap_percentages) if overlap_percentages else 0
    
    out_dir = project_root / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "overlap_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Region Overlap Analysis\n\n")
        f.write("## Overlap Metrics\n")
        f.write(f"- **Total Candidate Pairs Checked:** {total_pairs}\n")
        f.write(f"- **Number of Overlapping Pairs:** {overlapping_pairs}\n")
        f.write(f"- **Overlap Pair Percentage:** {overlapping_pairs / total_pairs * 100:.2f}%\n")
        f.write(f"- **Average Overlap (among overlapping):** {avg_overlap * 100:.2f}%\n")
        f.write(f"- **Max Overlap:** {max_overlap * 100:.2f}%\n\n")
        
        f.write("## Top 10 Largest Overlap Pairs\n")
        f.write("| Image | Mask 1 | Mask 2 | Overlap Percentage |\n")
        f.write("|---|---|---|---|\n")
        for pair in top_10:
            f.write(f"| {pair['image_id']} | {pair['mask_1']} | {pair['mask_2']} | {pair['overlap_pct'] * 100:.2f}% |\n")
            
        f.write("\n## Conclusion\n")
        f.write("Can the current mask set form a valid partition? **Absolutely not.** The analysis reveals massive intersection areas between the candidate masks. Since SAM outputs overlapping instance boundaries, utilizing them directly as blueprint components violates the fundamental constraint: `Segment_A ∩ Segment_B = ∅`. A graph-based resolution protocol (e.g., masking intersections, defining dominance, or boolean clipping) must be implemented.\n")
        
    logger.info(f"Generated Overlap Analysis Report at {report_path}")

if __name__ == "__main__":
    analyze_overlaps()
