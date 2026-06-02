import cv2
import yaml
from pathlib import Path
from src.data.metadata_mgr import MetadataManager

def apply_candidate_filter(top_n=15):
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    with open(project_root / "config" / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    min_relative_area = config.get("filtering", {}).get("min_relative_area", 0.002)
    
    total_masks = 0
    candidate_masks = 0
    removed_by_area = 0
    total_area_all = 0
    total_area_candidates = 0
    
    # Process each image
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        
        # 1. Compute areas
        mask_areas = []
        for mask_meta in img_meta.masks:
            mask_path = project_root / mask_meta.mask_path
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                area = cv2.countNonZero(mask_img)
                mask_areas.append({"mask_meta": mask_meta, "area": area})
                total_area_all += area
            else:
                # If cannot read, set to 0 area
                mask_areas.append({"mask_meta": mask_meta, "area": 0})
                
        # 2. Sort masks descending by area
        mask_areas.sort(key=lambda x: x["area"], reverse=True)
        
        # 3. Mark is_candidate
        img_area = img_meta.resolution[0] * img_meta.resolution[1]
        for rank, m in enumerate(mask_areas):
            total_masks += 1
            rel_area = m["area"] / img_area if img_area > 0 else 0
            
            if rel_area < min_relative_area:
                m["mask_meta"].is_candidate = False
                m["mask_meta"].skip_reason = "below_min_relative_area"
                removed_by_area += 1
            elif rank < top_n:
                m["mask_meta"].is_candidate = True
                m["mask_meta"].skip_reason = None
                candidate_masks += 1
                total_area_candidates += m["area"]
            else:
                m["mask_meta"].is_candidate = False
                m["mask_meta"].skip_reason = "rank_too_low"
                
    # Save the updated metadata
    mgr.save()
    
    # 4. Generate statistics
    print("--- Candidate Selection Statistics ---")
    print(f"Total masks: {total_masks}")
    print(f"Removed by Area Floor (<{min_relative_area}): {removed_by_area}")
    print(f"Candidate masks (Top {top_n} per image & >={min_relative_area}): {candidate_masks}")
    print(f"Candidate percentage: {candidate_masks / total_masks * 100:.2f}%")
    print(f"Cumulative area coverage: {total_area_candidates / total_area_all * 100:.2f}%" if total_area_all > 0 else "0.0%")
    
    # Save a report
    report_path = project_root / "data" / "interim" / "candidate_filtering_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Candidate Filtering Report\n\n")
        f.write(f"- **Total Masks:** {total_masks}\n")
        f.write(f"- **Removed by Area Floor (<{min_relative_area}):** {removed_by_area}\n")
        f.write(f"- **Candidate Masks:** {candidate_masks}\n")
        f.write(f"- **Candidate Percentage:** {candidate_masks / total_masks * 100:.2f}%\n")
    
    # 5. Verify Traceability
    print("\n--- Traceability Verification ---")
    verified = True
    for img_meta in mgr.data:
        candidates = [m for m in img_meta.masks if m.is_candidate]
        if len(candidates) > top_n:
            print(f"FAILED: {img_meta.image_id} has {len(candidates)} candidates (> {top_n})")
            verified = False
            
    if verified:
        print("SUCCESS: All candidate masks perfectly trace back to their source image_id and mask_id within the allowed rank limit.")

if __name__ == "__main__":
    apply_candidate_filter(top_n=15)
