import os
import numpy as np

# Import configuration constants
from config import (
    REFERENCE_IMAGE,
    TARGET_IMAGE,
    OUTPUT_DASHBOARD
)

# Import modular sub-packages
from preprocessing import load_image, extract_foreground
from clustering import MeanShiftExtractor
from color_processing import (
    get_dominant_color,
    get_color_family,
    get_meta_family,
    rgb_to_hex,
    CompareColors
)
from visualization import generate_comparison_dashboard
from utils import print_comparison_results

def main():
    print("Stage 1: Initializing Perceptual Text-Color Semantic Comparison Pipeline...")
    
    # Ensure output directory exists (Refinement 2)
    output_dir = os.path.dirname(OUTPUT_DASHBOARD)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        # 1. Load Images
        print("Stage 2: Loading cropped text images...")
        ref_image = load_image(REFERENCE_IMAGE)
        tgt_image = load_image(TARGET_IMAGE)
        
        # 2. Extract Foreground Masks & Colors with Empty Foreground Safety Checks (Refinement 3)
        print("Stage 3: Extracting foreground text masks...")
        ref_fg = extract_foreground(ref_image)
        tgt_fg = extract_foreground(tgt_image)
        
        # 3. Perform LAB MeanShift Clustering
        print("Stage 4: Performing density-based MeanShift clustering...")
        ref_extractor = MeanShiftExtractor(ref_fg["lab"])
        tgt_extractor = MeanShiftExtractor(tgt_fg["lab"])
        
        ref_cluster_result = ref_extractor.extract_colors(
            delta_from_bg_values=ref_fg["delta_from_bg_values"],
            is_lab=True
        )
        tgt_cluster_result = tgt_extractor.extract_colors(
            delta_from_bg_values=tgt_fg["delta_from_bg_values"],
            is_lab=True
        )
        
        # 4. Extract Semantic Dominant Color
        print("Stage 5: Selecting semantic dominant color clusters...")
        ref_color = get_dominant_color(ref_cluster_result["cluster_info"])
        tgt_color = get_dominant_color(tgt_cluster_result["cluster_info"])
        
        # 5. Determine Color Family Mapping dynamically using colour-science (Refined Hierarchy)
        print("Stage 6: Mapping dominant colors to semantic families...")
        ref_exact = get_color_family(ref_color)
        ref_meta = get_meta_family(ref_exact)
        
        tgt_exact = get_color_family(tgt_color)
        tgt_meta = get_meta_family(tgt_exact)
        
        # 6. Format Colors & Calculate Distance
        ref_rgb_list = ref_color.tolist()
        tgt_rgb_list = tgt_color.tolist()
        
        ref_hex = rgb_to_hex(ref_color)
        tgt_hex = rgb_to_hex(tgt_color)
        
        comparator = CompareColors()
        comparison = comparator.is_match(ref_color, tgt_color)
        delta_e = comparison["delta_e"]
        
        # Calculate Euclidean RGB Distance
        rgb_dist = float(np.linalg.norm(np.array(ref_color) - np.array(tgt_color)))
        
        same_meta = "YES" if ref_meta == tgt_meta else "NO"
        
        # Final Match decision: MATCH if perceptually very close (DeltaE <= Match Threshold)
        match_status = "MATCH" if comparison["match"] else "NO MATCH"
        
        # 7. Print Standard Comparative Tabular Log
        print_comparison_results(
            ref_rgb=ref_rgb_list,
            ref_hex=ref_hex,
            ref_exact=ref_exact,
            ref_meta=ref_meta,
            tgt_rgb=tgt_rgb_list,
            tgt_hex=tgt_hex,
            tgt_exact=tgt_exact,
            tgt_meta=tgt_meta,
            delta_e=delta_e,
            rgb_dist=rgb_dist,
            same_meta=same_meta,
            match_status=match_status
        )
        
        # 8. Squeeze and Map MeanShift Cluster Labels Spatially for Visualization
        ref_fg_indices = np.where(ref_fg["mask"] > 0)
        ref_cluster_map = np.zeros(ref_image.shape[:2], dtype=np.uint8)
        ref_cluster_map[ref_fg_indices[0], ref_fg_indices[1]] = ref_cluster_result["labels"] + 1
        
        tgt_fg_indices = np.where(tgt_fg["mask"] > 0)
        tgt_cluster_map = np.zeros(tgt_image.shape[:2], dtype=np.uint8)
        tgt_cluster_map[tgt_fg_indices[0], tgt_fg_indices[1]] = tgt_cluster_result["labels"] + 1
        
        # 9. Generate and Save Comparison Dashboard
        print(f"Stage 7: Generating high-fidelity Comparison Dashboard at {OUTPUT_DASHBOARD}...")
        generate_comparison_dashboard(
            reference_image=ref_image,
            reference_mask=ref_fg["mask"],
            ref_cluster_map=ref_cluster_map,
            reference_color=ref_color,
            ref_rgb_list=ref_rgb_list,
            ref_hex=ref_hex,
            ref_exact=ref_exact,
            ref_meta=ref_meta,
            target_image=tgt_image,
            target_mask=tgt_fg["mask"],
            tgt_cluster_map=tgt_cluster_map,
            target_color=tgt_color,
            tgt_rgb_list=tgt_rgb_list,
            tgt_hex=tgt_hex,
            tgt_exact=tgt_exact,
            tgt_meta=tgt_meta,
            delta_e=delta_e,
            rgb_dist=rgb_dist,
            same_meta=same_meta,
            match_status=match_status,
            output_path=OUTPUT_DASHBOARD
        )
        print("Success: Comparative pipeline executed flawlessly.")

    except Exception as e:
        print(f"Failure: Comparative pipeline execution failed with error: {str(e)}")

if __name__ == "__main__":
    main()
