"""
main.py — Pharmaceutical Carton Segmentation Pipeline

Runs two parallel segmentation approaches for every image in data/input/:

  1. Structural (OpenCV-based) — StructuralSegmenter
  2. SAM-based                 — SAMSegmenter + SAMRegionExtractor +
                                  SAMMaskFiller + SAMSegmentationEvaluator

SAM is guarded: if the checkpoint file is absent the structural pipeline
still runs normally and a clear message is printed.

Set the SAM_CHECKPOINT environment variable or place the weights file at:
    task2/weights/sam_vit_h_4b8939.pth
"""

import sys
import os

# ---------------------------------------------------------------------------
# Resolve task2 package root
# main.py lives inside task2/, so its own directory IS the package root.
# ---------------------------------------------------------------------------
task2_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, task2_dir)

import cv2
import json
import traceback

import numpy as np

# ── Structural pipeline ──────────────────────────────────────────────────────
from segmentation.structural_segmenter import StructuralSegmenter
from configs.config import config
from utils.loader import load_images_from_folder
from evaluation.batch_summary import BatchSummary
from debug.debug_exporter import DebugExporter
from serialization.segment_serializer import SegmentSerializer

# ── SAM pipeline ─────────────────────────────────────────────────────────────
from configs.sam_config import sam_config
from segmentation.sam_segmenter import SAMSegmenter
from segmentation.sam_region_extractor import SAMRegionExtractor
from segmentation.sam_mask_filler import SAMMaskFiller
from visualization.sam_overlay_generator import SAMOverlayGenerator
from debug.sam_debug_exporter import SAMDebugExporter
from evaluation.sam_segmentation_evaluator import SAMSegmentationEvaluator
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# SAM checkpoint location
# Override via:   set SAM_CHECKPOINT=C:\path\to\sam_vit_h_4b8939.pth
# ---------------------------------------------------------------------------
SAM_CHECKPOINT = os.environ.get(
    "SAM_CHECKPOINT",
    os.path.join(task2_dir, "weights", "sam_vit_h_4b8939.pth"),
)


# ===========================================================================
# SAM SUB-PIPELINE
# ===========================================================================

def run_sam_pipeline(
    image,
    image_name: str,
    image_output_dir: str,
    structural_mask,        # binary uint8 mask from the loader
    sam_exporter: SAMDebugExporter,
    merged_config: dict,
) -> dict:
    """
    Execute the full SAM segmentation sub-pipeline for one image.

    Steps
    -----
    1. Generate automatic masks              (SAMSegmenter)
    2. Extract + normalise regions           (SAMRegionExtractor)
    3. Filter — area / IoU / stability /
       geometry / duplicates                 (SAMMaskFiller)
    4. Evaluate quality                      (SAMSegmentationEvaluator)
    5. Visualise overlay + heatmaps          (SAMOverlayGenerator)
    6. Export debug artefacts                (SAMDebugExporter)
    7. Save outputs to disk

    Parameters
    ----------
    image            : np.ndarray   BGR source image
    image_name       : str
    image_output_dir : str          per-image output folder (already created)
    structural_mask  : np.ndarray   binary mask used as GT for coverage metric
    sam_exporter     : SAMDebugExporter
    merged_config    : dict         sam_config merged with base config

    Returns
    -------
    dict
        sam_regions, sam_raw_count, sam_filtered_count, sam_report
    """

    # ── Component instances ──────────────────────────────────────────────────
    segmenter  = SAMSegmenter(SAM_CHECKPOINT, merged_config)
    extractor  = SAMRegionExtractor(merged_config)
    filler     = SAMMaskFiller(merged_config)
    visualizer = SAMOverlayGenerator(merged_config)
    evaluator  = SAMSegmentationEvaluator(merged_config)

    # ── Step 1: Generate raw SAM masks ───────────────────────────────────────
    raw_masks = segmenter.generate_masks(image)

    # ── Step 2: Convert to project region dicts ──────────────────────────────
    raw_regions = extractor.extract_regions(raw_masks)

    # ── Step 3: Filter ───────────────────────────────────────────────────────
    filtered_regions = filler.filter_masks(raw_regions)

    print(
        f"  [SAM] raw={len(raw_regions):>3}  "
        f"filtered={len(filtered_regions):>3}"
    )

    # ── Step 4: Evaluate ─────────────────────────────────────────────────────
    sam_report = evaluator.evaluate(
        masks            = filtered_regions,
        image_shape      = image.shape,
        structural_mask  = structural_mask,
    )

    print(
        f"  [SAM] status={sam_report['status']}  "
        f"P={sam_report['mask_count_summary']['panels']}  "
        f"F={sam_report['mask_count_summary']['flaps']}  "
        f"T={sam_report['mask_count_summary']['tabs']}  "
        f"coverage="
        f"{sam_report['coverage']:.2f}" if sam_report["coverage"] is not None
        else f"  [SAM] status={sam_report['status']}  coverage=N/A"
    )

    # ── Step 5: Visualise ─────────────────────────────────────────────────────
    sam_overlay   = visualizer.generate(image, filtered_regions)
    iou_heatmap   = visualizer.generate_iou_heatmap(image.shape, filtered_regions)
    stab_heatmap  = visualizer.generate_stability_heatmap(image.shape, filtered_regions)
    eval_overlay  = evaluator.create_evaluation_overlay(image, sam_report)

    # ── Step 6: Debug export ─────────────────────────────────────────────────
    sam_debug_store = {
        "sam_01_raw_regions"      : extractor.create_extraction_debug_image(image.shape, raw_regions),
        "sam_02_filtered_regions" : extractor.create_extraction_debug_image(image.shape, filtered_regions),
        "sam_03_overlay"          : sam_overlay,
        "sam_04_iou_heatmap"      : iou_heatmap,
        "sam_05_stability_heatmap": stab_heatmap,
        "sam_06_eval_overlay"     : eval_overlay,
    }
    sam_exporter.export(image_name, sam_debug_store)
    sam_exporter.export_mask_scores(image_name, filtered_regions)
    sam_exporter.export_mask_strip(image_name, image, filtered_regions)
    sam_exporter.export_filtered_vs_raw(
        image_name, image.shape, raw_regions, filtered_regions
    )

    # ── Step 7: Save outputs ─────────────────────────────────────────────────
    # Overlay image
    if sam_overlay is not None:
        cv2.imwrite(
            os.path.join(image_output_dir, "sam_overlay.png"),
            sam_overlay,
        )

    # Evaluation report JSON
    evaluator.save_report_json(
        sam_report,
        os.path.join(image_output_dir, "sam_report.json"),
    )

    # Compact summary JSON (raw vs filtered counts + top-level status)
    sam_summary = {
        "raw_mask_count"     : len(raw_regions),
        "filtered_mask_count": len(filtered_regions),
        "status"             : sam_report["status"],
        "coverage"           : sam_report.get("coverage"),
        "mask_count_summary" : sam_report.get("mask_count_summary", {}),
    }
    with open(os.path.join(image_output_dir, "sam_summary.json"), "w") as f:
        json.dump(sam_summary, f, indent=4)

    return {
        "sam_regions"       : filtered_regions,
        "sam_raw_count"     : len(raw_regions),
        "sam_filtered_count": len(filtered_regions),
        "sam_report"        : sam_report,
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    INPUT_DIR  = os.path.join(task2_dir, "data", "input")
    OUTPUT_DIR = os.path.join(task2_dir, "data", "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Merge configs — SAM keys don't collide with structural keys
    merged_config = {**sam_config, **config}

    # ── Structural pipeline ──────────────────────────────────────────────────
    segmenter     = StructuralSegmenter(config)
    exporter      = DebugExporter(OUTPUT_DIR)
    batch_summary = BatchSummary(OUTPUT_DIR)
    serializer    = SegmentSerializer()

    # ── SAM availability check ───────────────────────────────────────────────
    sam_available = os.path.isfile(SAM_CHECKPOINT)
    if sam_available:
        sam_exporter = SAMDebugExporter(OUTPUT_DIR)
        print(f"[INFO] SAM checkpoint found  : {SAM_CHECKPOINT}")
    else:
        sam_exporter = None
        print(
            f"[INFO] SAM checkpoint NOT found at: {SAM_CHECKPOINT}\n"
            f"       Set the SAM_CHECKPOINT environment variable or place\n"
            f"       sam_vit_h_4b8939.pth in task2/weights/ to enable SAM.\n"
            f"       Running structural pipeline only."
        )

    # ── Load images ──────────────────────────────────────────────────────────
    all_images = load_images_from_folder(INPUT_DIR)
    if not all_images:
        print(f"[ERROR] No images found in {INPUT_DIR}")
        return

    print(f"[INFO] Found {len(all_images)} image(s) in {INPUT_DIR}\n")

    # ── Per-image loop ───────────────────────────────────────────────────────
    for item in all_images:
        image_name = item["name"]
        image      = item["image"]
        mask       = item["mask"]

        print(f"{'='*60}")
        print(f"  Processing: {image_name}")
        print(f"{'='*60}")

        image_output_dir = os.path.join(OUTPUT_DIR, image_name)
        os.makedirs(image_output_dir, exist_ok=True)

        # ── Structural segmentation ──────────────────────────────────────────
        try:
            result = segmenter.segment(image, mask)

            # Overlay
            cv2.imwrite(
                os.path.join(image_output_dir, "overlay.png"),
                result["overlay"],
            )

            # Segment map
            cv2.imwrite(
                os.path.join(image_output_dir, "segment_map.png"),
                result["segment_map"],
            )

            # Debug images
            exporter.export(image_name, result["debug_store"])

            # Segmentation report JSON
            if "report" in result:
                segmenter.segmentation_evaluator.save_report_json(
                    result["report"],
                    os.path.join(image_output_dir, "report.json"),
                )

            # Segments JSON
            serializer.save_segments(
                result["segments"],
                os.path.join(image_output_dir, "segments.json"),
            )

            batch_summary.add_result(image_name, result["rectangles"])

            n_segs = len(result.get("segments", []))
            n_rects = len(result.get("rectangles", []))
            print(f"  [Structural] segments={n_segs}  rectangles={n_rects}")

        except Exception as exc:
            print(f"  [Structural] ERROR: {exc}")
            traceback.print_exc()

        # ── SAM segmentation ─────────────────────────────────────────────────
        if sam_available:
            try:
                run_sam_pipeline(
                    image            = image,
                    image_name       = image_name,
                    image_output_dir = image_output_dir,
                    structural_mask  = mask,
                    sam_exporter     = sam_exporter,
                    merged_config    = merged_config,
                )
            except Exception as exc:
                print(f"  [SAM] ERROR: {exc}")
                traceback.print_exc()

        print()

    # ── Batch summary ────────────────────────────────────────────────────────
    batch_summary.generate_report()
    batch_summary.save_report()
    batch_summary.print_console_report()
    print("\nAll processing successfully completed!")


if __name__ == "__main__":
    main()
