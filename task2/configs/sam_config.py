"""
SAM-specific configuration defaults.

All values can be overridden by passing a merged dict to SAMSegmenter,
SAMMaskFiller, or SAMRegionExtractor.
"""

sam_config = {
    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    # SAM model variant: "vit_h" | "vit_l" | "vit_b"
    "sam_model_type": "vit_h",

    # ---------------------------------------------------------------
    # Automatic Mask Generator
    # ---------------------------------------------------------------
    # Number of points sampled per side of the image grid
    "sam_points_per_side": 32,

    # Minimum predicted IoU for a mask to be kept by SAM internally
    "pred_iou_thresh": 0.88,

    # Minimum stability score for a mask to be kept by SAM internally
    "stability_score_thresh": 0.95,

    # Number of extra crop layers (0 = no crops)
    "crop_n_layers": 0,

    # Points per layer when crop_n_layers > 0
    "crop_n_points_per_layer": 1,

    # Minimum connected-region area (pixels) post-processing inside SAM
    "min_mask_region_area": 100,

    # Output mode for SamAutomaticMaskGenerator
    "output_mode": "binary_mask",

    # ---------------------------------------------------------------
    # SAMMaskFiller — post-processing gates
    # ---------------------------------------------------------------
    # Area (pixels) gates — discard masks outside this range
    "min_area": 5000,
    "max_area": 500_000,

    # Predicted-IoU gate — discard masks with score below this
    "iou_score_threshold": 0.80,

    # Stability gate — discard masks with score below this
    "stability_score_threshold": 0.88,

    # Geometry — aspect ratio limits
    "max_aspect_ratio": 10.0,
    "min_aspect_ratio": 0.1,

    # Geometry — minimum fill ratio (mask_area / bbox_area)
    "min_fill_ratio": 0.25,

    # Duplicate removal — mask-IoU above which the lower-quality
    # mask is treated as a duplicate and dropped
    "duplicate_iou_threshold": 0.85,

    # ---------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------
    # Transparency of the coloured mask fill in SAMOverlayGenerator
    "mask_alpha": 0.35,

    # Font scale for score labels in SAMOverlayGenerator
    "label_font_scale": 0.38,
}
