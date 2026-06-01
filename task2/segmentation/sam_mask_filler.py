import cv2
import numpy as np


class SAMMaskFiller:
    """
    Post-processing filter for raw SAM regions.

    Receives a list of region dicts (output of SAMRegionExtractor) and
    applies a configurable quality-gate chain before the regions enter the
    shared pipeline (RegionMerger → RegionAnalyzer → RectangleFitter …).

    All thresholds are read from the instance config so no arguments need
    to be threaded through every call site.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    # ------------------------------------------------------------------
    # PRIMARY ENTRY POINT
    # ------------------------------------------------------------------

    def filter_masks(self, regions: list) -> list:
        """
        Run the full filter chain using thresholds from self.config.

        Chain order:
            1. Area gate
            2. IoU score gate
            3. Stability score gate
            4. Geometry gate (aspect ratio + fill ratio)
            5. Duplicate removal (mask-IoU)

        Parameters
        ----------
        regions : list[dict]
            Raw regions from SAMRegionExtractor.

        Returns
        -------
        list[dict]
            Filtered regions.
        """
        min_area  = self.config.get("min_area",  5_000)
        max_area  = self.config.get("max_area",  500_000)
        iou_thr   = self.config.get("iou_score_threshold",      0.75)
        stab_thr  = self.config.get("stability_score_threshold", 0.85)
        dup_thr   = self.config.get("duplicate_iou_threshold",   0.85)

        filtered = self.filter_by_area(regions, min_area, max_area)
        filtered = self.filter_by_iou(filtered, iou_thr)
        filtered = self.filter_by_stability(filtered, stab_thr)
        filtered = self.filter_by_geometry(filtered)
        filtered = self.remove_duplicate_masks(filtered, dup_thr)

        return filtered

    # ------------------------------------------------------------------
    # INDIVIDUAL FILTERS
    # ------------------------------------------------------------------

    def filter_by_area(self, regions: list, min_area: float, max_area: float) -> list:
        """
        Keep only regions whose pixel area falls within [min_area, max_area].
        """
        filtered = []
        for r in regions:
            area = r.get("area", 0)
            if min_area <= area <= max_area:
                filtered.append(r)
        return filtered

    def filter_by_iou(self, regions: list, threshold: float) -> list:
        """
        Discard regions whose SAM predicted_iou is below *threshold*.

        Region key: "iou"
        """
        if threshold <= 0.0:
            return list(regions)
        filtered = []
        for r in regions:
            if r.get("iou", 1.0) >= threshold:
                filtered.append(r)
        return filtered

    def filter_by_stability(self, regions: list, threshold: float) -> list:
        """
        Discard regions whose SAM stability_score is below *threshold*.

        Region key: "stability"
        """
        if threshold <= 0.0:
            return list(regions)
        filtered = []
        for r in regions:
            if r.get("stability", 1.0) >= threshold:
                filtered.append(r)
        return filtered

    def filter_by_geometry(self, regions: list) -> list:
        """
        Remove regions with degenerate geometry:
          - Extreme aspect ratios  (very thin strips)
          - Poor fill ratios       (sparse / hollow masks)
        """
        max_ar   = self.config.get("max_aspect_ratio", 10.0)
        min_ar   = self.config.get("min_aspect_ratio",  0.1)
        min_fill = self.config.get("min_fill_ratio",    0.25)

        filtered = []
        for r in regions:
            bbox = r.get("bbox", {})
            w    = bbox.get("w", 0)
            h    = bbox.get("h", 0)

            if h == 0 or w == 0:
                continue

            ar = float(w) / h
            if ar > max_ar or ar < min_ar:
                continue

            area      = r.get("area", 0)
            bbox_area = w * h
            fill      = float(area) / bbox_area if bbox_area > 0 else 0.0
            if fill < min_fill:
                continue

            filtered.append(r)
        return filtered

    def remove_duplicate_masks(self, regions: list, threshold: float) -> list:
        """
        Greedily remove masks that overlap significantly with a higher-quality
        mask already kept (mask pixel-IoU > *threshold*).

        Regions are sorted descending by combined quality (iou × stability)
        so the best mask wins each overlapping pair.
        """
        if not regions or threshold >= 1.0:
            return list(regions)

        def _quality(r):
            return r.get("iou", 1.0) * r.get("stability", 1.0)

        sorted_regions = sorted(regions, key=_quality, reverse=True)
        kept = []

        for candidate in sorted_regions:
            mask_c = candidate.get("binary_mask")
            if mask_c is None:
                kept.append(candidate)
                continue

            is_dup = False
            for kept_r in kept:
                mask_k = kept_r.get("binary_mask")
                if mask_k is None:
                    continue
                if self._mask_iou(mask_c, mask_k) > threshold:
                    is_dup = True
                    break

            if not is_dup:
                kept.append(candidate)

        return kept

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _mask_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Pixel-level Intersection-over-Union of two binary masks."""
        inter = cv2.countNonZero(cv2.bitwise_and(mask_a, mask_b))
        union = cv2.countNonZero(cv2.bitwise_or(mask_a, mask_b))
        if union == 0:
            return 0.0
        return float(inter) / union
