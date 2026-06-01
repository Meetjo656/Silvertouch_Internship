"""
evaluation/sam_segmentation_evaluator.py

Evaluates the quality of SAM-generated masks specifically for pharmaceutical
carton structural segmentation.

The goal is NOT generic segmentation quality measurement.  The goal is to
answer:

    "Do these SAM masks correspond to meaningful carton structures such as
     Front Panels, Back Panels, Side Panels, Top/Bottom Flaps, Glue Flaps,
     Lock Tabs, and Dust Flaps?"

All public methods accept either SAMMask objects or the project-standard
region dicts so the evaluator can be inserted at any point in the pipeline.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# SAMMask is imported for type hints only; we never crash if it is absent.
try:
    from models.sam_mask import SAMMask
    _SAMMaskType = SAMMask
except ImportError:  # pragma: no cover
    _SAMMaskType = object  # type: ignore

# ---------------------------------------------------------------------------
# Type alias for the dual-input contract
# ---------------------------------------------------------------------------
MaskInput = Union["SAMMask", dict]


class SAMSegmentationEvaluator:
    """
    Quantitative evaluator for SAM segmentation results on pharmaceutical
    carton images.

    Responsibilities
    ----------------
    * Measure how well the union of SAM masks covers a known structural mask.
    * Detect over-segmentation (fragmentation) vs under-segmentation.
    * Summarise mask-area and SAM quality-score distributions.
    * Estimate how many masks are plausible carton structure candidates.
    * Measure raw bounding-box coverage to detect anomalously large masks.
    * Produce a single, JSON-serialisable report dictionary.

    Usage
    -----
    ::

        evaluator = SAMSegmentationEvaluator(config=sam_config)
        report    = evaluator.evaluate(
            masks            = filtered_masks,   # list[SAMMask | dict]
            image_shape      = image.shape,
            structural_mask  = gt_binary_mask,   # optional
        )
        evaluator.save_report_json(report, "output/sam_report.json")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Parameters
        ----------
        config : dict, optional
            Project-wide or SAM-specific config dict.  All thresholds fall
            back to safe defaults when the key is absent.
        """
        self.config: Dict[str, Any] = config or {}

    # ==================================================================
    # PUBLIC ENTRY POINT
    # ==================================================================

    def evaluate(
        self,
        masks: List[MaskInput],
        image_shape: Tuple[int, ...],
        structural_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Primary public API.  Runs the full evaluation pipeline and returns
        a single merged report dictionary.

        Pipeline
        --------
        evaluate()
          -> compute_coverage()
          -> compute_fragmentation()
          -> compute_mask_statistics()
          -> compute_quality_statistics()
          -> compute_structural_candidates()
          -> compute_bbox_coverage()
          -> generate_report()

        Parameters
        ----------
        masks : list[SAMMask | dict]
            Already-filtered SAM masks (output of SAMMaskFiller).
        image_shape : tuple
            Shape of the source image: (H, W) or (H, W, C).
        structural_mask : np.ndarray, optional
            Binary uint8 mask (0/255) representing the known carton geometry.
            When supplied, coverage is measured against it; otherwise None
            is returned for coverage.

        Returns
        -------
        dict
            Full evaluation report — see generate_report() for schema.
        """
        h, w = image_shape[:2]

        coverage     = self.compute_coverage(masks, structural_mask)
        fragmentation = self.compute_fragmentation(masks)
        mask_stats   = self.compute_mask_statistics(masks)
        quality_stats = self.compute_quality_statistics(masks)
        candidates   = self.compute_structural_candidates(masks)
        bbox_cov     = self.compute_bbox_coverage(masks, (h, w))

        return self.generate_report(
            coverage      = coverage,
            fragmentation = fragmentation,
            mask_stats    = mask_stats,
            quality_stats = quality_stats,
            candidates    = candidates,
            bbox_coverage = bbox_cov,
            masks         = masks,
            image_shape   = (h, w),
        )

    # ==================================================================
    # 1. COVERAGE
    # ==================================================================

    def compute_coverage(
        self,
        masks: List[MaskInput],
        structural_mask: Optional[np.ndarray],
    ) -> Optional[float]:
        """
        Measure how much of the known structural mask is covered by SAM masks.

        This answers: *"Is SAM finding the carton at all?"*

        Process
        -------
        1. Compute the pixel-wise union of all SAM binary masks.
        2. Intersect the union with ``structural_mask``.
        3. Return ``intersection_area / structural_mask_area``.

        Parameters
        ----------
        masks : list[SAMMask | dict]
        structural_mask : np.ndarray | None
            Binary uint8 mask (0/255).  If None the method returns None.

        Returns
        -------
        float  in [0, 1], or None when no structural_mask is provided.
        """
        if structural_mask is None:
            return None

        if not masks:
            return 0.0

        gt_area = int(cv2.countNonZero(structural_mask))
        if gt_area == 0:
            return 0.0

        h, w = structural_mask.shape[:2]
        union = self._build_union_mask(masks, h, w)

        intersection = cv2.bitwise_and(union, structural_mask)
        inter_area   = int(cv2.countNonZero(intersection))

        return float(inter_area) / gt_area

    # ==================================================================
    # 2. FRAGMENTATION
    # ==================================================================

    def compute_fragmentation(
        self,
        masks: List[MaskInput],
    ) -> Dict[str, Any]:
        """
        Measure over-segmentation tendency.

        Metrics
        -------
        mask_count : int
            Raw number of masks after filtering.
        total_area : int
            Sum of all mask pixel areas.
        average_area : float
            Mean mask pixel area.
        fragmentation_score : float  in [0, 1]
            Composite score.  *Lower is better.*

            Score is derived from:
              * normalised mask count  (many masks → higher score)
              * normalised average area reciprocal  (tiny masks → higher score)

            Both components are clipped and blended equally.

        is_over_fragmented : bool
            True when mask_count exceeds ``over_frag_mask_count`` (default 40)
            AND average_area falls below ``over_frag_avg_area`` (default 15 000).
        is_under_segmented : bool
            True when mask_count is below ``under_seg_mask_count`` (default 2).

        Parameters
        ----------
        masks : list[SAMMask | dict]

        Returns
        -------
        dict
        """
        if not masks:
            return {
                "mask_count"         : 0,
                "total_area"         : 0,
                "average_area"       : 0.0,
                "fragmentation_score": 1.0,   # maximally fragmented (empty)
                "is_over_fragmented" : False,
                "is_under_segmented" : True,
            }

        areas = np.array([self._get_area(m) for m in masks], dtype=float)
        n        = len(masks)
        total    = int(areas.sum())
        avg_area = float(areas.mean())

        # Fragmentation score ∈ [0, 1]
        ref_count    = float(self.config.get("frag_ref_mask_count", 20))
        ref_avg_area = float(self.config.get("frag_ref_avg_area",   40_000.0))

        count_component    = min(n / ref_count, 1.0)
        area_component     = 1.0 - min(avg_area / ref_avg_area, 1.0)
        fragmentation_score = 0.5 * count_component + 0.5 * area_component

        over_frag_count    = self.config.get("over_frag_mask_count", 40)
        over_frag_avg_area = self.config.get("over_frag_avg_area",   15_000)
        under_seg_count    = self.config.get("under_seg_mask_count",  2)

        return {
            "mask_count"         : n,
            "total_area"         : total,
            "average_area"       : avg_area,
            "fragmentation_score": round(fragmentation_score, 4),
            "is_over_fragmented" : (n > over_frag_count) and (avg_area < over_frag_avg_area),
            "is_under_segmented" : n < under_seg_count,
        }

    # ==================================================================
    # 3. MASK STATISTICS
    # ==================================================================

    def compute_mask_statistics(
        self,
        masks: List[MaskInput],
    ) -> Dict[str, Any]:
        """
        Analyse the pixel-area distribution of the mask set.

        Returns
        -------
        dict
            count, total_area, largest_area, smallest_area, average_area,
            median_area, std_area, p25_area, p75_area.
            All values are 0 / 0.0 when *masks* is empty.

        Also includes
        -------------
        area_histogram : dict
            {"bin_edges": [...], "counts": [...]}  — 10-bin histogram data
            suitable for plotting without importing matplotlib here.
        """
        if not masks:
            return {
                "count"        : 0,
                "total_area"   : 0,
                "largest_area" : 0.0,
                "smallest_area": 0.0,
                "average_area" : 0.0,
                "median_area"  : 0.0,
                "std_area"     : 0.0,
                "p25_area"     : 0.0,
                "p75_area"     : 0.0,
                "area_histogram": {"bin_edges": [], "counts": []},
            }

        areas = np.array([self._get_area(m) for m in masks], dtype=float)

        counts, bin_edges = np.histogram(areas, bins=10)

        return {
            "count"        : int(len(areas)),
            "total_area"   : int(areas.sum()),
            "largest_area" : float(areas.max()),
            "smallest_area": float(areas.min()),
            "average_area" : float(areas.mean()),
            "median_area"  : float(np.median(areas)),
            "std_area"     : float(areas.std()),
            "p25_area"     : float(np.percentile(areas, 25)),
            "p75_area"     : float(np.percentile(areas, 75)),
            "area_histogram": {
                "bin_edges": [round(float(v), 1) for v in bin_edges],
                "counts"   : [int(c) for c in counts],
            },
        }

    # ==================================================================
    # 4. QUALITY STATISTICS
    # ==================================================================

    def compute_quality_statistics(
        self,
        masks: List[MaskInput],
    ) -> Dict[str, Any]:
        """
        Evaluate mask confidence using SAM's own quality signals.

        Signals used
        ------------
        * ``predicted_iou``   — SAM's own estimate of mask quality.
        * ``stability_score`` — consistency under threshold perturbation.
        * ``quality_score``   — geometric mean: √(iou × stability).

        For each signal the method computes: average, minimum, maximum,
        standard deviation.

        Returns
        -------
        dict
            Nested sub-dicts for each signal plus a top-level
            ``quality_histogram`` with bin data.
            All values are 0.0 when *masks* is empty.
        """
        if not masks:
            _zero = {"average": 0.0, "minimum": 0.0, "maximum": 0.0, "std": 0.0}
            return {
                "count"            : 0,
                "predicted_iou"    : _zero,
                "stability_score"  : _zero,
                "quality_score"    : _zero,
                "quality_histogram": {"bin_edges": [], "counts": []},
            }

        ious     = np.array([self._get_predicted_iou(m)    for m in masks], dtype=float)
        stabs    = np.array([self._get_stability_score(m)  for m in masks], dtype=float)
        qualities = np.sqrt(np.clip(ious * stabs, 0.0, 1.0))

        def _stats(arr: np.ndarray) -> Dict[str, float]:
            return {
                "average": round(float(arr.mean()), 6),
                "minimum": round(float(arr.min()),  6),
                "maximum": round(float(arr.max()),  6),
                "std"    : round(float(arr.std()),  6),
            }

        counts, bin_edges = np.histogram(qualities, bins=10, range=(0.0, 1.0))

        return {
            "count"           : len(masks),
            "predicted_iou"   : _stats(ious),
            "stability_score" : _stats(stabs),
            "quality_score"   : _stats(qualities),
            "quality_histogram": {
                "bin_edges": [round(float(v), 3) for v in bin_edges],
                "counts"   : [int(c) for c in counts],
            },
        }

    # ==================================================================
    # 5. STRUCTURAL CANDIDATES
    # ==================================================================

    def compute_structural_candidates(
        self,
        masks: List[MaskInput],
    ) -> Dict[str, Any]:
        """
        Estimate how many SAM masks likely correspond to meaningful carton
        structural parts (Panels, Flaps, Tabs, etc.).

        Classification criteria (all configurable)
        ------------------------------------------
        Panels
            Large, roughly rectangular regions.
            area ≥ ``panel_area_min``  AND  rectangularity ≥ ``panel_min_rect``

        Flaps
            Medium area, moderately elongated.
            ``flap_area_min`` ≤ area < ``panel_area_min``

        Tabs
            Small area, often elongated or rounded.
            area < ``flap_area_min``

        In addition, any mask whose quality_score < ``min_candidate_quality``
        is always classified as noise regardless of area.

        Parameters
        ----------
        masks : list[SAMMask | dict]

        Returns
        -------
        dict
            candidate_count, panel_count, flap_count, tab_count, noise_count,
            candidates (list of lightweight dicts).
        """
        if not masks:
            return {
                "candidate_count": 0,
                "panel_count"    : 0,
                "flap_count"     : 0,
                "tab_count"      : 0,
                "noise_count"    : 0,
                "candidates"     : [],
            }

        panel_area_min      = self.config.get("panel_area_min",      60_000)
        flap_area_min       = self.config.get("flap_area_min",        8_000)
        panel_min_rect      = self.config.get("panel_min_rect",        0.50)
        min_candidate_qual  = self.config.get("min_candidate_quality", 0.75)

        panels, flaps, tabs, noise = [], [], [], []

        for m in masks:
            area    = self._get_area(m)
            qs      = self._get_quality_score(m)
            bbox    = self._get_bbox(m)
            bw, bh  = bbox.get("w", 1), bbox.get("h", 1)
            ar      = float(bw) / bh if bh > 0 else 1.0
            rect    = min(ar, 1.0 / ar) if ar > 0 else 0.0   # ∈ (0, 1]; 1 = square

            entry = {
                "id"            : self._get_id(m),
                "area"          : area,
                "quality_score" : round(qs, 4),
                "aspect_ratio"  : round(ar, 3),
                "rectangularity": round(rect, 3),
            }

            if qs < min_candidate_qual:
                entry["role"] = "noise"
                noise.append(entry)
                continue

            if area >= panel_area_min and rect >= panel_min_rect:
                entry["role"] = "panel"
                panels.append(entry)
            elif area >= flap_area_min:
                entry["role"] = "flap"
                flaps.append(entry)
            else:
                entry["role"] = "tab"
                tabs.append(entry)

        candidate_count = len(panels) + len(flaps) + len(tabs)

        return {
            "candidate_count": candidate_count,
            "panel_count"    : len(panels),
            "flap_count"     : len(flaps),
            "tab_count"      : len(tabs),
            "noise_count"    : len(noise),
            "candidates"     : panels + flaps + tabs,
        }

    # ==================================================================
    # 6. BBOX COVERAGE
    # ==================================================================

    def compute_bbox_coverage(
        self,
        masks: List[MaskInput],
        image_size: Tuple[int, int],
    ) -> float:
        """
        Determine how much image space is occupied by mask bounding boxes.

        Formula:  sum(bbox_w × bbox_h)  /  (image_H × image_W)

        Useful for detecting anomalously large or overlapping masks that
        together claim more than 100 % of the image (because bboxes overlap).

        Parameters
        ----------
        masks : list[SAMMask | dict]
        image_size : tuple  (H, W)

        Returns
        -------
        float  ≥ 0.  Values > 1.0 indicate significant bbox overlap.
        """
        if not masks:
            return 0.0

        h, w = image_size
        image_area = h * w
        if image_area == 0:
            return 0.0

        total_bbox_area = 0
        for m in masks:
            bbox = self._get_bbox(m)
            total_bbox_area += bbox.get("w", 0) * bbox.get("h", 0)

        return float(total_bbox_area) / image_area

    # ==================================================================
    # 7. REPORT
    # ==================================================================

    def generate_report(
        self,
        coverage     : Optional[float],
        fragmentation: Dict[str, Any],
        mask_stats   : Dict[str, Any],
        quality_stats: Dict[str, Any],
        candidates   : Dict[str, Any],
        bbox_coverage: float,
        masks        : List[MaskInput],
        image_shape  : Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Combine all computed metrics into a single JSON-serialisable report.

        Also derives:
        * ``status`` — overall assessment string
        * ``best_mask``  — mask with highest quality_score
        * ``worst_mask`` — mask with lowest quality_score
        * ``mask_count_summary`` — friendly breakdown

        Report Schema
        -------------
        {
            "status"           : str,
            "coverage"         : float | None,
            "fragmentation"    : { mask_count, total_area, average_area,
                                   fragmentation_score, is_over_fragmented,
                                   is_under_segmented },
            "mask_statistics"  : { count, largest_area, smallest_area,
                                   average_area, median_area, std_area,
                                   p25_area, p75_area, area_histogram },
            "quality_statistics": { count, predicted_iou, stability_score,
                                    quality_score, quality_histogram },
            "structural_candidates": { candidate_count, panel_count,
                                       flap_count, tab_count, noise_count,
                                       candidates },
            "bbox_coverage"    : float,
            "mask_count_summary": { total, candidates, noise, panels,
                                    flaps, tabs },
            "best_mask"        : dict | None,
            "worst_mask"       : dict | None,
        }

        Parameters
        ----------
        All parameters are the outputs of the individual compute_* methods
        plus the raw ``masks`` and ``image_shape`` for extras.

        Returns
        -------
        dict
        """
        status = self._derive_status(
            coverage      = coverage,
            fragmentation = fragmentation,
            quality_stats = quality_stats,
            candidates    = candidates,
        )

        best_mask  = self._identify_best_mask(masks)
        worst_mask = self._identify_worst_mask(masks)

        mask_count_summary = {
            "total"     : fragmentation.get("mask_count", 0),
            "candidates": candidates.get("candidate_count", 0),
            "noise"     : candidates.get("noise_count", 0),
            "panels"    : candidates.get("panel_count", 0),
            "flaps"     : candidates.get("flap_count", 0),
            "tabs"      : candidates.get("tab_count", 0),
        }

        return {
            "status"               : status,
            "coverage"             : coverage,
            "fragmentation"        : fragmentation,
            "mask_statistics"      : mask_stats,
            "quality_statistics"   : quality_stats,
            "structural_candidates": candidates,
            "bbox_coverage"        : round(bbox_coverage, 4),
            "mask_count_summary"   : mask_count_summary,
            "best_mask"            : best_mask,
            "worst_mask"           : worst_mask,
        }

    # ==================================================================
    # STATUS DERIVATION
    # ==================================================================

    def _derive_status(
        self,
        coverage     : Optional[float],
        fragmentation: Dict[str, Any],
        quality_stats: Dict[str, Any],
        candidates   : Dict[str, Any],
    ) -> str:
        """
        Derive a single human-readable status string from the metrics.

        Priority order (first match wins)
        ----------------------------------
        UNDER_SEGMENTED       — too few masks
        OVER_FRAGMENTED       — too many tiny masks
        LOW_QUALITY           — average quality score is poor
        LOW_COVERAGE          — masks don't cover the structural region
        NO_CANDIDATES         — no masks pass the structural candidate gate
        OK                    — everything looks healthy
        """
        if fragmentation.get("is_under_segmented"):
            return "UNDER_SEGMENTED"

        if fragmentation.get("is_over_fragmented"):
            return "OVER_FRAGMENTED"

        low_q_threshold = self.config.get("low_quality_threshold", 0.70)
        avg_quality = quality_stats.get("quality_score", {}).get("average", 1.0)
        if avg_quality < low_q_threshold:
            return "LOW_QUALITY"

        low_cov_threshold = self.config.get("min_coverage_threshold", 0.10)
        if coverage is not None and coverage < low_cov_threshold:
            return "LOW_COVERAGE"

        if candidates.get("candidate_count", 0) == 0:
            return "NO_CANDIDATES"

        return "OK"

    # ==================================================================
    # BEST / WORST MASK IDENTIFICATION
    # ==================================================================

    def _identify_best_mask(
        self,
        masks: List[MaskInput],
    ) -> Optional[Dict[str, Any]]:
        """
        Return a lightweight summary dict for the mask with the highest
        quality_score.  Returns None if *masks* is empty.
        """
        if not masks:
            return None

        best = max(masks, key=self._get_quality_score)
        return {
            "id"             : self._get_id(best),
            "area"           : self._get_area(best),
            "quality_score"  : round(self._get_quality_score(best), 4),
            "predicted_iou"  : round(self._get_predicted_iou(best), 4),
            "stability_score": round(self._get_stability_score(best), 4),
        }

    def _identify_worst_mask(
        self,
        masks: List[MaskInput],
    ) -> Optional[Dict[str, Any]]:
        """
        Return a lightweight summary dict for the mask with the lowest
        quality_score.  Returns None if *masks* is empty.
        """
        if not masks:
            return None

        worst = min(masks, key=self._get_quality_score)
        return {
            "id"             : self._get_id(worst),
            "area"           : self._get_area(worst),
            "quality_score"  : round(self._get_quality_score(worst), 4),
            "predicted_iou"  : round(self._get_predicted_iou(worst), 4),
            "stability_score": round(self._get_stability_score(worst), 4),
        }

    # ==================================================================
    # PERSISTENCE
    # ==================================================================

    def save_report_json(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Serialise the evaluation report to a JSON file.

        Parameters
        ----------
        report : dict
            Output of evaluate() or generate_report().
        output_path : str
            Destination file path.
        """
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4, default=self._json_default)

    # ==================================================================
    # DEBUG VISUALIZATION
    # ==================================================================

    def create_evaluation_overlay(
        self,
        image: np.ndarray,
        report: Dict[str, Any],
    ) -> np.ndarray:
        """
        Render a compact evaluation summary on top of *image*.

        Draws status, coverage, quality, and candidate breakdown as text
        annotations in the top-left corner.

        Parameters
        ----------
        image : np.ndarray  BGR source image
        report : dict       Output of evaluate() / generate_report().

        Returns
        -------
        np.ndarray  BGR image with overlay annotations.
        """
        vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        status = report.get("status", "?")
        cand   = report.get("structural_candidates", {})
        frag   = report.get("fragmentation", {})
        qual   = report.get("quality_statistics", {})
        cov    = report.get("coverage")

        ok_color  = (40, 200, 60)
        err_color = (30, 30, 230)
        status_ok = (status == "OK")

        lines = [
            (f"Status: {status}",                                    ok_color if status_ok else err_color),
            (f"Masks:  {frag.get('mask_count', 0)}",                 ok_color),
            (f"Coverage: {cov:.2f}" if cov is not None else "Coverage: N/A", ok_color),
            (f"AvgIoU:  {qual.get('predicted_iou', {}).get('average', 0):.2f}", ok_color),
            (f"AvgStab: {qual.get('stability_score', {}).get('average', 0):.2f}", ok_color),
            (f"AvgQual: {qual.get('quality_score', {}).get('average', 0):.2f}", ok_color),
            (f"P={cand.get('panel_count',0)}  F={cand.get('flap_count',0)}  T={cand.get('tab_count',0)}  N={cand.get('noise_count',0)}", ok_color),
        ]

        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 1
        y     = 28

        for text, color in lines:
            cv2.putText(vis, text, (7, y), font, scale, (0, 0, 0), thick + 2)
            cv2.putText(vis, text, (7, y), font, scale, color,     thick)
            y += 24

        return vis

    # ==================================================================
    # PRIVATE HELPERS — mask field accessors
    # ==================================================================
    # All methods accept both SAMMask objects and project region dicts.

    def _get_binary_mask(self, m: MaskInput) -> Optional[np.ndarray]:
        """Return the binary (uint8) mask array from either input type."""
        if hasattr(m, "segmentation"):
            return m.segmentation              # SAMMask
        return m.get("binary_mask") or m.get("segmentation")

    def _get_area(self, m: MaskInput) -> int:
        """Return the pixel area from either input type."""
        if hasattr(m, "area"):
            return int(m.area)
        return int(m.get("area", 0))

    def _get_id(self, m: MaskInput) -> Any:
        """Return the mask identifier."""
        if hasattr(m, "mask_id"):
            return m.mask_id
        return m.get("id", m.get("mask_id"))

    def _get_bbox(self, m: MaskInput) -> Dict[str, int]:
        """Return the bounding box dict {x, y, w, h}."""
        if hasattr(m, "bbox"):
            return m.bbox or {}
        return m.get("bbox", {})

    def _get_predicted_iou(self, m: MaskInput) -> float:
        """Return SAM's predicted IoU score."""
        if hasattr(m, "predicted_iou"):
            return float(m.predicted_iou)
        return float(m.get("iou", m.get("predicted_iou", 1.0)))

    def _get_stability_score(self, m: MaskInput) -> float:
        """Return SAM's stability score."""
        if hasattr(m, "stability_score"):
            return float(m.stability_score)
        return float(m.get("stability", m.get("stability_score", 1.0)))

    def _get_quality_score(self, m: MaskInput) -> float:
        """Return the combined quality score (geometric mean of iou × stability)."""
        if hasattr(m, "quality_score"):
            return float(m.quality_score)
        iou  = self._get_predicted_iou(m)
        stab = self._get_stability_score(m)
        return math.sqrt(max(iou * stab, 0.0))

    # ==================================================================
    # PRIVATE HELPERS — pixel operations
    # ==================================================================

    def _build_union_mask(
        self,
        masks  : List[MaskInput],
        h      : int,
        w      : int,
    ) -> np.ndarray:
        """
        Pixel-wise OR of all binary masks.

        Parameters
        ----------
        masks : list[SAMMask | dict]
        h, w  : int  target canvas dimensions

        Returns
        -------
        np.ndarray  uint8 (H, W)
        """
        union = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            bm = self._get_binary_mask(m)
            if bm is None:
                continue
            if bm.shape[:2] != (h, w):
                bm = cv2.resize(bm, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.bitwise_or(union, bm, dst=union)
        return union

    # ==================================================================
    # PRIVATE HELPERS — serialisation
    # ==================================================================

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """JSON encoder fallback for numpy scalars / arrays."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bool):
            return bool(obj)
        return str(obj)