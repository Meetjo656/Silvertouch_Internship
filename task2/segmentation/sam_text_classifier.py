import cv2
import numpy as np


class SAMTextClassifier:
    """
    Classifies SAM-produced binary masks as 'text', 'structure', or 'unknown'.

    Public API
    ----------
    compute_features(mask)   -> dict | None
    classify_mask(mask)      -> dict | None
    classify_masks(masks)    -> list[dict | None]
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_mask(mask: np.ndarray) -> None:
        """Raise if *mask* is not a 2-D binary numpy array."""
        if not isinstance(mask, np.ndarray):
            raise TypeError(
                f"Expected mask to be numpy.ndarray but got {type(mask)}"
            )
        if mask.ndim != 2:
            raise ValueError(
                f"Expected mask to be 2-D, but got {mask.ndim} dimensions"
            )
        unique_values = np.unique(mask)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError("Mask must contain only 0 and 1 values")

    # ------------------------------------------------------------------
    # Public method 1 – feature extraction (no classification)
    # ------------------------------------------------------------------

    def compute_features(self, mask: np.ndarray) -> dict | None:
        """
        Extract shape and coverage features from a single binary mask.

        Returns
        -------
        dict with keys:
            mask_area       – foreground pixel count
            contour_area    – largest contour area (cv2.contourArea)
            fill_ratio      – mask_area / bounding-box area
            solidity        – contour_area / convex-hull area
            stability_score – mask_area / total image pixels
            predicted_iou   – IoU proxy: min(mask_area, contour_area) / union
            eccentricity    – fitEllipse axis ratio (requires ≥ 5 contour pts)
            thinness_ratio  – short / long side of min-area rect
            elongation      – long / short side of min-area rect
        """
        try:
            self._validate_mask(mask)

            # ── mask_area ────────────────────────────────────────────
            foreground_pixels = int(np.sum(mask))
            mask_area = foreground_pixels

            # ── fill_ratio ───────────────────────────────────────────
            coords = np.argwhere(mask)
            if coords.size > 0:
                r_min, c_min = coords.min(axis=0)
                r_max, c_max = coords.max(axis=0)
                bbox_area = max((r_max - r_min + 1) * (c_max - c_min + 1), 1)
                fill_ratio = float(foreground_pixels) / bbox_area
            else:
                fill_ratio = 0.0

            # ── contour-based features ────────────────────────────────
            mask_u8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                cnt = max(contours, key=cv2.contourArea)

                # contour_area (distinct from mask_area)
                contour_area = float(cv2.contourArea(cnt))

                # solidity
                hull = cv2.convexHull(cnt)
                hull_area = float(cv2.contourArea(hull))
                solidity = contour_area / hull_area if hull_area > 0 else 0.0

                # eccentricity – fitEllipse requires ≥ 5 points
                if len(cnt) >= 5:
                    _, (a, b), _ = cv2.fitEllipse(cnt)
                    eccentricity = float(a) / b if b > 0 else 0.0
                else:
                    eccentricity = 0.0

                # thinness_ratio & elongation from min-area rect
                _, (w, h), _ = cv2.minAreaRect(cnt)
                thinness_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
                elongation     = max(w, h) / min(w, h) if min(w, h) > 0 else 0.0
            else:
                contour_area  = 0.0
                solidity      = 0.0
                eccentricity  = 0.0
                thinness_ratio = 0.0
                elongation    = 0.0

            # ── stability_score ───────────────────────────────────────
            stability_score = float(foreground_pixels) / mask.size

            # ── predicted_iou ─────────────────────────────────────────
            fp = float(foreground_pixels)
            ca = contour_area
            intersection = min(fp, ca)
            union = fp + ca - intersection
            predicted_iou = intersection / union if union > 0 else 0.0

            return {
                "mask_area":       mask_area,
                "contour_area":    contour_area,
                "fill_ratio":      fill_ratio,
                "solidity":        solidity,
                "stability_score": stability_score,
                "predicted_iou":   predicted_iou,
                "eccentricity":    eccentricity,
                "thinness_ratio":  thinness_ratio,
                "elongation":      elongation,
            }

        except Exception as e:
            print(f"[compute_features] {e}")
            return None

    # ------------------------------------------------------------------
    # Public method 2 – classify a single mask
    # ------------------------------------------------------------------

    def classify_mask(self, mask: np.ndarray) -> dict | None:
        """
        Classify one binary mask.

        Returns
        -------
        {"label": "text" | "structure" | "unknown", "features": {...}}
        or None on error.
        """
        try:
            features = self.compute_features(mask)
            if features is None:
                return None

            mask_area  = features["mask_area"]
            fill_ratio = features["fill_ratio"]

            # Text: small, sparse, fragmented
            if mask_area < 3000 and fill_ratio < 0.45:
                label = "text"

            # Structure: large, dense
            elif mask_area > 5000 and fill_ratio > 0.50:
                label = "structure"

            else:
                label = "unknown"

            return {"label": label, "features": features}

        except Exception as e:
            print(f"[classify_mask] {e}")
            return None

    # ------------------------------------------------------------------
    # Public method 3 – classify a batch of masks
    # ------------------------------------------------------------------

    def classify_masks(self, masks: list) -> list | None:
        """
        Classify a list of binary masks.

        Returns
        -------
        list of classify_mask() results (one entry per mask), or None on error.
        """
        if not isinstance(masks, list):
            raise TypeError(
                f"Expected masks to be list but got {type(masks)}"
            )
        try:
            return [self.classify_mask(mask) for mask in masks]
        except Exception as e:
            print(f"[classify_masks] {e}")
            return None
