import cv2
import numpy as np


class SAMOverlayGenerator:
    """
    Generates visual overlays specific to SAM segmentation results.

    Draws each SAM region as a semi-transparent coloured mask and
    annotates it with its predicted-IoU and stability scores —
    information that has no equivalent in the structural pipeline.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    # ------------------------------------------------------------------
    # PRIMARY ENTRY POINT
    # ------------------------------------------------------------------

    def generate(self, image: np.ndarray, regions: list) -> np.ndarray:
        """
        Render all SAM regions onto *image*.

        Parameters
        ----------
        image : np.ndarray
            BGR source image (H, W, 3).
        regions : list[dict]
            Project-format region dicts (output of SAMMaskFiller.filter_masks).

        Returns
        -------
        np.ndarray
            BGR overlay image.
        """
        if image is None:
            return None

        overlay = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for region in regions:
            color = self._region_color(region.get("id", 0))
            self._draw_mask(overlay, region, color)
            self._draw_bbox(overlay, region, color)
            self._draw_centroid(overlay, region)
            self._draw_label(overlay, region, color)

        return overlay

    # ------------------------------------------------------------------
    # DRAWING PRIMITIVES
    # ------------------------------------------------------------------

    def _draw_mask(self, canvas: np.ndarray, region: dict, color: tuple):
        """
        Paint the mask pixels with a semi-transparent fill.
        """
        mask = region.get("binary_mask")
        if mask is None:
            return

        alpha = self.config.get("mask_alpha", 0.35)
        colored = np.zeros_like(canvas)
        colored[mask > 0] = color
        cv2.addWeighted(canvas, 1.0, colored, alpha, 0, dst=canvas)

        # Thin border
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 1)

    def _draw_bbox(self, canvas: np.ndarray, region: dict, color: tuple):
        """
        Draw the bounding box as a thin rectangle.
        """
        bbox = region.get("bbox", {})
        x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0)
        if w > 0 and h > 0:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)

    def _draw_centroid(self, canvas: np.ndarray, region: dict):
        """
        Draw a small dot at the region centroid.
        """
        centroid = region.get("centroid")
        if centroid is None:
            return
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(canvas, (cx, cy), 3, (0, 255, 255), -1)

    def _draw_label(self, canvas: np.ndarray, region: dict, color: tuple):
        """
        Draw region ID + SAM scores near the centroid.
        """
        centroid = region.get("centroid")
        if centroid is None:
            return

        cx, cy = int(centroid[0]), int(centroid[1])
        rid     = region.get("id", "?")
        iou     = region.get("iou", 0.0)
        stab    = region.get("stability", 0.0)

        lines = [
            f"#{rid}",
            f"iou={iou:.2f}",
            f"stb={stab:.2f}",
        ]

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.get("label_font_scale", 0.38)
        thickness  = 1
        line_h     = 13

        for i, line in enumerate(lines):
            y_pos = cy - (len(lines) - 1 - i) * line_h
            # Shadow
            cv2.putText(canvas, line, (cx + 1, y_pos + 1), font, font_scale, (0, 0, 0), thickness + 1)
            # Text
            cv2.putText(canvas, line, (cx, y_pos), font, font_scale, (255, 255, 255), thickness)

    # ------------------------------------------------------------------
    # SCORE HEATMAP
    # ------------------------------------------------------------------

    def generate_iou_heatmap(self, image_shape: tuple, regions: list) -> np.ndarray:
        """
        Create a grayscale heatmap where pixel brightness represents
        the predicted IoU of the SAM mask at that location.

        Parameters
        ----------
        image_shape : tuple  (H, W) or (H, W, C)
        regions : list[dict]

        Returns
        -------
        np.ndarray  BGR heatmap image
        """
        h, w = image_shape[:2]
        heat = np.zeros((h, w), dtype=np.float32)

        for region in regions:
            mask = region.get("binary_mask")
            iou  = region.get("iou", 0.0)
            if mask is None:
                continue
            heat[mask > 0] = np.maximum(heat[mask > 0], iou)

        heat_u8 = (heat * 255).clip(0, 255).astype(np.uint8)
        return cv2.applyColorMap(heat_u8, cv2.COLORMAP_PLASMA)

    def generate_stability_heatmap(self, image_shape: tuple, regions: list) -> np.ndarray:
        """
        Same as generate_iou_heatmap but using stability scores.
        """
        h, w = image_shape[:2]
        heat = np.zeros((h, w), dtype=np.float32)

        for region in regions:
            mask = region.get("binary_mask")
            stab = region.get("stability", 0.0)
            if mask is None:
                continue
            heat[mask > 0] = np.maximum(heat[mask > 0], stab)

        heat_u8 = (heat * 255).clip(0, 255).astype(np.uint8)
        return cv2.applyColorMap(heat_u8, cv2.COLORMAP_VIRIDIS)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _region_color(self, region_id: int) -> tuple:
        """
        Deterministic per-region BGR colour derived from the region ID.
        """
        rng = np.random.default_rng(int(region_id) + 1)
        b, g, r = int(rng.integers(60, 230)), int(rng.integers(60, 230)), int(rng.integers(60, 230))
        return (b, g, r)
