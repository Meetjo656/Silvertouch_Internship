"""
mask_reconstructor.py
---------------------
Converts a list of kept structural masks into a single binary
text-free closed mask:

    Kept Masks
          ↓
    Merge (draw_masks / merge_masks)
          ↓
    Gap Closing   (close_gaps)
          ↓
    Hole Filling  (fill_holes)
          ↓
    Noise Removal (clean_noise)
          ↓
    Validation    (validate_mask)
          ↓
    new_closed_mask  →  np.ndarray (uint8, values 0/1)
"""

import cv2
import numpy as np


class MaskReconstructor:
    """
    Merges structural masks and applies morphological repair to
    produce a text-free closed mask ready for downstream processing:

        Connected Components → Polygon Extraction → Region Segmentation

    Public API
    ----------
    draw_masks(kept_masks, image_shape)   -> np.ndarray
    merge_masks(masks, image_shape)       -> np.ndarray
    close_gaps(mask)                      -> np.ndarray
    fill_holes(mask)                      -> np.ndarray
    clean_noise(mask)                     -> np.ndarray
    validate_mask(mask)                   -> bool
    reconstruct(kept_masks, image_shape)  -> np.ndarray
    """

    # Minimum contiguous foreground area kept by clean_noise().
    MIN_COMPONENT_AREA: int = 100

    # Structuring element size used by close_gaps().
    CLOSE_KERNEL_SIZE: tuple[int, int] = (5, 5)

    # ------------------------------------------------------------------
    # Method 1 – draw_masks
    # ------------------------------------------------------------------

    def draw_masks(
        self,
        kept_masks: list[dict],
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Paint every kept mask onto a blank canvas.

        Each entry in *kept_masks* must contain a ``"mask"`` key whose
        value is a 2-D numpy array with foreground pixels > 0.

        Parameters
        ----------
        kept_masks : list[dict]
            Output of SAMTextSuppressor.get_kept_masks() —
            each element is ``{"label": ..., "features": ..., "mask": ndarray}``.
        image_shape : tuple[int, int]
            ``(height, width)`` of the source image.

        Returns
        -------
        np.ndarray
            uint8 binary canvas; 1 = structural foreground, 0 = background.
        """
        canvas = np.zeros(image_shape[:2], dtype=np.uint8)

        for item in kept_masks:
            if item is None:
                continue
            mask = item.get("mask")
            if mask is None or not isinstance(mask, np.ndarray):
                continue
            canvas[mask > 0] = 1

        return canvas

    # ------------------------------------------------------------------
    # Method 2 – merge_masks
    # ------------------------------------------------------------------

    def merge_masks(
        self,
        masks: list[np.ndarray],
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Combine a list of raw 2-D numpy arrays into one binary mask.

        Use this when masks are plain arrays rather than dicts.
        For dict-wrapped masks use draw_masks() instead.

        Parameters
        ----------
        masks : list[np.ndarray]
            Raw binary arrays; foreground pixels must be > 0.
        image_shape : tuple[int, int]
            ``(height, width)`` of the canvas to create.

        Returns
        -------
        np.ndarray
            uint8 binary mask (values 0 / 1).
        """
        merged = np.zeros(image_shape[:2], dtype=np.uint8)

        for mask in masks:
            if mask is None or not isinstance(mask, np.ndarray):
                continue
            merged[mask > 0] = 1

        return merged

    # ------------------------------------------------------------------
    # Method 3 – close_gaps
    # ------------------------------------------------------------------

    def close_gaps(self, mask: np.ndarray) -> np.ndarray:
        """
        Morphological closing to repair broken boundaries left by
        text removal.

        Dilation followed by erosion bridges small gaps between
        structural regions so downstream polygon extraction works
        on continuous contours.

        Parameters
        ----------
        mask : np.ndarray
            uint8 binary mask (values 0 / 1).

        Returns
        -------
        np.ndarray
            Closed mask (same dtype and shape).
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, self.CLOSE_KERNEL_SIZE
        )
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return closed

    # ------------------------------------------------------------------
    # Method 4 – fill_holes
    # ------------------------------------------------------------------

    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill internal voids (holes) created by text removal.

        Algorithm
        ---------
        1. Flood-fill from the image border on an inverted copy.
           This isolates background pixels that are truly "outside"
           the structure.
        2. Invert the flood-filled image → internal holes become
           foreground.
        3. OR with the original mask so the holes are closed.

        Parameters
        ----------
        mask : np.ndarray
            uint8 binary mask (values 0 / 1).

        Returns
        -------
        np.ndarray
            Hole-filled mask (values 0 / 1, uint8).
        """
        # Work on a 0/255 copy so OpenCV flood-fill operates correctly.
        mask_255 = (mask * 255).astype(np.uint8)

        # Flood-fill canvas must be 2 pixels larger in each dimension.
        h, w = mask_255.shape
        flood_canvas = mask_255.copy()
        fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Fill from top-left corner with white (255); lo/hi diff = 10
        # so we only flood through dark (background) pixels.
        cv2.floodFill(flood_canvas, fill_mask, seedPoint=(0, 0), newVal=255,
                      loDiff=10, upDiff=10)

        # Invert: pixels that were NOT reached by flood-fill are holes.
        flood_inv = cv2.bitwise_not(flood_canvas)

        # Combine original mask with recovered holes.
        filled_255 = cv2.bitwise_or(mask_255, flood_inv)

        return (filled_255 > 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Method 5 – clean_noise
    # ------------------------------------------------------------------

    def clean_noise(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove tiny isolated fragments via connected-component analysis.

        Any component whose pixel area is below MIN_COMPONENT_AREA is
        erased. This strips leftover specks that survive morphological
        closing without damaging meaningful structural regions.

        Parameters
        ----------
        mask : np.ndarray
            uint8 binary mask (values 0 / 1).

        Returns
        -------
        np.ndarray
            Cleaned mask (values 0 / 1, uint8).
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        cleaned = np.zeros_like(mask)

        # Label 0 is background; start from 1.
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.MIN_COMPONENT_AREA:
                cleaned[labels == label] = 1

        return cleaned

    # ------------------------------------------------------------------
    # Method 6 – validate_mask
    # ------------------------------------------------------------------

    def validate_mask(self, mask: np.ndarray) -> bool:
        """
        Assert that the reconstructed mask contains at least one
        foreground pixel.

        Parameters
        ----------
        mask : np.ndarray
            The final binary mask.

        Returns
        -------
        bool
            True if valid.

        Raises
        ------
        ValueError
            If the mask is completely empty (no structural content
            survived reconstruction).
        """
        foreground = int(np.count_nonzero(mask))
        if foreground == 0:
            raise ValueError(
                "Empty reconstructed mask: no structural content survived. "
                "Check that kept_masks contains non-empty structure masks."
            )
        return True

    # ------------------------------------------------------------------
    # Master entry point – reconstruct
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        kept_masks: list[dict],
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Full pipeline: merge → close → fill → clean → validate.

        Parameters
        ----------
        kept_masks : list[dict]
            ``result["kept_masks"]`` from SAMTextSuppressor.suppress().
            Each dict must contain a ``"mask"`` key.
        image_shape : tuple[int, int]
            ``(height, width)`` of the original image.

        Returns
        -------
        np.ndarray
            ``new_closed_mask`` — uint8 binary array (values 0 / 1)
            containing panels, flaps, and boundaries with text removed.

        Example
        -------
        ::

            result       = suppressor.suppress(classified_masks)
            new_closed_mask = reconstructor.reconstruct(
                result["kept_masks"], image_shape
            )
        """
        # Step 1 – merge all structural masks onto one canvas
        merged = self.draw_masks(kept_masks, image_shape)

        # Step 2 – bridge gaps between structural regions
        merged = self.close_gaps(merged)

        # Step 3 – fill internal voids left by text removal
        merged = self.fill_holes(merged)

        # Step 4 – strip tiny noise fragments
        merged = self.clean_noise(merged)

        # Step 5 – safety check before returning
        self.validate_mask(merged)

        return merged
