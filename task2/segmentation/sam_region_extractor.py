import cv2
import numpy as np


class SAMRegionExtractor:
    """
    Converts raw SAM annotation dicts into the project's standardized
    region format (same shape as the regions produced by
    StructuralSegmenter.extract_regions).

    This makes SAM-derived regions compatible with the shared downstream
    pipeline:  RegionMerger → RegionAnalyzer → RectangleFitter → …

    Region dict schema
    ------------------
    {
        "id"          : int,          # 1-based sequential index
        "binary_mask" : np.ndarray,   # uint8 (H×W), 0 or 255
        "area"        : int,          # pixel count
        "bbox"        : {x, y, w, h}, # bounding box (ints)
        "centroid"    : (cx, cy),     # float centroid
        "iou"         : float,        # SAM predicted_iou
        "stability"   : float,        # SAM stability_score
    }
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------

    def extract_regions(self, raw_masks: list) -> list:
        """
        Convert a list of raw SAM annotation dicts to project region dicts.

        Parameters
        ----------
        raw_masks : list[dict]
            Output of SamAutomaticMaskGenerator.generate().

        Returns
        -------
        list[dict]
            Project-format region dicts with sequential 1-based IDs.
        """
        regions = []
        for idx, raw in enumerate(raw_masks):
            region = self.extract_single_region(raw, region_id=idx + 1)
            if region is not None:
                regions.append(region)
        return regions

    # ------------------------------------------------------------------
    # SINGLE MASK CONVERSION
    # ------------------------------------------------------------------

    def extract_single_region(self, raw_mask: dict, region_id: int) -> dict | None:
        """
        Convert one SAM annotation dict into a project region dict.

        Parameters
        ----------
        raw_mask : dict
            Single element from SamAutomaticMaskGenerator.generate().
        region_id : int
            Sequential 1-based ID to assign.

        Returns
        -------
        dict | None
            Region dict, or None if the mask is empty / invalid.
        """
        seg = raw_mask.get("segmentation")
        if seg is None:
            return None

        binary_mask = self.build_binary_mask(seg)
        if binary_mask is None:
            return None

        area   = self.compute_area(binary_mask)
        bbox   = self.compute_bbox(raw_mask, binary_mask)
        centroid = self.compute_centroid(binary_mask, bbox)

        return {
            "id"          : region_id,
            "binary_mask" : binary_mask,
            "area"        : area,
            "bbox"        : bbox,
            "centroid"    : centroid,
            "iou"         : float(raw_mask.get("predicted_iou", 1.0)),
            "stability"   : float(raw_mask.get("stability_score", 1.0)),
        }

    # ------------------------------------------------------------------
    # MASK BUILDING
    # ------------------------------------------------------------------

    def build_binary_mask(self, segmentation) -> np.ndarray | None:
        """
        Convert SAM's boolean segmentation array to a uint8 mask (0 / 255).

        Parameters
        ----------
        segmentation : bool ndarray (H, W)

        Returns
        -------
        np.ndarray | None
            uint8 array, or None if the mask is entirely empty.
        """
        mask = np.asarray(segmentation, dtype=np.uint8) * 255
        if cv2.countNonZero(mask) == 0:
            return None
        return mask

    # ------------------------------------------------------------------
    # GEOMETRY HELPERS
    # ------------------------------------------------------------------

    def compute_area(self, binary_mask: np.ndarray) -> int:
        """
        Pixel count of the non-zero region in *binary_mask*.
        """
        return int(cv2.countNonZero(binary_mask))

    def compute_bbox(self, raw_mask: dict, binary_mask: np.ndarray) -> dict:
        """
        Return a bounding-box dict {x, y, w, h}.

        Prefers SAM's own bbox field (x, y, w, h list); falls back to
        cv2.boundingRect on the binary mask.

        Parameters
        ----------
        raw_mask : dict
        binary_mask : np.ndarray

        Returns
        -------
        dict  {x: int, y: int, w: int, h: int}
        """
        sam_bbox = raw_mask.get("bbox")
        if sam_bbox is not None and len(sam_bbox) == 4:
            x, y, w, h = [int(v) for v in sam_bbox]
            return {"x": x, "y": y, "w": w, "h": h}

        # Fallback
        pts = cv2.findNonZero(binary_mask)
        if pts is None:
            return {"x": 0, "y": 0, "w": 0, "h": 0}
        x, y, w, h = cv2.boundingRect(pts)
        return {"x": x, "y": y, "w": w, "h": h}

    def compute_centroid(self, binary_mask: np.ndarray, bbox: dict) -> tuple:
        """
        Compute centroid from image moments.
        Falls back to bbox centre if moments are degenerate.

        Parameters
        ----------
        binary_mask : np.ndarray
        bbox : dict  {x, y, w, h}

        Returns
        -------
        (float, float)  (cx, cy)
        """
        moments = cv2.moments(binary_mask)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = bbox["x"] + bbox["w"] / 2.0
            cy = bbox["y"] + bbox["h"] / 2.0
        return (cx, cy)

    # ------------------------------------------------------------------
    # BATCH HELPERS
    # ------------------------------------------------------------------

    def sort_by_area(self, regions: list, descending: bool = True) -> list:
        """
        Return *regions* sorted by pixel area.

        Parameters
        ----------
        regions : list[dict]
        descending : bool

        Returns
        -------
        list[dict]
        """
        return sorted(regions, key=lambda r: r.get("area", 0), reverse=descending)

    def filter_empty_regions(self, regions: list) -> list:
        """
        Drop any regions whose binary_mask is None or all-zero.
        """
        valid = []
        for r in regions:
            mask = r.get("binary_mask")
            if mask is not None and cv2.countNonZero(mask) > 0:
                valid.append(r)
        return valid

    # ------------------------------------------------------------------
    # DEBUG VISUALIZATION
    # ------------------------------------------------------------------

    def create_extraction_debug_image(
        self,
        image_shape: tuple,
        regions: list
    ) -> np.ndarray:
        """
        Render each extracted region in a random colour on a black canvas.

        Parameters
        ----------
        image_shape : tuple  (H, W) or (H, W, C)
        regions : list[dict]

        Returns
        -------
        np.ndarray  BGR debug image
        """
        h, w = image_shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        rng = np.random.default_rng(42)

        for region in regions:
            mask = region.get("binary_mask")
            if mask is None:
                continue
            color = tuple(int(c) for c in rng.integers(50, 255, size=3))
            canvas[mask > 0] = color

        return canvas
