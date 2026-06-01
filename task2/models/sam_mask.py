import numpy as np
import json


class SAMMask:
    """
    Lightweight data model for a single SAM-generated mask.

    Wraps the raw annotation dict produced by SamAutomaticMaskGenerator
    and exposes typed, named properties instead of raw dict access.
    Also provides conversion helpers to/from the project's region dict
    format so it can be slotted into either representation at will.
    """

    def __init__(
        self,
        mask_id: int,
        segmentation: np.ndarray,
        area: int,
        bbox: dict,
        centroid: tuple,
        predicted_iou: float,
        stability_score: float,
    ):
        """
        Parameters
        ----------
        mask_id : int
            Sequential 1-based identifier assigned by SAMRegionExtractor.
        segmentation : np.ndarray
            Boolean (H, W) or uint8 (0/255) mask array.
        area : int
            Pixel count of the mask.
        bbox : dict
            {x: int, y: int, w: int, h: int}
        centroid : tuple
            (cx: float, cy: float)
        predicted_iou : float
            SAM's predicted mask quality (0 – 1).
        stability_score : float
            SAM's stability score (0 – 1).
        """
        self.mask_id         = mask_id
        self.segmentation    = segmentation
        self.area            = area
        self.bbox            = bbox
        self.centroid        = centroid
        self.predicted_iou   = predicted_iou
        self.stability_score = stability_score

    # ------------------------------------------------------------------
    # FACTORIES
    # ------------------------------------------------------------------

    @classmethod
    def from_raw(cls, raw_mask: dict, mask_id: int) -> "SAMMask":
        """
        Build a SAMMask from a raw SamAutomaticMaskGenerator annotation dict.

        Parameters
        ----------
        raw_mask : dict
        mask_id : int

        Returns
        -------
        SAMMask
        """
        seg = np.asarray(raw_mask["segmentation"], dtype=np.uint8) * 255

        sam_bbox = raw_mask.get("bbox", [0, 0, 0, 0])
        bbox = {
            "x": int(sam_bbox[0]),
            "y": int(sam_bbox[1]),
            "w": int(sam_bbox[2]),
            "h": int(sam_bbox[3]),
        }

        # Centroid via moments
        import cv2
        m = cv2.moments(seg)
        if m["m00"] != 0:
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        else:
            cx = bbox["x"] + bbox["w"] / 2.0
            cy = bbox["y"] + bbox["h"] / 2.0

        return cls(
            mask_id         = mask_id,
            segmentation    = seg,
            area            = int(raw_mask.get("area", int(np.count_nonzero(seg)))),
            bbox            = bbox,
            centroid        = (cx, cy),
            predicted_iou   = float(raw_mask.get("predicted_iou", 1.0)),
            stability_score = float(raw_mask.get("stability_score", 1.0)),
        )

    @classmethod
    def from_region(cls, region: dict) -> "SAMMask":
        """
        Reconstruct a SAMMask from a project region dict.
        """
        return cls(
            mask_id         = region.get("id", 0),
            segmentation    = region.get("binary_mask"),
            area            = region.get("area", 0),
            bbox            = region.get("bbox", {}),
            centroid        = region.get("centroid", (0.0, 0.0)),
            predicted_iou   = region.get("iou", 1.0),
            stability_score = region.get("stability", 1.0),
        )

    # ------------------------------------------------------------------
    # CONVERSION
    # ------------------------------------------------------------------

    def to_region(self) -> dict:
        """
        Convert to the project's standard region dict format so this
        object can be passed directly to RegionMerger, RegionAnalyzer, etc.

        Returns
        -------
        dict
        """
        return {
            "id"          : self.mask_id,
            "binary_mask" : self.segmentation,
            "area"        : self.area,
            "bbox"        : self.bbox,
            "centroid"    : self.centroid,
            "iou"         : self.predicted_iou,
            "stability"   : self.stability_score,
        }

    def to_dict(self) -> dict:
        """
        Serialise to a JSON-safe dict (excludes the numpy array).
        """
        return {
            "mask_id"        : self.mask_id,
            "area"           : self.area,
            "bbox"           : self.bbox,
            "centroid"       : list(self.centroid) if self.centroid else None,
            "predicted_iou"  : self.predicted_iou,
            "stability_score": self.stability_score,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    # ------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------

    @property
    def binary_mask(self) -> np.ndarray:
        """Alias so region-dict consumers can access mask as binary_mask."""
        return self.segmentation

    @property
    def quality_score(self) -> float:
        """Combined quality: geometric mean of iou × stability."""
        return float(self.predicted_iou * self.stability_score) ** 0.5

    # ------------------------------------------------------------------
    # DUNDER
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SAMMask("
            f"id={self.mask_id}, "
            f"area={self.area}, "
            f"iou={self.predicted_iou:.3f}, "
            f"stability={self.stability_score:.3f})"
        )
