import cv2
import numpy as np

try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    _SAM_AVAILABLE = True
except ImportError:
    _SAM_AVAILABLE = False


class SAMSegmenter:
    """
    Wraps Meta's Segment Anything Model (SAM) to:
      - Load a checkpoint and move it to the best available device.
      - Run automatic mask generation over a full image.
      - Convert raw SAM mask dicts into the project's region format.
      - Expose a high-level segment() entry point that mirrors the
        interface of StructuralSegmenter.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, model_path: str, config: dict = None):
        """
        Parameters
        ----------
        model_path : str
            Path to the SAM ViT-H checkpoint (.pth file).
        config : dict, optional
            Project-wide config dict.  Falls back to sensible defaults
            when a key is missing.
        """
        if not _SAM_AVAILABLE:
            raise ImportError(
                "The 'segment_anything' package is required for SAMSegmenter.\n"
                "Install it with:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        self.config = config or {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        model_type = self.config.get("sam_model_type", "vit_h")
        self.model = sam_model_registry[model_type](checkpoint=model_path)
        self.model.to(device=self.device)

        # Predictor (point / box queries)
        self.predictor = SamPredictor(self.model)

        # Automatic mask generator (created on demand via create_mask_generator)
        self.mask_generator = None

        # The last image set on the predictor
        self.image = None

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self, image_path: str):
        """
        Read an image from disk and set it on the SAM predictor.

        Parameters
        ----------
        image_path : str
            Absolute or relative path to a BGR image file (OpenCV format).
        """
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        self.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.image)

    def set_image(self, image: np.ndarray):
        """
        Accept a BGR numpy array (already loaded) and set it on the predictor.

        Parameters
        ----------
        image : np.ndarray
            BGR image array (H, W, 3).
        """
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.image)

    # ------------------------------------------------------------------
    # Mask generator
    # ------------------------------------------------------------------

    def create_mask_generator(self):
        """
        Instantiate SamAutomaticMaskGenerator using config values.
        Must be called before generate_masks() when using automatic mode.
        """
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=self.config.get("sam_points_per_side", 32),
            pred_iou_thresh=self.config.get("pred_iou_thresh", 0.88),
            stability_score_thresh=self.config.get("stability_score_thresh", 0.95),
            crop_n_layers=self.config.get("crop_n_layers", 0),
            crop_n_points_per_layer=self.config.get("crop_n_points_per_layer", 1),
            min_mask_region_area=self.config.get("min_mask_region_area", 100),
            output_mode=self.config.get("output_mode", "binary_mask"),
        )

    # ------------------------------------------------------------------
    # Raw mask generation
    # ------------------------------------------------------------------

    def generate_masks(self, image: np.ndarray) -> list:
        """
        Run automatic mask generation on a BGR image array.

        Parameters
        ----------
        image : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        list[dict]
            Raw SAM annotation dicts, each containing at minimum:
            'segmentation', 'area', 'bbox', 'predicted_iou',
            'stability_score'.
        """
        if self.mask_generator is None:
            self.create_mask_generator()

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_masks = self.mask_generator.generate(rgb_image)
        return raw_masks

    # ------------------------------------------------------------------
    # Mask conversion
    # ------------------------------------------------------------------

    def convert_mask(self, raw_mask: dict) -> dict:
        """
        Convert a single raw SAM annotation dict into the project's
        region format (compatible with RegionMerger / RegionAnalyzer).

        Parameters
        ----------
        raw_mask : dict
            One element from the list returned by generate_masks().

        Returns
        -------
        dict
            Region dict with keys: mask, binary_mask, area, bbox,
            centroid, stability, iou.
        """
        seg = raw_mask["segmentation"]          # bool array (H, W)
        binary_mask = seg.astype(np.uint8) * 255

        area = raw_mask["area"]
        x, y, w, h = raw_mask["bbox"]          # SAM returns [x, y, w, h]
        stability = raw_mask["stability_score"]
        iou = raw_mask["predicted_iou"]

        # Compute centroid from moments
        moments = cv2.moments(binary_mask)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = x + w / 2.0
            cy = y + h / 2.0

        return {
            "mask": seg,
            "binary_mask": binary_mask,
            "area": area,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "centroid": (cx, cy),
            "stability": stability,
            "iou": iou,
        }

    def convert_masks(self, raw_masks: list) -> list:
        """
        Convert an entire list of raw SAM annotation dicts.

        Parameters
        ----------
        raw_masks : list[dict]
            Output of generate_masks().

        Returns
        -------
        list[dict]
            List of project-format region dicts with sequential IDs.
        """
        regions = []
        for idx, raw_mask in enumerate(raw_masks):
            region = self.convert_mask(raw_mask)
            region["id"] = idx + 1          # 1-based, consistent with
                                            # connectedComponentsWithStats
            regions.append(region)
        return regions

    # ------------------------------------------------------------------
    # High-level segment entry point
    # ------------------------------------------------------------------

    def segment(self, image: np.ndarray) -> list:
        """
        End-to-end segmentation: generate automatic masks for *image*
        and return a list of project-format region dicts.

        Parameters
        ----------
        image : np.ndarray
            BGR image array (H, W, 3).

        Returns
        -------
        list[dict]
            Region dicts ready for downstream processing (e.g. passed to
            RegionMerger, RegionAnalyzer, RectangleFitter, etc.).

        Raises
        ------
        ValueError
            If the model has not been loaded (should not normally happen
            after __init__, but kept as a guard).
        """
        if self.model is None:
            raise ValueError("SAM model is not loaded.")

        if image is None or image.size == 0:
            raise ValueError("Received an empty or None image.")

        # Generate raw SAM masks
        raw_masks = self.generate_masks(image)

        # Convert to project region format
        regions = self.convert_masks(raw_masks)

        return regions