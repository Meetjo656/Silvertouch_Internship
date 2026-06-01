import cv2
import numpy as np


class SAMPolygonBuilder:
    """
    Builds polygon representations from SAM binary masks.

    Unlike PolygonExtractor (which works with generic regions),
    SAMPolygonBuilder is tailored to the properties of SAM masks:
      - Masks may have multiple disconnected components.
      - Quality signals (iou, stability) inform which contour to use.
      - Epsilon for Douglas-Peucker simplification is tuned separately
        for SAM (tends to produce smoother, larger masks).

    Output polygon format
    ---------------------
    {
        "region_id"   : int,
        "points"      : [{"x": int, "y": int}, ...],
        "raw_polygon" : np.ndarray,           # OpenCV contour array
        "vertex_count": int,
        "area"        : float,
        "perimeter"   : float,
        "source"      : "SAM"
    }
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------

    def build_polygons(self, regions: list) -> list:
        """
        Build polygon dicts for all SAM regions.

        Parameters
        ----------
        regions : list[dict]
            Project-format SAM region dicts (binary_mask, id, iou, …).

        Returns
        -------
        list[dict]
            Polygon dicts (one per region that has a valid contour).
        """
        polygons = []
        for region in regions:
            poly = self.build_polygon(region)
            if poly is not None:
                polygons.append(poly)
        return polygons

    def build_polygon(self, region: dict) -> dict | None:
        """
        Build a single polygon dict from one SAM region.

        Parameters
        ----------
        region : dict

        Returns
        -------
        dict | None  — None if the mask has no usable contour.
        """
        mask = region.get("binary_mask")
        if mask is None or cv2.countNonZero(mask) == 0:
            return None

        contour = self.select_contour(mask)
        if contour is None or len(contour) < 3:
            return None

        simplified = self.simplify_contour(contour)
        if len(simplified) < 3:
            simplified = contour          # fall back to unsimplified

        points    = self.contour_to_points(simplified)
        area      = float(cv2.contourArea(simplified))
        perimeter = float(cv2.arcLength(simplified, True))

        return {
            "region_id"   : region.get("id"),
            "points"      : points,
            "raw_polygon" : simplified,
            "vertex_count": len(points),
            "area"        : area,
            "perimeter"   : perimeter,
            "iou"         : region.get("iou", 1.0),
            "stability"   : region.get("stability", 1.0),
            "source"      : "SAM",
        }

    # ------------------------------------------------------------------
    # CONTOUR SELECTION
    # ------------------------------------------------------------------

    def extract_contours(self, mask: np.ndarray) -> list:
        """
        Extract all external contours from a binary mask.

        Parameters
        ----------
        mask : np.ndarray  uint8 (0/255)

        Returns
        -------
        list of cv2 contour arrays
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return list(contours)

    def select_contour(self, mask: np.ndarray):
        """
        Choose the best single contour from *mask*.

        Strategy: pick the largest by area, which for SAM masks
        represents the dominant segment component.

        Parameters
        ----------
        mask : np.ndarray

        Returns
        -------
        np.ndarray contour, or None
        """
        contours = self.extract_contours(mask)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def select_all_contours(self, mask: np.ndarray) -> list:
        """
        Return all contours above the minimum area threshold.
        Useful for masks with multiple distinct components.

        Parameters
        ----------
        mask : np.ndarray

        Returns
        -------
        list of contour arrays
        """
        min_area = self.config.get("min_component_area", 500)
        contours = self.extract_contours(mask)
        return [c for c in contours if cv2.contourArea(c) >= min_area]

    # ------------------------------------------------------------------
    # SIMPLIFICATION
    # ------------------------------------------------------------------

    def simplify_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Apply Douglas-Peucker polygon approximation to reduce vertex count.

        Uses ``sam_approx_epsilon_factor`` from config (default 0.008,
        slightly larger than the structural pipeline's 0.005 to reflect
        SAM's already-smooth masks).

        Parameters
        ----------
        contour : np.ndarray  OpenCV contour

        Returns
        -------
        np.ndarray  simplified contour
        """
        eps_factor = self.config.get("sam_approx_epsilon_factor", 0.008)
        epsilon    = eps_factor * cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, epsilon, True)

    def smooth_contour(self, contour: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply a simple moving-average smoothing to contour coordinates to
        reduce jagged edges from the binary mask boundary.

        Parameters
        ----------
        contour : np.ndarray  (N, 1, 2)
        kernel_size : int  must be odd

        Returns
        -------
        np.ndarray  smoothed contour (same shape)
        """
        if len(contour) < kernel_size:
            return contour

        pts = contour[:, 0, :]            # (N, 2)
        kernel = np.ones(kernel_size) / kernel_size

        xs = np.convolve(pts[:, 0], kernel, mode="wrap").astype(np.int32)
        ys = np.convolve(pts[:, 1], kernel, mode="wrap").astype(np.int32)

        smoothed = np.stack([xs, ys], axis=1)[:, np.newaxis, :]   # (N, 1, 2)
        return smoothed

    # ------------------------------------------------------------------
    # FORMAT CONVERSION
    # ------------------------------------------------------------------

    def contour_to_points(self, contour: np.ndarray) -> list:
        """
        Convert an OpenCV contour array (N, 1, 2) to a list of
        ``{"x": int, "y": int}`` dicts for JSON serialisation.

        Parameters
        ----------
        contour : np.ndarray

        Returns
        -------
        list[dict]
        """
        return [{"x": int(pt[0][0]), "y": int(pt[0][1])} for pt in contour]

    def points_to_contour(self, points: list) -> np.ndarray:
        """
        Inverse of contour_to_points.

        Parameters
        ----------
        points : list[dict]  with "x", "y" keys

        Returns
        -------
        np.ndarray  (N, 1, 2) int32
        """
        arr = np.array([[p["x"], p["y"]] for p in points], dtype=np.int32)
        return arr[:, np.newaxis, :]

    # ------------------------------------------------------------------
    # QUALITY METRICS
    # ------------------------------------------------------------------

    def compute_polygon_metrics(self, polygon: dict) -> dict:
        """
        Derive secondary metrics from an already-built polygon dict.

        Parameters
        ----------
        polygon : dict  output of build_polygon()

        Returns
        -------
        dict  {area, perimeter, vertex_count, compactness, rectangularity}
        """
        raw = polygon.get("raw_polygon")
        if raw is None or len(raw) < 3:
            return {}

        area      = float(cv2.contourArea(raw))
        perimeter = float(cv2.arcLength(raw, True))
        compactness = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

        # Rectangularity: area / minAreaRect area
        rect = cv2.minAreaRect(raw)
        rect_area = rect[1][0] * rect[1][1]
        rectangularity = area / rect_area if rect_area > 0 else 0.0

        return {
            "area"          : area,
            "perimeter"     : perimeter,
            "vertex_count"  : len(raw),
            "compactness"   : compactness,
            "rectangularity": rectangularity,
        }

    # ------------------------------------------------------------------
    # MULTI-COMPONENT SUPPORT
    # ------------------------------------------------------------------

    def build_multi_component_polygons(self, region: dict) -> list:
        """
        Build one polygon per connected component in the mask.
        Useful when a SAM mask contains multiple disconnected blobs.

        Parameters
        ----------
        region : dict

        Returns
        -------
        list[dict]  one polygon dict per component (with suffixed IDs)
        """
        mask = region.get("binary_mask")
        if mask is None:
            return []

        contours = self.select_all_contours(mask)
        polygons = []

        for i, contour in enumerate(contours):
            simplified = self.simplify_contour(contour)
            if len(simplified) < 3:
                continue

            points    = self.contour_to_points(simplified)
            area      = float(cv2.contourArea(simplified))
            perimeter = float(cv2.arcLength(simplified, True))

            polygons.append({
                "region_id"   : f"{region.get('id')}_{i}",
                "points"      : points,
                "raw_polygon" : simplified,
                "vertex_count": len(points),
                "area"        : area,
                "perimeter"   : perimeter,
                "iou"         : region.get("iou", 1.0),
                "stability"   : region.get("stability", 1.0),
                "source"      : "SAM",
            })

        return polygons

    # ------------------------------------------------------------------
    # DEBUG VISUALIZATION
    # ------------------------------------------------------------------

    def create_polygon_debug_image(
        self,
        image_shape: tuple,
        polygons: list
    ) -> np.ndarray:
        """
        Render all polygons on a black canvas with random colours.

        Parameters
        ----------
        image_shape : tuple  (H, W) or (H, W, C)
        polygons : list[dict]  output of build_polygons()

        Returns
        -------
        np.ndarray  BGR debug image
        """
        h, w = image_shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        rng = np.random.default_rng(7)

        for poly in polygons:
            pts = poly.get("raw_polygon")
            if pts is None or len(pts) < 3:
                continue
            color = tuple(int(c) for c in rng.integers(60, 230, size=3))
            cv2.polylines(canvas, [pts], True, color, 2)
            cv2.fillPoly(canvas, [pts], tuple(c // 3 for c in color))

            # Label
            m = cv2.moments(pts)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                label = str(poly.get("region_id", "?"))
                cv2.putText(canvas, label, (cx - 8, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(canvas, label, (cx - 8, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return canvas
