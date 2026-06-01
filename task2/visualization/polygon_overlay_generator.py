import cv2
import numpy as np


class PolygonOverlayGenerator:

    def __init__(self, config):
        self.config = config

    # ==========================================
    # PUBLIC ENTRY POINT
    # ==========================================

    def generate(
        self,
        image,
        polygons
    ):
        """
        Generate complete polygon overlay.
        """
        overlay = image.copy() if image is not None else None
        if overlay is None:
            return None

        for seg in polygons:
            poly_data = seg.polygon if hasattr(seg, 'polygon') else seg
            if not poly_data or "points" not in poly_data:
                continue
                
            region_id = poly_data.get("region_id", seg.segment_id if hasattr(seg, 'segment_id') else 0)
            points = poly_data["points"]
            if not points:
                continue
                
            contour = np.array([[[p["x"], p["y"]]] for p in points], dtype=np.int32)
            color = self.get_polygon_color(region_id)
            
            self.draw_polygon(overlay, contour, color=color, thickness=2)
            self.draw_vertices(overlay, points, color=(0, 0, 255))
            self.draw_segment_id(overlay, contour, region_id, color)
            
        return overlay

    # ==========================================
    # POLYGON DRAWING
    # ==========================================

    def draw_polygon(
        self,
        image,
        polygon,
        color=(0, 255, 0),
        thickness=2
    ):
        """
        Draw polygon boundary.
        """
        cv2.drawContours(image, [polygon], -1, color, thickness)
        return image

    def draw_polygons(
        self,
        image,
        polygons
    ):
        """
        Draw all polygons.
        """
        for p in polygons:
            self.draw_polygon(image, p)
        return image

    # ==========================================
    # VERTEX VISUALIZATION
    # ==========================================

    def draw_vertices(
        self,
        image,
        points,
        color=(0, 0, 255)
    ):
        """
        Draw vertex points.
        """
        for p in points:
            x, y = p["x"], p["y"]
            cv2.circle(image, (x, y), 3, color, -1)
        return image

    def draw_vertex_indices(
        self,
        image,
        points
    ):
        """
        Optional debug view.
        """
        for i, p in enumerate(points):
            x, y = p["x"], p["y"]
            cv2.putText(image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return image

    # ==========================================
    # LABELS
    # ==========================================

    def draw_segment_id(
        self,
        image,
        contour,
        segment_id,
        color=(0, 255, 0)
    ):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = contour[0][0][0], contour[0][0][1]
            
        cv2.putText(image, str(segment_id), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image

    def draw_polygon_info(
        self,
        image,
        polygon_data
    ):
        pass

    # ==========================================
    # COLOR HELPERS
    # ==========================================

    def get_polygon_color(
        self,
        polygon_id
    ):
        """
        Deterministic color generation.
        """
        import hashlib
        seed_val = int(hashlib.sha256(str(polygon_id).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_val)
        return tuple(int(x) for x in rng.integers(50, 255, size=3))

    # ==========================================
    # DEBUG VIEWS
    # ==========================================

    def create_vertex_debug_view(
        self,
        image,
        polygons
    ):
        view = image.copy()
        for poly in polygons:
            pts = poly.get("points", []) if isinstance(poly, dict) else poly
            self.draw_vertices(view, pts)
            self.draw_vertex_indices(view, pts)
        return view

    def create_outline_view(
        self,
        image,
        polygons
    ):
        view = np.zeros_like(image)
        for poly in polygons:
            pts = poly.get("points", []) if isinstance(poly, dict) else poly
            if pts:
                contour = np.array([[[p["x"], p["y"]]] for p in pts], dtype=np.int32)
                self.draw_polygon(view, contour)
        return view

    def create_filled_polygon_view(
        self,
        image,
        polygons
    ):
        view = np.zeros_like(image)
        for poly in polygons:
            region_id = poly.get("region_id", 0) if isinstance(poly, dict) else 0
            pts = poly.get("points", []) if isinstance(poly, dict) else poly
            if pts:
                contour = np.array([[[p["x"], p["y"]]] for p in pts], dtype=np.int32)
                color = self.get_polygon_color(region_id)
                cv2.drawContours(view, [contour], -1, color, -1)
        return view

    # ==========================================
    # EXPORT
    # ==========================================

    def save_overlay(
        self,
        overlay,
        output_path
    ):
        cv2.imwrite(output_path, overlay)