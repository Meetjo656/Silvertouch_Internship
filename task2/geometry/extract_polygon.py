import cv2
import numpy as np


class PolygonExtractor:

    def __init__(self, config):
        self.config = config

    # =====================================================
    # PUBLIC ENTRY POINT
    # =====================================================

    def extract_polygons(self, regions):
        """
        Extract polygons from all regions.

        Input:
            list of region dictionaries

        Output:
            list of polygon dictionaries
        """
        polygons = []
        for region in regions:
            poly = self.extract_polygon(region)
            if poly is not None:
                polygons.append(poly)
        return polygons

    def extract_polygon(self, region):
        """
        Extract polygon from one region.
        """
        mask = region.get("binary_mask")
        if mask is None:
            return None
            
        contours = self.extract_contours(mask)
        if not contours:
            return None
            
        largest = self.get_largest_contour(contours)
        if largest is None:
            return None
            
        polygon = self.approximate_polygon(largest)
        
        return self.build_polygon_object(region, polygon)

    # =====================================================
    # CONTOUR EXTRACTION
    # =====================================================

    def extract_contours(self, mask):
        """
        Extract contours from binary mask.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    def get_largest_contour(
        self,
        contours
    ):
        """
        Keep only largest contour.
        """
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def filter_contours(
        self,
        contours
    ):
        """
        Remove tiny contours.
        """
        pass

    # =====================================================
    # POLYGON GENERATION
    # =====================================================

    def contour_to_polygon(
        self,
        contour
    ):
        """
        Convert contour to simplified polygon.
        """
        pass

    def approximate_polygon(
        self,
        contour
    ):
        """
        Wrapper around cv2.approxPolyDP.
        """
        epsilon_factor = self.config.get("approx_epsilon_factor", 0.005)
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx

    def simplify_polygon(
        self,
        polygon
    ):
        """
        Additional simplification.
        """
        pass

    # =====================================================
    # POLYGON METRICS
    # =====================================================

    def compute_polygon_area(
        self,
        polygon
    ):
        pass

    def compute_polygon_perimeter(
        self,
        polygon
    ):
        pass

    def compute_polygon_centroid(
        self,
        polygon
    ):
        pass

    def compute_bounding_box(
        self,
        polygon
    ):
        pass

    def count_vertices(
        self,
        polygon
    ):
        pass

    # =====================================================
    # POLYGON REPRESENTATION
    # =====================================================

    def polygon_to_points(
        self,
        polygon
    ):
        """
        Convert OpenCV format
        into simple point list.
        """
        points = []
        for pt in polygon:
            points.append({
                "x": int(pt[0][0]),
                "y": int(pt[0][1])
            })
        return points

    def build_polygon_object(
        self,
        region,
        polygon
    ):
        """
        Standard output format.
        """
        return {
            "region_id": region["id"],
            "points": self.polygon_to_points(polygon),
            "raw_polygon": polygon
        }

    # =====================================================
    # QUALITY CHECKS
    # =====================================================

    def is_valid_polygon(
        self,
        polygon
    ):
        pass

    def has_minimum_vertices(
        self,
        polygon
    ):
        pass

    def has_reasonable_area(
        self,
        polygon
    ):
        pass

    # =====================================================
    # DEBUG HELPERS
    # =====================================================

    def draw_polygon(
        self,
        image,
        polygon
    ):
        pass

    def draw_vertices(
        self,
        image,
        polygon
    ):
        pass

    def create_polygon_debug_image(
        self,
        image,
        polygon
    ):
        pass