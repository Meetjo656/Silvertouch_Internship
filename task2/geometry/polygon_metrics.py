import cv2
import numpy as np


class PolygonMetrics:

    def __init__(self, config=None):
        self.config = config

    # ==========================================
    # BASIC GEOMETRY
    # ==========================================

    def polygon_area(
        self,
        polygon
    ):
        """
        Compute polygon area.
        """
        pass

    def polygon_perimeter(
        self,
        polygon
    ):
        """
        Compute polygon perimeter.
        """
        pass

    def vertex_count(
        self,
        polygon
    ):
        """
        Number of vertices.
        """
        pass

    # ==========================================
    # CENTROID
    # ==========================================

    def polygon_centroid(
        self,
        polygon
    ):
        """
        Compute geometric centroid.
        """
        pass

    # ==========================================
    # BOUNDING GEOMETRY
    # ==========================================

    def bounding_box(
        self,
        polygon
    ):
        """
        Axis-aligned bounding box.
        """
        pass

    def rotated_bounding_box(
        self,
        polygon
    ):
        """
        Rotated min-area rectangle.
        """
        pass

    def bounding_box_area(
        self,
        polygon
    ):
        pass

    # ==========================================
    # SHAPE METRICS
    # ==========================================

    def aspect_ratio(
        self,
        polygon
    ):
        pass

    def fill_ratio(
        self,
        polygon
    ):
        """
        polygon_area / bbox_area
        """
        pass

    def compactness(
        self,
        polygon
    ):
        """
        4πA / P²
        """
        pass

    def circularity(
        self,
        polygon
    ):
        pass

    def solidity(
        self,
        polygon
    ):
        pass

    def extent(
        self,
        polygon
    ):
        pass

    # ==========================================
    # EDGE METRICS
    # ==========================================

    def edge_lengths(
        self,
        polygon
    ):
        pass

    def shortest_edge(
        self,
        polygon
    ):
        pass

    def longest_edge(
        self,
        polygon
    ):
        pass

    def average_edge_length(
        self,
        polygon
    ):
        pass

    # ==========================================
    # ANGLE METRICS
    # ==========================================

    def internal_angles(
        self,
        polygon
    ):
        pass

    def minimum_angle(
        self,
        polygon
    ):
        pass

    def maximum_angle(
        self,
        polygon
    ):
        pass

    def average_angle(
        self,
        polygon
    ):
        pass

    # ==========================================
    # CONVEXITY METRICS
    # ==========================================

    def convex_hull(
        self,
        polygon
    ):
        pass

    def hull_area(
        self,
        polygon
    ):
        pass

    def convexity_ratio(
        self,
        polygon
    ):
        pass

    # ==========================================
    # CARTON-SPECIFIC METRICS
    # ==========================================

    def rectangularity(
        self,
        polygon
    ):
        """
        Area / minAreaRect area
        """
        pass

    def trapezoid_score(
        self,
        polygon
    ):
        """
        Heuristic score for
        trapezoidal flaps.
        """
        pass

    def tab_score(
        self,
        polygon
    ):
        """
        Heuristic score for
        rounded tabs.
        """
        pass

    # ==========================================
    # COMPOSITE METRICS
    # ==========================================

    def metric_report(
        self,
        polygon
    ):
        """
        Return all metrics
        in one dictionary.
        """
        pass