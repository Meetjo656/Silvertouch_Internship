import cv2
import numpy as np
from typing import List, Tuple
from .edge_detector import EdgeDetector
from .contour_detector import ContourDetector

class RectangleDetector:
    @staticmethod
    def extract_rectangles(image_crop: np.ndarray, offset_x: int, offset_y: int) -> List[List[int]]:
        """
        Combines Canny edges, Hough lines, and Connected Components to extract a comprehensive list of bounding boxes.
        Returns boxes in absolute image coordinates [x_min, y_min, x_max, y_max].
        """
        edges = EdgeDetector.detect_edges(image_crop)
        lines_mask = EdgeDetector.detect_lines(edges)
        
        # Combine edges and lines
        combined = cv2.bitwise_or(edges, lines_mask)
        
        # Get connected components to solidify regions
        solid_mask = ContourDetector.get_connected_components_mask(combined)
        
        # Find contours on both the raw edges and the solidified mask
        contours_raw = ContourDetector.find_contours(edges)
        contours_solid = ContourDetector.find_contours(solid_mask)
        
        all_contours = contours_raw + contours_solid
        
        rectangles = []
        for cnt in all_contours:
            # Polygon approximation
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # If it has roughly 4 corners, we consider it a rectangle candidate
            # Even if it's not perfect 4, we will take bounding box and let the filter handle it
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(cnt)
                
                abs_x_min = x + offset_x
                abs_y_min = y + offset_y
                abs_x_max = abs_x_min + w
                abs_y_max = abs_y_min + h
                
                rectangles.append([abs_x_min, abs_y_min, abs_x_max, abs_y_max])
                
        # Remove duplicates
        unique_rects = []
        for r in rectangles:
            if r not in unique_rects:
                unique_rects.append(r)
                
        return unique_rects
