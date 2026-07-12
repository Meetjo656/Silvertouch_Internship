import cv2
import numpy as np

class EdgeDetector:
    @staticmethod
    def detect_edges(image_crop: np.ndarray) -> np.ndarray:
        """
        Runs Grayscale -> Adaptive Threshold -> Morph closing -> Canny
        """
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding to handle uneven lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological closing to connect fragmented lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Canny edge detection
        edges = cv2.Canny(closed, 50, 150)
        
        return edges

    @staticmethod
    def detect_lines(edges: np.ndarray) -> np.ndarray:
        """
        Returns a blank image with hough lines drawn on it, useful for contour detection
        """
        lines_img = np.zeros_like(edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_img, (x1, y1), (x2, y2), 255, 2)
                
        return lines_img
