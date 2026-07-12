import cv2
import numpy as np
from typing import List

class ContourDetector:
    @staticmethod
    def find_contours(image_mask: np.ndarray) -> List[np.ndarray]:
        """
        Finds external and internal contours from a binary mask.
        """
        contours, _ = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)
        
    @staticmethod
    def get_connected_components_mask(image_mask: np.ndarray) -> np.ndarray:
        """
        Uses connected components to find solid regions, returning a mask of the largest non-background components.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_mask, connectivity=8)
        
        # Create a new mask keeping only reasonably sized components
        out_mask = np.zeros_like(image_mask)
        
        for i in range(1, num_labels): # skip background 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 100: # filter tiny noise
                out_mask[labels == i] = 255
                
        return out_mask
