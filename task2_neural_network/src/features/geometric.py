"""
Geometric Feature Extraction.

This module provides functions to calculate geometric features from binary masks.
These features serve as the input vector for our baseline Random Forest classifier.
"""

import cv2
import numpy as np
from typing import Dict, Any

def extract_geometric_features(binary_mask: np.ndarray, image_area: float = None) -> Dict[str, float]:
    """
    Extracts geometric properties from a 2D binary mask.
    
    Args:
        binary_mask (np.ndarray): 2D numpy array containing 0s and 1s (or 255s).
        image_area (float, optional): Total area of the original image, for relative features.
        
    Returns:
        Dict[str, float]: A dictionary containing calculated geometric features.
    """
    # Ensure binary (0 and 1) robustly
    binary_mask = (binary_mask > 0).astype(np.uint8)
        
    features: Dict[str, float] = {}
    
    # 1. Area
    area = np.sum(binary_mask)
    features["area"] = float(area)
    
    if image_area and image_area > 0:
        features["relative_area"] = float(area / image_area)
    else:
        features["relative_area"] = 0.0
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # If no contours found (empty mask), return zeros
        features.update({
            "perimeter": 0.0,
            "bbox_width": 0.0,
            "bbox_height": 0.0,
            "aspect_ratio": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "equiv_diameter": 0.0,
            "circularity": 0.0,
            "rotated_width": 0.0,
            "rotated_height": 0.0,
            "rotated_aspect_ratio": 0.0,
            "rotated_area": 0.0,
            "vertex_count": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0
        })
        return features

    # Assume largest contour is the object of interest
    main_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(main_contour)
    
    # 2. Perimeter
    perimeter = cv2.arcLength(main_contour, True)
    features["perimeter"] = float(perimeter)
    
    # 3. Bounding Box
    x, y, w, h = cv2.boundingRect(main_contour)
    features["bbox_width"] = float(w)
    features["bbox_height"] = float(h)
    
    # 4. Aspect Ratio
    features["aspect_ratio"] = float(w / h) if h > 0 else 0.0
    
    # 5. Solidity (Contour Area / Convex Hull Area)
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    features["solidity"] = float(contour_area / hull_area) if hull_area > 0 else 0.0
    
    # 6. Extent (Contour Area / Bounding Box Area)
    bbox_area = w * h
    features["extent"] = float(contour_area / bbox_area) if bbox_area > 0 else 0.0
    
    # 7. Equivalent Diameter
    features["equiv_diameter"] = float(np.sqrt(4 * contour_area / np.pi))
    
    # 8. Circularity
    if perimeter > 0:
        features["circularity"] = float((4 * np.pi * contour_area) / (perimeter * perimeter))
    else:
        features["circularity"] = 0.0
        
    # --- TASK 3: Rotation-Invariant Features ---
    rect = cv2.minAreaRect(main_contour)
    (cx, cy), (rw, rh), angle = rect
    features["rotated_width"] = float(rw)
    features["rotated_height"] = float(rh)
    
    # Min(w,h) / Max(w,h) to ensure rotation invariance
    if max(rw, rh) > 0:
        features["rotated_aspect_ratio"] = float(min(rw, rh) / max(rw, rh))
    else:
        features["rotated_aspect_ratio"] = 0.0
        
    features["rotated_area"] = float(rw * rh)
    
    # --- TASK 4: Polygon Complexity ---
    epsilon = 0.02 * perimeter  # 2% of arc length is standard for approxPolyDP
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    features["vertex_count"] = float(len(approx))
    
    # --- TASK 5: Centroids ---
    M = cv2.moments(main_contour)
    if M["m00"] != 0:
        features["centroid_x"] = float(M["m10"] / M["m00"])
        features["centroid_y"] = float(M["m01"] / M["m00"])
    else:
        features["centroid_x"] = 0.0
        features["centroid_y"] = 0.0
        
    return features
