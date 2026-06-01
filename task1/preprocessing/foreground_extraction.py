import numpy as np
import colour
from .estimate_background import estimate_background
from config import FOREGROUND_DELTA_THRESHOLD

def extract_foreground(image, threshold: int = FOREGROUND_DELTA_THRESHOLD, border_width: int = 5):
    """
    Extract foreground pixels using DeltaE distance relative to estimated background.
    Uses the colour-science library for conversion and DeltaE calculations.
    """
    # Estimate background color
    bg_rgb = estimate_background(image, border_width=border_width)

    # Normalize image and background to [0, 1] for colour-science
    image_norm = image.astype(np.float64) / 255.0
    bg_rgb_norm = np.array(bg_rgb, dtype=np.float64) / 255.0

    # Convert image and background to LAB using colour-science
    xyz_image = colour.sRGB_to_XYZ(image_norm)
    lab_image = colour.XYZ_to_Lab(xyz_image)

    xyz_bg = colour.sRGB_to_XYZ(bg_rgb_norm)
    lab_bg = colour.XYZ_to_Lab(xyz_bg)

    lab_pixels = lab_image.reshape(-1, 3)
    rgb_pixels = image.reshape(-1, 3)

    # Perceptual DeltaE CIE 1976 using colour-science (vectorized)
    delta_from_bg = colour.delta_E(lab_pixels, lab_bg, method='CIE 1976')

    foreground_mask = delta_from_bg > threshold

    foreground_lab = lab_pixels[foreground_mask]
    foreground_rgb = rgb_pixels[foreground_mask]

    # Robust empty foreground safety check to prevent downstream clustering crash
    if len(foreground_lab) == 0:
        raise ValueError("No foreground detected.")

    delta_from_bg_values = delta_from_bg[foreground_mask]
    mask_2d = foreground_mask.reshape(image.shape[:2]).astype(np.uint8) * 255

    return {
        "lab": foreground_lab,
        "rgb": foreground_rgb,
        "background_rgb": bg_rgb,
        "mask": mask_2d,
        "delta_from_bg_values": delta_from_bg_values
    }
