import numpy as np
import colour
from config import DELTA_E_MATCH_THRESHOLD

class CompareColors:
    """
    Class to compare two colors perceptually using colour-science DeltaE CIE 2000.
    """
    def rgb_to_lab(self, rgb):
        """
        Convert RGB array to LAB using colour-science.
        """
        rgb_norm = np.array(rgb, dtype=np.float64) / 255.0
        xyz = colour.sRGB_to_XYZ(rgb_norm)
        lab = colour.XYZ_to_Lab(xyz)
        return lab

    def delta_e(self, color1, color2):
        """
        Calculate CIEDE2000 distance between two colors using colour-science.
        """
        lab1 = self.rgb_to_lab(color1)
        lab2 = self.rgb_to_lab(color2)
        distance = colour.delta_E(lab1, lab2, method='CIE 2000')
        return round(float(distance), 2)

    def is_match(self, color1, color2, threshold: float = DELTA_E_MATCH_THRESHOLD):
        """
        Determine color matching based on perceptual threshold.
        """
        distance = self.delta_e(color1, color2)
        return {
            "match": distance <= threshold,
            "delta_e": distance
        }
