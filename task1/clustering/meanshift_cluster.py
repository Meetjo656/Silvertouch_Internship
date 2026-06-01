import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.color import lab2rgb
from config import MEANSHIFT_QUANTILE

class MeanShiftExtractor:
    def __init__(self, pixels):
        self.pixels = pixels

    def extract_colors(self, delta_from_bg_values=None, is_lab=True):
        """
        MeanShift clustering in LAB space for
        perceptual color extraction with Simple Importance Scoring.
        """
        try:
            bandwidth = estimate_bandwidth(
                self.pixels,
                quantile=MEANSHIFT_QUANTILE,
                n_samples=500
            )
            if bandwidth is None or bandwidth <= 0:
                bandwidth = 15.0
        except Exception:
            bandwidth = 15.0

        meanshift = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=True
        )

        labels = meanshift.fit_predict(self.pixels)
        centers = meanshift.cluster_centers_

        if is_lab:
            # Convert LAB centers back to RGB
            centers_rgb_norm = lab2rgb(centers.reshape(-1, 1, 3))
            centers_rgb = np.round(np.clip(centers_rgb_norm, 0, 1) * 255).astype(int).reshape(-1, 3)
        else:
            centers_rgb = np.round(centers).astype(int)

        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )

        cluster_info = []

        for label, count in zip(unique_labels, counts):
            if delta_from_bg_values is not None:
                mean_delta_e = np.mean(delta_from_bg_values[labels == label])
                score = count * mean_delta_e
            else:
                score = float(count)

            cluster_info.append({
                "label": label,
                "color": centers_rgb[label],
                "pixel_count": count,
                "importance_score": score
            })

        # Sort by Importance Score descending
        cluster_info = sorted(
            cluster_info,
            key=lambda x: x["importance_score"],
            reverse=True
        )

        return {
            "labels": labels,
            "colors": centers_rgb,
            "cluster_info": cluster_info
        }
