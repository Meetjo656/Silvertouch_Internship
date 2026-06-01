import cv2
import numpy as np


class RegionAnalyzer:

    def __init__(self, config):
        self.config = config

    # ==================================================
    # PUBLIC ENTRY POINT
    # ==================================================

    def analyze_regions(self, regions):
        """
        Main function.

        Input:
            list of regions

        Output:
            analyzed regions with metrics
        """
        analyzed_regions = []
        for region in regions:
            analyzed = region.copy()
            metrics = self.compute_region_metrics(region)
            analyzed["metrics"] = metrics
            
            if metrics is not None:
                flags = self.assign_flags(metrics)
            else:
                flags = ["INVALID_CONTOUR"]
                
            analyzed["flags"] = flags
            analyzed["report"] = self.build_region_report(region, metrics, flags)
            analyzed_regions.append(analyzed)
            
        return analyzed_regions

    # ==================================================
    # REGION METRICS
    # ==================================================

    def compute_region_metrics(self, region):
        """
        Compute all region statistics.

        Returns:
            metrics dictionary
        """
        mask = region["binary_mask"]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        contour = max(contours, key=cv2.contourArea)
        
        area = self.compute_area(contour)
        perimeter = self.compute_perimeter(contour)
        bbox = self.compute_bounding_box(contour)
        aspect_ratio = self.compute_aspect_ratio(bbox)
        extent = self.compute_extent(area, bbox)
        solidity = self.compute_solidity(contour, area)
        fill_ratio = self.compute_fill_ratio(area, bbox)
        
        return {
            "area": area,
            "perimeter": perimeter,
            "bbox": bbox,
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "solidity": solidity,
            "fill_ratio": fill_ratio,
            "contour": contour
        }

    def compute_area(self, contour):
        """
        Contour area.
        """
        return cv2.contourArea(contour)

    def compute_perimeter(self, contour):
        """
        Contour perimeter.
        """
        return cv2.arcLength(contour, True)

    def compute_bounding_box(self, contour):
        """
        x, y, w, h
        """
        return cv2.boundingRect(contour)

    def compute_aspect_ratio(self, bbox):
        """
        width / height
        """
        x, y, w, h = bbox
        if h == 0:
            return 0
        return float(w) / h

    def compute_extent(self, area, bbox):
        """
        area / bbox_area
        """
        x, y, w, h = bbox
        bbox_area = w * h
        if bbox_area == 0:
            return 0
        return float(area) / bbox_area

    def compute_solidity(
        self,
        contour,
        area
    ):
        """
        area / convex_hull_area
        """
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return 0
        return float(area) / hull_area

    def compute_fill_ratio(
        self,
        area,
        bbox
    ):
        """
        Area occupancy inside bbox.
        """
        return self.compute_extent(area, bbox)

    # ==================================================
    # REGION VALIDATION
    # ==================================================

    def validate_region(
        self,
        metrics
    ):
        """
        Decide if region looks valid.
        """
        if metrics is None:
            return False
        flags = self.assign_flags(metrics)
        return len(flags) == 0

    def is_too_small(
        self,
        metrics
    ):
        """
        Small noise detection.
        """
        min_area = self.config.get("min_region_area", 500)
        return metrics["area"] < min_area

    def is_too_large(
        self,
        metrics
    ):
        """
        Merged panel detection.
        """
        max_area = self.config.get("max_region_area", 100000000)
        return metrics["area"] > max_area

    def is_thin_artifact(
        self,
        metrics
    ):
        """
        Fold line artifact detection.
        """
        ar = metrics["aspect_ratio"]
        max_ar = self.config.get("max_aspect_ratio", 10.0)
        min_ar = self.config.get("min_aspect_ratio", 0.1)
        return ar > max_ar or ar < min_ar

    def has_poor_fill_ratio(
        self,
        metrics
    ):
        """
        Sparse region detection.
        """
        min_fill = self.config.get("min_fill_ratio", 0.2)
        return metrics["fill_ratio"] < min_fill

    # ==================================================
    # REGION FLAGS
    # ==================================================

    def assign_flags(
        self,
        metrics
    ):
        """
        Generate warnings.

        Example:

        [
            "TOO_SMALL",
            "THIN_ARTIFACT"
        ]
        """
        flags = []
        if self.is_too_small(metrics):
            flags.append("TOO_SMALL")
        if self.is_too_large(metrics):
            flags.append("TOO_LARGE")
        if self.is_thin_artifact(metrics):
            flags.append("THIN_ARTIFACT")
        if self.has_poor_fill_ratio(metrics):
            flags.append("POOR_FILL_RATIO")
        return flags

    # ==================================================
    # REPORTING
    # ==================================================

    def build_region_report(
        self,
        region,
        metrics,
        flags
    ):
        """
        Standardized report object.
        """
        return {
            "region_id": region["id"],
            "is_valid": len(flags) == 0,
            "flags": flags,
            "metrics_summary": {
                "area": metrics["area"] if metrics else 0,
                "solidity": metrics["solidity"] if metrics else 0
            }
        }

    def summarize_regions(
        self,
        analyzed_regions
    ):
        """
        Dataset-level statistics.

        Average area
        Average solidity
        etc.
        """
        valid_regions = [r for r in analyzed_regions if r["report"]["is_valid"]]
        
        summary = {
            "total_regions": len(analyzed_regions),
            "valid_regions": len(valid_regions),
            "average_area": 0,
            "average_solidity": 0,
            "average_fill_ratio": 0
        }
        
        if valid_regions:
            summary["average_area"] = sum(r["metrics"]["area"] for r in valid_regions) / len(valid_regions)
            summary["average_solidity"] = sum(r["metrics"]["solidity"] for r in valid_regions) / len(valid_regions)
            summary["average_fill_ratio"] = sum(r["metrics"]["fill_ratio"] for r in valid_regions) / len(valid_regions)
            
        return summary

    # ==================================================
    # DEBUG VISUALIZATION
    # ==================================================

    def create_region_statistics_image(
        self,
        image,
        analyzed_regions
    ):
        """
        Draw metrics directly
        onto image.

        Optional debug utility.
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
            
        for r in analyzed_regions:
            metrics = r["metrics"]
            if not metrics:
                continue
            
            x, y, w, h = metrics["bbox"]
            color = (0, 255, 0) if r["report"]["is_valid"] else (0, 0, 255)
            
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
            
            text = f"A:{int(metrics['area'])} S:{metrics['solidity']:.2f}"
            cv2.putText(vis, text, (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        return vis

    def create_flag_visualization(
        self,
        image,
        analyzed_regions
    ):
        """
        Highlight suspicious regions.

        Example:

        Red = problematic

        Green = valid
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
            
        for r in analyzed_regions:
            mask = r["binary_mask"]
            color = (0, 255, 0) if r["report"]["is_valid"] else (0, 0, 255)
            
            colored_mask = np.zeros_like(vis)
            colored_mask[mask > 0] = color
            
            vis = cv2.addWeighted(vis, 1.0, colored_mask, 0.5, 0)
            
        return vis