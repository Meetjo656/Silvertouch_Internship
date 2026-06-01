import cv2
import numpy as np
import json


class SegmentationEvaluator:

    def __init__(self, config):
        self.config = config

    # ==========================================
    # PUBLIC ENTRY POINT
    # ==========================================

    def evaluate(
        self,
        image,
        regions,
        rectangles
    ):
        """
        Main evaluation pipeline.
        """
        metrics = {
            "region_coverage": self.compute_region_coverage(image, regions),
            "rectangle_coverage": self.compute_rectangle_coverage(image, rectangles),
            "carton_coverage": self.compute_carton_coverage(image, rectangles),
            
            "num_regions": self.count_regions(regions),
            "num_rectangles": self.count_rectangles(rectangles),
            
            "avg_region_area": self.average_region_area(regions),
            "avg_rectangle_area": self.average_rectangle_area(rectangles),
            "largest_region_area": self.largest_region_area(regions),
            "smallest_region_area": self.smallest_region_area(regions),
            
            "is_under_segmented": self.detect_under_segmentation(regions),
            "is_over_segmented": self.detect_over_segmentation(regions),
            "has_sparse_rectangles": self.detect_sparse_rectangles(rectangles)
        }
        
        return self.build_report(metrics)

    # ==========================================
    # COVERAGE METRICS
    # ==========================================

    def compute_region_coverage(self, image, regions):
        if not regions:
            return 0.0
        h, w = image.shape[:2]
        total_area = h * w
        
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for r in regions:
            cv2.bitwise_or(combined_mask, r["binary_mask"], dst=combined_mask)
            
        region_area = cv2.countNonZero(combined_mask)
        return float(region_area) / total_area

    def compute_rectangle_coverage(self, image, rectangles):
        if not rectangles:
            return 0.0
        h, w = image.shape[:2]
        total_area = h * w
        
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for rect in rectangles:
            box = rect["box"]
            cv2.fillPoly(combined_mask, [box], 255)
            
        rect_area = cv2.countNonZero(combined_mask)
        return float(rect_area) / total_area

    def compute_carton_coverage(self, image, rectangles):
        if not rectangles:
            return 0.0
        h, w = image.shape[:2]
        total_area = h * w
        
        all_points = []
        for rect in rectangles:
            all_points.extend(rect["box"])
            
        if not all_points:
            return 0.0
            
        points = np.array(all_points)
        hull = cv2.convexHull(points)
        carton_area = cv2.contourArea(hull)
        
        return float(carton_area) / total_area

    # ==========================================
    # COUNT METRICS
    # ==========================================

    def count_regions(self, regions):
        return len(regions)

    def count_rectangles(self, rectangles):
        return len(rectangles)

    # ==========================================
    # AREA METRICS
    # ==========================================

    def average_region_area(self, regions):
        if not regions: return 0.0
        return sum(r.get("area", 0) for r in regions) / len(regions)

    def average_rectangle_area(self, rectangles):
        if not rectangles: return 0.0
        return sum(r.get("area", 0) for r in rectangles) / len(rectangles)

    def largest_region_area(self, regions):
        if not regions: return 0.0
        return max(r.get("area", 0) for r in regions)

    def smallest_region_area(self, regions):
        if not regions: return 0.0
        return min(r.get("area", 0) for r in regions)

    # ==========================================
    # QUALITY CHECKS
    # ==========================================

    def detect_under_segmentation(self, regions):
        num = len(regions)
        max_area = self.largest_region_area(regions)
        threshold_area = self.config.get("under_seg_area", 500000)
        return (num < 2) or (max_area > threshold_area)

    def detect_over_segmentation(self, regions):
        num = len(regions)
        avg_area = self.average_region_area(regions)
        threshold_count = self.config.get("over_seg_count", 50)
        threshold_area = self.config.get("over_seg_avg_area", 500)
        return (num > threshold_count) or (avg_area < threshold_area)

    def detect_sparse_rectangles(self, rectangles):
        if not rectangles:
            return True
        avg_area = self.average_rectangle_area(rectangles)
        return avg_area < self.config.get("sparse_rect_area", 100)

    # ==========================================
    # REPORTING
    # ==========================================

    def build_report(self, metrics):
        status = "OK"
        if metrics["is_under_segmented"]:
            status = "UNDER_SEGMENTED"
        elif metrics["is_over_segmented"]:
            status = "OVER_SEGMENTED"
        elif metrics["has_sparse_rectangles"]:
            status = "SPARSE_RECTANGLES"
            
        return {
            "status": status,
            "metrics": metrics
        }

    def print_summary(self, report):
        print("=== Segmentation Report ===")
        print(f"Status: {report['status']}")
        m = report["metrics"]
        print(f"Regions: {m['num_regions']}, Rectangles: {m['num_rectangles']}")
        print(f"Coverage - Region: {m['region_coverage']:.2f}, Rect: {m['rectangle_coverage']:.2f}, Carton: {m['carton_coverage']:.2f}")
        print("===========================")

    def save_report_json(self, report, output_path):
        def default_encoder(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)
            
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4, default=default_encoder)

    # ==========================================
    # DEBUG
    # ==========================================

    def create_evaluation_overlay(self, image, report):
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
            
        m = report["metrics"]
        lines = [
            f"Status: {report['status']}",
            f"Regions: {m['num_regions']}",
            f"Rects: {m['num_rectangles']}",
            f"Reg Cov: {m['region_coverage']:.2f}",
            f"Rect Cov: {m['rectangle_coverage']:.2f}",
            f"Cart Cov: {m['carton_coverage']:.2f}"
        ]
        
        y = 30
        for line in lines:
            color = (0, 0, 255) if report["status"] != "OK" and "Status" in line else (0, 255, 0)
            cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 30
            
        return vis
