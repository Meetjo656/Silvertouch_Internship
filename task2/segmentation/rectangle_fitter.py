import cv2
import numpy as np


class RectangleFitter:

    def __init__(self, config):
        self.config = config

    # ==========================================
    # PUBLIC ENTRY POINT
    # ==========================================

    def fit_rectangles(self, regions):
        """
        Fit rectangles for all valid regions.
        """
        rectangles = []
        for region in regions:
            if "report" in region and not region["report"]["is_valid"]:
                continue
                
            rect = self.fit_rectangle(region)
            if rect is not None:
                rectangles.append(rect)
                
        return rectangles

    # ==========================================
    # SINGLE REGION FITTING
    # ==========================================

    def fit_rectangle(self, region):
        """
        Fit one rotated rectangle.
        """
        points = self.extract_points(region)
        if points is None or len(points) < 3:
            return None
            
        cv_rect = self.compute_min_area_rect(points)
        box = self.convert_rect_to_box(cv_rect)
        
        width = self.compute_width(cv_rect)
        height = self.compute_height(cv_rect)
        area = self.compute_area(width, height)
        aspect_ratio = self.compute_aspect_ratio(width, height)
        angle = self.compute_angle(cv_rect)
        
        rectangle = {
            "region_id": region["id"],
            "cv_rect": cv_rect,
            "center": cv_rect[0],
            "box": box,
            "width": width,
            "height": height,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "angle": angle
        }
        
        coverage = self.compute_coverage_ratio(region, rectangle)
        rectangle["coverage_ratio"] = coverage
        
        if not self.validate_rectangle(rectangle):
            return None
            
        score = self.compute_quality_score(region, rectangle)
        rectangle["quality_score"] = score
        rectangle["report"] = self.build_rectangle_report(region, rectangle)
        
        return rectangle

    def extract_points(self, region):
        """
        Get non-zero points
        from binary mask.
        """
        return cv2.findNonZero(region["binary_mask"])

    def compute_min_area_rect(self, points):
        """
        Wrapper around cv2.minAreaRect().
        """
        return cv2.minAreaRect(points)

    def convert_rect_to_box(self, rect):
        """
        Convert OpenCV rect
        into 4-corner polygon.
        """
        box = cv2.boxPoints(rect)
        return np.int32(box)

    # ==========================================
    # RECTANGLE METRICS
    # ==========================================

    def compute_width(self, rect):
        return rect[1][0]

    def compute_height(self, rect):
        return rect[1][1]

    def compute_area(self, width, height):
        return width * height

    def compute_aspect_ratio(self, width, height):
        if height == 0:
            return 0
        return float(width) / height

    def compute_angle(self, rect):
        return rect[2]

    # ==========================================
    # VALIDATION
    # ==========================================

    def validate_rectangle(self, rectangle):
        """
        Main validation logic.
        """
        if self.is_too_small(rectangle): return False
        if self.is_too_large(rectangle): return False
        if self.is_thin_artifact(rectangle): return False
        if self.has_bad_aspect_ratio(rectangle): return False
        return True

    def is_too_small(self, rectangle):
        min_area = self.config.get("min_rect_area", 100)
        min_width = self.config.get("min_rect_width", 5)
        min_height = self.config.get("min_rect_height", 5)
        
        return (rectangle["area"] < min_area or 
                rectangle["width"] < min_width or 
                rectangle["height"] < min_height)

    def is_too_large(self, rectangle):
        max_area = self.config.get("max_rect_area", 100000000)
        return rectangle["area"] > max_area

    def is_thin_artifact(self, rectangle):
        min_dim = self.config.get("min_rect_dim", 2)
        return rectangle["width"] < min_dim or rectangle["height"] < min_dim

    def has_bad_aspect_ratio(self, rectangle):
        ar = rectangle["aspect_ratio"]
        max_ar = self.config.get("max_rect_aspect_ratio", 20.0)
        min_ar = self.config.get("min_rect_aspect_ratio", 0.05)
        return ar > max_ar or ar < min_ar

    # ==========================================
    # QUALITY SCORING
    # ==========================================

    def compute_quality_score(self, region, rectangle):
        """
        Optional. Estimate how well rectangle fits region.
        """
        coverage = rectangle.get("coverage_ratio", 0)
        return coverage

    def compute_coverage_ratio(self, region, rectangle):
        """
        Region area / Rectangle area
        """
        rect_area = rectangle["area"]
        if rect_area == 0:
            return 0
            
        region_area = region.get("metrics", {}).get("area", 0)
        if region_area == 0 and "area" in region:
            region_area = region["area"]
            
        return float(region_area) / rect_area

    # ==========================================
    # REPORTING
    # ==========================================

    def build_rectangle_report(self, region, rectangle):
        return {
            "region_id": rectangle["region_id"],
            "quality_score": rectangle.get("quality_score", 0),
            "coverage_ratio": rectangle.get("coverage_ratio", 0)
        }

    def summarize_rectangles(self, rectangles):
        total = len(rectangles)
        if total == 0:
            return {"total_rectangles": 0}
            
        avg_quality = sum(r.get("quality_score", 0) for r in rectangles) / total
        avg_area = sum(r["area"] for r in rectangles) / total
        
        return {
            "total_rectangles": total,
            "average_quality_score": avg_quality,
            "average_area": avg_area
        }

    # ==========================================
    # DEBUG
    # ==========================================

    def create_rectangle_debug_image(self, image, rectangles):
        """
        Draw all rectangles.
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
            
        for rect in rectangles:
            box = rect["box"]
            cv2.drawContours(vis, [box], 0, (0, 165, 255), 2)
            
            cx, cy = int(rect["center"][0]), int(rect["center"][1])
            cv2.putText(vis, str(rect["region_id"]), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        return vis

    def visualize_single_rectangle(self, image, rectangle):
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
            
        box = rectangle["box"]
        cv2.drawContours(vis, [box], 0, (255, 0, 0), 2)
        return vis