import cv2
import numpy as np
import os


from segmentation.region_analyzer import RegionAnalyzer
from segmentation.rectangle_fitter import RectangleFitter
from segmentation.region_merger import RegionMerger
from segmentation.region_splitter import RegionSplitter
from segmentation.segmentation_evaluator import SegmentationEvaluator
from visualization.polygon_overlay_generator import PolygonOverlayGenerator
from visualization.segment_map_generator import SegmentMapGenerator
from geometry.extract_polygon import PolygonExtractor
from geometry.polygon_refiner import PolygonRefiner
from geometry.polygon_validator import PolygonValidator
from models.segment import Segment

class StructuralSegmenter:

    def __init__(self, config):
        self.config = config
        self.region_analyzer = RegionAnalyzer(config)
        self.rectangle_fitter = RectangleFitter(config)
        self.region_merger = RegionMerger(config)
        self.region_splitter = RegionSplitter(config)
        self.segmentation_evaluator = SegmentationEvaluator(config)
        self.overlay_generator = PolygonOverlayGenerator(config)
        self.segment_map_generator = SegmentMapGenerator(config)
        self.polygon_extractor = PolygonExtractor(config)
        self.polygon_refiner = PolygonRefiner(config)
        self.polygon_validator = PolygonValidator(config)

    def segment(self, image, structural_mask):

        debug_store = {}

        # --------------------------------------------------
        # Step 1: Close structural gaps
        # --------------------------------------------------
        closed_mask = self.close_gaps(structural_mask)
        debug_store["01_closed_mask"] = closed_mask

        # --------------------------------------------------
        # Step 2: Build floorplan
        # --------------------------------------------------
        floorplan = self.build_floorplan(closed_mask)
        debug_store["02_floorplan"] = floorplan

        # --------------------------------------------------
        # Step 3: Invert
        # --------------------------------------------------
        inverted = self.invert_floorplan(floorplan)
        debug_store["03_inverted"] = inverted

        # --------------------------------------------------
        # Step 4: Extract enclosed regions
        # --------------------------------------------------
        regions, labels = self.extract_regions(inverted)
        debug_store["03a_connected_components"] = self.create_connected_components_debug(labels)
        debug_store["03b_extracted_regions"] = self.create_region_debug_image(regions, floorplan.shape)

        # --------------------------------------------------
        # Step 4.5: Merge regions
        # --------------------------------------------------
        regions = self.region_merger.merge_regions(regions)
        # Note: can add debug_store here if merge debug visuals are implemented later

        # --------------------------------------------------
        # Step 4.75: Split overly large/complex regions
        # --------------------------------------------------
        regions = self.region_splitter.split_regions(regions)

        # --------------------------------------------------
        # Step 5: Analyze and Filter regions
        # --------------------------------------------------
        analyzed_regions = self.region_analyzer.analyze_regions(regions)
        debug_store["04a_region_flags"] = self.region_analyzer.create_flag_visualization(floorplan, analyzed_regions)
        debug_store["04b_region_stats"] = self.region_analyzer.create_region_statistics_image(floorplan, analyzed_regions)

        filtered_regions = [r for r in analyzed_regions if r["report"]["is_valid"]]
        debug_store["04c_filtered_regions"] = self.create_region_debug_image(filtered_regions, floorplan.shape)

        # --------------------------------------------------
        # Step 6: Fit rotated rectangles
        # --------------------------------------------------
        rects = self.rectangle_fitter.fit_rectangles(filtered_regions)
        debug_store["05a_fitted_rectangles"] = self.rectangle_fitter.create_rectangle_debug_image(floorplan, rects)

        # --------------------------------------------------
        # Step 6.5: Extract & Refine Polygons
        # --------------------------------------------------
        extracted_polys = self.polygon_extractor.extract_polygons(filtered_regions)
        polygons = []
        for poly in extracted_polys:
            refined = self.polygon_refiner.refine_polygon(poly)
            if refined is not None:
                val_report = self.polygon_validator.validate_polygon(refined)
                refined["validation"] = val_report
                if val_report["valid"]:
                    polygons.append(refined)
        
        debug_store["05b_extracted_polygons"] = self.polygon_extractor.create_polygon_debug_image(floorplan, polygons)

        # --------------------------------------------------
        # Step 7: Evaluate
        # --------------------------------------------------
        report = self.segmentation_evaluator.evaluate(floorplan, filtered_regions, rects)
        debug_store["06a_evaluation_overlay"] = self.segmentation_evaluator.create_evaluation_overlay(floorplan, report)
        
        # --------------------------------------------------
        # Step 8: Build Segment Objects
        # --------------------------------------------------
        segments = []
        # Build lookup for rectangles and polygons by region_id
        rect_map = {r["region_id"]: r for r in rects}
        poly_map = {p["region_id"]: p for p in polygons}
        
        for region in filtered_regions:
            region_id = region["id"]
            rect = rect_map.get(region_id)
            poly = poly_map.get(region_id)
            
            # Create object
            seg = Segment(region_id, region["binary_mask"])
            
            # Base properties
            if "metrics" in region and region["metrics"] is not None:
                seg.set_area(region["metrics"].get("area"))
                seg.set_bbox(region["metrics"].get("bbox"))
                seg.set_centroid(region.get("centroid"))
            else:
                seg.set_area(region.get("area"))
                seg.set_bbox(region.get("bbox"))
                seg.set_centroid(region.get("centroid"))
                
            if "flags" in region:
                for f in region["flags"]:
                    seg.add_flag(f)
                    
            # Rectangle properties
            if rect is not None:
                seg.set_rectangle(rect["box"])
                seg.set_dimensions(rect["width"], rect["height"])
                seg.set_angle(rect["angle"])
                
            # Polygon properties
            if poly is not None:
                seg.set_polygon(poly)
                
            segments.append(seg)

        # --------------------------------------------------
        # Step 9: Visualization
        # --------------------------------------------------
        overlay = self.overlay_generator.generate(image, segments)

        segment_map = self.segment_map_generator.generate(
            floorplan.shape,
            segments
        )

        debug_store["07a_overlay"] = overlay
        debug_store["07b_segment_map"] = segment_map

        return {
            "segments": segments,
            "rectangles": rects,
            "overlay": overlay,
            "segment_map": segment_map,
            "debug_store": debug_store,
            "report": report
        }

    # ======================================================
    # STEP 1
    # ======================================================

    def close_gaps(self, mask):

        kernel_size = self.config.get(
            "closing_kernel_size",
            7
        )

        kernel = np.ones(
            (kernel_size, kernel_size),
            np.uint8
        )

        closed = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        return closed

    # ======================================================
    # STEP 2
    # ======================================================

    def add_outer_contour(self, mask):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return np.zeros_like(mask)

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        contour_mask = np.zeros_like(mask)

        cv2.drawContours(
            contour_mask,
            [largest_contour],
            -1,
            255,
            thickness=3
        )

        return contour_mask

    def build_floorplan(self, mask):

        outer_contour = self.add_outer_contour(mask)

        floorplan = cv2.bitwise_or(
            mask,
            outer_contour
        )

        return floorplan

    # ======================================================
    # STEP 3
    # ======================================================

    def invert_floorplan(self, floorplan):

        return cv2.bitwise_not(floorplan)

    # ======================================================
    # STEP 4
    # ======================================================

    def extract_regions(self, inverted):

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                inverted,
                connectivity=8
            )
        )

        regions = []

        for i in range(1, num_labels):

            region_mask = (
                labels == i
            ).astype(np.uint8) * 255

            region = {
                "id": i,
                "area": stats[i, cv2.CC_STAT_AREA],
                "bbox": {
                    "x": stats[i, cv2.CC_STAT_LEFT],
                    "y": stats[i, cv2.CC_STAT_TOP],
                    "w": stats[i, cv2.CC_STAT_WIDTH],
                    "h": stats[i, cv2.CC_STAT_HEIGHT]
                },
                "centroid": centroids[i],
                "binary_mask": region_mask
            }

            regions.append(region)

        return regions, labels









    # ======================================================
    # DEBUG VISUALIZATIONS
    # ======================================================

    def create_connected_components_debug(self, labels):

        num_labels = np.max(labels) + 1
        rng = np.random.default_rng(42)

        colors = rng.integers(
            0,
            255,
            size=(num_labels, 3),
            dtype=np.uint8
        )
        colors[0] = [0, 0, 0]

        debug_img = colors[labels]

        return debug_img

    def create_region_debug_image(self, regions, shape):

        h, w = shape[:2]
        debug_img = np.zeros((h, w, 3), dtype=np.uint8)

        rng = np.random.default_rng(42)

        for region in regions:
            color = rng.integers(
                0,
                255,
                size=3,
                dtype=np.uint8
            )

            mask = region["binary_mask"]
            debug_img[mask > 0] = color

        return debug_img