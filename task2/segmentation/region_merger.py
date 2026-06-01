import cv2
import numpy as np


class RegionMerger:

    def __init__(self, config):
        self.config = config

    # ==========================================
    # PUBLIC ENTRY POINT
    # ==========================================

    def merge_regions(self, regions):
        """
        Main pipeline.

        Input:
            regions

        Output:
            merged_regions
        """
        merged = list(regions)
        changed = True
        
        while changed:
            changed = False
            candidates = self.find_merge_candidates(merged)
            for (a, b) in candidates:
                if self.should_merge(a, b):
                    new_region = self.merge_two_regions(a, b)
                    if new_region is None:
                        continue
                    if not self.is_merge_reasonable(new_region):
                        continue
                        
                    merged.remove(a)
                    merged.remove(b)
                    merged.append(new_region)
                    changed = True
                    break
                    
        return merged

    # ==========================================
    # PAIR GENERATION
    # ==========================================

    def find_merge_candidates(
        self,
        regions
    ):
        """
        Generate region pairs
        worth evaluating.
        """
        candidates = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if self.are_regions_nearby(regions[i], regions[j]):
                    candidates.append((regions[i], regions[j]))
        return candidates

    def are_regions_nearby(
        self,
        region_a,
        region_b
    ):
        """
        Fast distance test.
        """
        distance = self.compute_centroid_distance(region_a, region_b)
        threshold = self.config.get("candidate_distance", 150)
        return distance < threshold

    # ==========================================
    # MERGE DECISION
    # ==========================================

    def should_merge(
        self,
        region_a,
        region_b
    ):
        """
        Master merge decision.
        """
        mask_a = region_a["binary_mask"]
        mask_b = region_b["binary_mask"]
        
        merge_distance = self.config.get("merge_distance", 5)
        kernel = np.ones((merge_distance, merge_distance), np.uint8)
        
        dilated_a = cv2.dilate(mask_a, kernel)
        overlap = cv2.bitwise_and(dilated_a, mask_b)
        
        distance = self.compute_centroid_distance(region_a, region_b)
        max_distance = self.config.get("merge_centroid_distance", 100)
        
        return (distance < max_distance and np.any(overlap))

    def compute_centroid_distance(
        self,
        region_a,
        region_b
    ):
        ax, ay = region_a["centroid"]
        bx, by = region_b["centroid"]
        return np.hypot(ax-bx, ay-by)

    def compute_bbox_overlap(
        self,
        region_a,
        region_b
    ):
        pass

    def compute_edge_distance(
        self,
        region_a,
        region_b
    ):
        pass

    def are_masks_touching(
        self,
        region_a,
        region_b
    ):
        pass

    # ==========================================
    # MERGE EXECUTION
    # ==========================================

    def merge_two_regions(
        self,
        region_a,
        region_b
    ):
        """
        Create one combined region.
        """
        new_mask = self.merge_masks(
            region_a["binary_mask"],
            region_b["binary_mask"]
        )
        
        new_id = f"{region_a['id']}_{region_b['id']}"
        
        merged_region = self.recompute_region_properties(new_mask)
        if merged_region is None:
            return None

        merged_region["id"] = new_id
        
        merged_region["source_regions"] = [
            region_a["id"],
            region_b["id"]
        ]

        merged_region["merge_depth"] = (
            max(
                region_a.get("merge_depth",0),
                region_b.get("merge_depth",0)
            ) + 1
        )
        
        return merged_region

    def merge_masks(
        self,
        mask_a,
        mask_b
    ):
        return cv2.bitwise_or(mask_a, mask_b)

    def recompute_region_properties(
        self,
        merged_mask
    ):
        area = cv2.countNonZero(merged_mask)
        points = cv2.findNonZero(merged_mask)
        
        if points is None:
            return None
            
        x, y, w, h = cv2.boundingRect(points)
        moments = cv2.moments(merged_mask)
        
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = 0
            cy = 0
            
        return {
            "id": None,
            "binary_mask": merged_mask,
            "area": area,
            "bbox": {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            },
            "centroid": (cx, cy)
        }

    # ==========================================
    # REGION GRAPH
    # ==========================================

    def build_adjacency_graph(
        self,
        regions
    ):
        """
        Optional future extension.
        """
        pass

    def find_connected_groups(
        self,
        graph
    ):
        pass

    # ==========================================
    # VALIDATION
    # ==========================================

    def validate_merged_region(
        self,
        region
    ):
        pass

    def is_merge_reasonable(
        self,
        merged_region
    ):
        max_area = self.config.get("max_merged_area", 1000000)
        bbox = merged_region["bbox"]
        bbox_area = (
            bbox["w"] * bbox["h"]
        )
        fill_ratio = merged_region["area"] / bbox_area
        min_fill = self.config.get("min_fill_ratio", 0.5)
        return merged_region["area"] < max_area and fill_ratio >= min_fill

    # ==========================================
    # DEBUG
    # ==========================================

    def create_merge_debug_image(
        self,
        image,
        original_regions,
        merged_regions
    ):
        pass

    def visualize_merge_candidate(
        self,
        image,
        region_a,
        region_b
    ):
        pass