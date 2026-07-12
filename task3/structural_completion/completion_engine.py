import numpy as np
from typing import List, Dict, Any, Tuple

# pyrefly: ignore [missing-import]
from .graph import PanelGraph
# pyrefly: ignore [missing-import]
from .panel import Panel
# pyrefly: ignore [missing-import]
from .reasoning.layout_rules import LayoutRules
# pyrefly: ignore [missing-import]
from .reasoning.validator import Validator
# pyrefly: ignore [missing-import]
from .reasoning.panel_assignment import PanelAssignment
# pyrefly: ignore [missing-import]
from .reasoning.sanity_checker import SanityChecker
# pyrefly: ignore [missing-import]
from .reasoning.failure_report import FailureReporter
# pyrefly: ignore [missing-import]
from .geometry.artwork_cropper import ArtworkCropper
# pyrefly: ignore [missing-import]
from .geometry.rectangle_detector import RectangleDetector
# pyrefly: ignore [missing-import]
from .geometry.rectangle_filter import RectangleFilter

class CompletionEngine:
    # pyrefly: ignore [bad-function-definition]
    def __init__(self, config_path: str = None, max_iterations: int = 3):
        self.rules = LayoutRules(config_path)
        self.validator = Validator(self.rules)
        self.assigner = PanelAssignment(self.rules)
        self.sanity_checker = SanityChecker()
        self.reporter = FailureReporter()
        self.cropper = ArtworkCropper(margin_pct=0.10)
        self.max_iterations = max_iterations

    def process(self, image: np.ndarray, raw_detections: List[Dict[str, Any]], image_id: str = "unknown") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Runs the iterative completion engine.
        """
        graph = PanelGraph()
        
        # 1. Build initial graph from RF-DETR detections
        for det in raw_detections:
            panel = Panel(
                class_name=det["class_name"],
                bbox=det["bbox"],
                confidence=det["confidence"],
                source="detector"
            )
            graph.add_node(panel)

        total_inferred = 0
        all_errors = []
        
        # Determine Artwork crop for geometry extraction
        h, w = image.shape[:2]
        crop_bounds = self.cropper.get_crop_bounds(graph, w, h)
        cx1, cy1, cx2, cy2 = crop_bounds
        
        # If crop is too small, fallback
        if (cx2 - cx1) < 10 or (cy2 - cy1) < 10:
            return [p.to_dict() for p in graph.nodes.values()], {"error": "Invalid crop"}
            
        img_crop = image[cy1:cy2, cx1:cx2]
        
        # Extract geometry once (it's independent of semantic assignments)
        raw_rectangles = RectangleDetector.extract_rectangles(img_crop, cx1, cy1)
        candidate_rects = RectangleFilter.filter_and_feature_extract(raw_rectangles, graph, crop_bounds)
        
        # Iterative Completion Loop
        for iteration in range(self.max_iterations):
            missing_classes = self.validator.get_missing_classes(graph)
            
            if not missing_classes:
                break
                
            inferred_panels, unused_candidates = self.assigner.assign(graph, candidate_rects, missing_classes)
            
            if not inferred_panels:
                # Log failures
                for m_class in missing_classes:
                    self.reporter.log_failure(image_id, m_class, "No valid matching geometry found")
                break
                
            # Add new panels to graph
            for p in inferred_panels:
                graph.add_node(p)
                total_inferred += 1
                
            candidate_rects = unused_candidates
            
        # Final Sanity Check
        is_sane, errors = self.sanity_checker.check(graph)
        all_errors.extend(errors)
        
        metadata = {
            "iterations_run": iteration + 1,
            "total_inferred": total_inferred,
            "is_sane": is_sane,
            "sanity_errors": all_errors,
            "crop_bounds": crop_bounds
        }
        
        final_detections = [p.to_dict() for p in graph.nodes.values()]
        return final_detections, metadata
