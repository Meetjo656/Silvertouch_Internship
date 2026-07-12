from typing import List, Tuple
from ..graph import PanelGraph
from ..geometry.rectangle_filter import calculate_iou

class SanityChecker:
    @staticmethod
    def check(graph: PanelGraph) -> Tuple[bool, List[str]]:
        """
        Runs final QA on the graph. Returns (is_valid, list_of_errors).
        """
        errors = []
        
        for panel in graph.nodes.values():
            if panel.x_min < 0 or panel.y_min < 0:
                errors.append(f"Panel {panel.id} ({panel.class_name}) has negative coordinates.")
            if panel.width <= 0 or panel.height <= 0:
                errors.append(f"Panel {panel.id} ({panel.class_name}) has zero or negative dimensions.")
                
        # Check for egregious overlaps between structural panels
        panel_list = list(graph.nodes.values())
        for i in range(len(panel_list)):
            for j in range(i + 1, len(panel_list)):
                p1 = panel_list[i]
                p2 = panel_list[j]
                
                # Exclude Artwork from overlap checks, as it's a container
                if p1.class_name in ["Artwork", "not_artwork", "Extra_Areas"] or p2.class_name in ["Artwork", "not_artwork", "Extra_Areas"]:
                    continue
                    
                if calculate_iou(p1.bbox, p2.bbox) > 0.2:
                    errors.append(f"Overlap detected between {p1.class_name} and {p2.class_name}.")
                    
        return len(errors) == 0, errors
