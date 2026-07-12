from typing import List, Dict, Any
from ..graph import PanelGraph
from .layout_rules import LayoutRules

class Validator:
    def __init__(self, rules: LayoutRules):
        self.rules = rules

    def get_missing_classes(self, graph: PanelGraph) -> List[str]:
        """
        Returns a list of missing classes based on current graph and rules.
        """
        class_counts = {}
        for panel in graph.nodes.values():
            class_counts[panel.class_name] = class_counts.get(panel.class_name, 0) + 1

        missing = []
        all_classes = set(list(self.rules.required.keys()) + list(self.rules.optional.keys()))

        for cls in all_classes:
            count = class_counts.get(cls, 0)
            min_c, max_c = self.rules.get_min_max(cls)
            
            if count < min_c:
                for _ in range(min_c - count):
                    missing.append(cls)

        return missing
