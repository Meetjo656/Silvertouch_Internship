import json
import os

class LayoutRules:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "layout_rules.json")
            
        with open(config_path, "r") as f:
            self.rules = json.load(f)
            
        self.layout_type = self.rules.get("layout_type", "default")
        self.required = self.rules.get("required", {})
        self.optional = self.rules.get("optional", {})
        self.adjacency = self.rules.get("adjacency", {})

    def get_min_max(self, class_name: str) -> tuple:
        if class_name in self.required:
            val = self.required[class_name]
            if isinstance(val, int):
                return (val, val)
            return tuple(val)
        elif class_name in self.optional:
            val = self.optional[class_name]
            if isinstance(val, list) and len(val) == 2:
                return tuple(val)
            elif isinstance(val, int):
                return (0, val)
        return (0, 999)

    def get_allowed_neighbors(self, class_name: str) -> list:
        return self.adjacency.get(class_name, [])
