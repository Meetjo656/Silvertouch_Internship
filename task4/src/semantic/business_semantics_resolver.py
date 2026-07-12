import os
import json
from typing import List, Dict

class BusinessSemanticsResolver:
    """
    Dynamically looks up enterprise domain knowledge (status flags, enum mappings) 
    for the specific tables and columns requested in the generation context.
    """
    def __init__(self, data_dir: str):
        self.catalog_path = os.path.join(data_dir, "processed", "business_semantics_catalog.json")
        self.catalog = {}
        if os.path.exists(self.catalog_path):
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                self.catalog = json.load(f)
                
    def resolve_for_table(self, table_name: str, selected_columns: List[str]) -> List[str]:
        """
        Returns explicit semantic rules for the specified columns in the table.
        """
        rules = []
        t_lower = table_name.lower()
        global_filters = self.catalog.get("_GLOBAL_MINED_FILTERS", {})
        
        explicit_mappings = self.catalog.get(t_lower, {})
        
        for col in selected_columns:
            c_lower = col.lower()
            
            # 1. Prefer explicit DDL/Doc mappings
            if c_lower in explicit_mappings:
                mapping = explicit_mappings[c_lower]
                formatted = [f"'{k}' = {v}" for k, v in mapping.items()]
                rules.append(f"{c_lower}: {', '.join(formatted)}")
                
        return rules
