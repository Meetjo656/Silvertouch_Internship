import os
import sys
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.sql_structure_extractor import SQLStructureExtractor

class StructuralMatch:
    @staticmethod
    def _calculate_score(expected_items: list, generated_items: list) -> float:

        if not expected_items and not generated_items:
            return 1.0
        if not expected_items and generated_items:
            return 0.0
            
        expected_set = set(str(x).lower().strip() for x in expected_items)
        generated_set = set(str(x).lower().strip() for x in generated_items)
        
        intersection = expected_set.intersection(generated_set)
        return len(intersection) / len(expected_set)

    @staticmethod
    def match(expected_sql: str, generated_sql: str, dialect: str = "postgres") -> Dict[str, float]:
        extractor = SQLStructureExtractor(dialect)
        
        expected = extractor.extract(expected_sql)
        generated = extractor.extract(generated_sql)
        
        table_score = StructuralMatch._calculate_score(expected["tables"], generated["tables"])
        
        # Joins are list of dicts, convert to hashable tuples for set comparison
        def hashable_joins(joins):
            return [tuple(sorted([j["table_a"].lower(), j["table_b"].lower()])) for j in joins]
            
        join_score = StructuralMatch._calculate_score(
            hashable_joins(expected["joins"]), 
            hashable_joins(generated["joins"])
        )
        
        filter_score = StructuralMatch._calculate_score(expected["filters"], generated["filters"])
        aggregation_score = StructuralMatch._calculate_score(expected["aggregations"], generated["aggregations"])
        
        # Calculate overall score as requested
        overall = (table_score + join_score + filter_score + aggregation_score) / 4.0
        
        return {
            "table_score": round(table_score, 3),
            "join_score": round(join_score, 3),
            "filter_score": round(filter_score, 3),
            "aggregation_score": round(aggregation_score, 3),
            "overall": round(overall, 3)
        }
