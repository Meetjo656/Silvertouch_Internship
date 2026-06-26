from typing import Dict, Any

class FailureClassifier:
    """
    Classifies generation failures into distinct bottleneck categories.
    """
    
    @staticmethod
    def classify(
        valid: bool,
        exact_match: bool,
        hallucinated_tables: list,
        hallucinated_columns: list,
        struct_scores: Dict[str, float],
        candidate_recall: bool = True
    ) -> str:
        
        # If it's functionally correct, no failure
        if exact_match or struct_scores.get("overall", 0.0) >= 0.99:
            return "NONE"
            
        # 1. Did we even retrieve the required tables?
        if not candidate_recall:
            return "RETRIEVAL_FAILURE"
            
        # 2. Did the SQL parse correctly?
        if not valid:
            return "SYNTAX_FAILURE"
            
        # 3. Did it invent schema?
        if hallucinated_tables or hallucinated_columns:
            return "HALLUCINATION_FAILURE"
            
        # 4. Structural mismatch hierarchy
        if struct_scores.get("table_score", 1.0) < 0.99:
            return "TABLE_SELECTION_FAILURE"
            
        if struct_scores.get("join_score", 1.0) < 0.99:
            return "JOIN_FAILURE"
            
        if struct_scores.get("filter_score", 1.0) < 0.99:
            return "FILTER_FAILURE"
            
        if struct_scores.get("aggregation_score", 1.0) < 0.99:
            return "AGGREGATION_FAILURE"
            
        # If we failed but can't isolate to these buckets (e.g. window functions, math operations)
        return "UNKNOWN"
