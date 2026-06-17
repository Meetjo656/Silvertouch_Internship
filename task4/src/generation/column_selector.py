import re
from typing import List

class ColumnSelector:
    """
    Selects only the most relevant columns from a table schema to prevent LLM context overload.
    Retains all structural keys natively, and filters data columns based on semantic overlap with the prompt.
    """
    @staticmethod
    def select(table_name: str, all_columns: List[str], question: str, docs: str, sql_examples: List[str]) -> List[str]:
        selected = set()
        q_lower = question.lower()
        q_tokens = set(re.findall(r'\b\w+\b', q_lower))
        
        doc_lower = docs.lower()
        sql_corpus = " ".join(sql_examples).lower()
        
        for col in all_columns:
            col_lower = col.lower()
            
            # 1. ALWAYS retain structural/foreign keys
            if col_lower == "id" or col_lower.endswith("_id") or col_lower.startswith("id_"):
                selected.add(col)
                continue
                
            # 2. ALWAYS retain status flags (crucial for WHERE clauses in this domain)
            if "status" in col_lower or "is_" in col_lower or "has_" in col_lower:
                selected.add(col)
                continue
                
            # 3. Historical SQL usage (Join/Filter columns implicitly covered if they were historically used)
            if col_lower in sql_corpus:
                selected.add(col)
                continue
                
            # 4. Question Overlap
            col_tokens = col_lower.split('_')
            # Check individual tokens
            if any(t in q_tokens for t in col_tokens if len(t) > 2):
                selected.add(col)
                continue
                
            # Check full phrase mapping
            if col_lower.replace('_', ' ') in q_lower or col_lower.replace('_', '') in q_lower:
                selected.add(col)
                continue
                
            # 5. Documentation Overlap
            if col_lower in doc_lower:
                selected.add(col)
                continue
                
        # Preserve original schema order
        return [c for c in all_columns if c in selected]
