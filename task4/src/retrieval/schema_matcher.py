import os
import json
import re
from typing import List, Dict, Any

class SchemaMatcher:
    def __init__(self, catalog_path: str):
        self.catalog = []
        if os.path.exists(catalog_path):
            try:
                with open(catalog_path, 'r', encoding='utf-8') as f:
                    self.catalog = json.load(f)
            except Exception as e:
                print(f"Failed to load schema catalog: {e}")

    def _tokenize(self, text: str) -> set:
        if not text: return set()
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        stop_words = {"the", "a", "an", "is", "of", "and", "in", "to", "for", "with", "on", "at", "by", "from", "chatbot", "private", "public", "master", "table", "id", "name"}
        return {t for t in tokens if t not in stop_words and len(t) > 2}

    def match(self, question: str, sql_patterns: Dict[str, Any], top_k: int = 5) -> List[str]:
        """
        Resolves relevant current tables using Question and Retrieved SQL patterns.
        sql_patterns example:
        {
            "tables": [...],
            "joins": [...],
            "columns": [...],
            "aggregations": [...],
            "filters": [...]
        }
        """
        question_tokens = self._tokenize(question)
        
        # Extract signals from historical SQL patterns
        sql_tokens = set()
        if "tables" in sql_patterns:
            for t in sql_patterns["tables"]:
                sql_tokens.update(self._tokenize(t))
        if "columns" in sql_patterns:
            for c in sql_patterns["columns"]:
                sql_tokens.update(self._tokenize(c))

        combined_signals = question_tokens.union(sql_tokens)
        
        scores = []
        for entry in self.catalog:
            table_name = entry.get("table_name", "")
            signals = set(entry.get("signals", []))
            
            # 1. Signal Overlap (semantic/documentation/module overlap surrogate)
            overlap_score = len(combined_signals.intersection(signals))
            
            # 2. SQL Pattern Overlap
            pattern_score = 0
            if "tables" in sql_patterns and table_name in sql_patterns["tables"]:
                # Historical SQL used exactly this table name
                pattern_score += 5.0
                
            col_overlap = 0
            if "columns" in sql_patterns:
                table_cols = set(entry.get("columns", []))
                col_overlap = len(set(sql_patterns["columns"]).intersection(table_cols))
            
            # Weight the scores
            total_score = overlap_score + pattern_score + (col_overlap * 2.0)
            scores.append((total_score, table_name))
            
        # Sort by highest score
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k table names that have at least some matching signal
        return [table for score, table in scores[:top_k] if score > 0]
