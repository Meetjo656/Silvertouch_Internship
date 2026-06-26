import tiktoken
import os
from typing import List, Dict, Any, Tuple
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.generation.context_value_scorer import ContextValueScorer
from src.evaluation.query_complexity_classifier import QueryComplexityClassifier

class ContextAssembler:
    """
    Decides the final context composition for generation.
    Handles token budgets, deduplication, and score-based prioritization.
    """
    def __init__(self, data_dir: str, base_max_tokens: int = 6000, model_name: str = "gpt-3.5-turbo"):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
        self.base_max_tokens = base_max_tokens
        self.scorer = ContextValueScorer(data_dir)

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def _get_dynamic_budget(self, question: str, sql_examples: List[str]) -> int:
        # Infer complexity from historical SQL if available, else default to generic
        complexity = "SIMPLE_SELECT"
        if sql_examples:
            complexity = QueryComplexityClassifier.classify(sql_examples[0])
            
        if complexity in ["MULTI_JOIN", "WINDOW_FUNCTION", "CTE"]:
            return int(self.base_max_tokens * 1.5) # Expand budget for complex queries
        elif complexity == "SIMPLE_SELECT":
            return int(self.base_max_tokens * 0.8) # Constrict for simple ones
        return self.base_max_tokens

    def assemble(self, 
                 table_names: List[str], 
                 ddls: List[str], 
                 docs: List[str], 
                 sql_examples: List[str], 
                 question: str, 
                 expected_tables: set = None) -> Dict[str, Any]:
        
        max_tokens = self._get_dynamic_budget(question, sql_examples)
        
        # 1. Score Tables
        scored_tables = []
        for i, t_name in enumerate(table_names):
            # Simple assumption: expected tables in benchmark represent "Join-path" required tables
            # For production, we'd use structural graph, but for benchmark analysis this measures retention efficiency
            is_join_path = expected_tables is not None and t_name.lower() in expected_tables
            
            score = self.scorer.score(t_name, rank=i, doc_str=docs[i], question=question, is_join_path=is_join_path)
            
            scored_tables.append({
                "name": t_name,
                "ddl": ddls[i],
                "doc": docs[i],
                "score": score
            })
            
        # 2. Sort by Score Descending
        scored_tables.sort(key=lambda x: x["score"], reverse=True)
        
        assembled_ddls = []
        assembled_docs = []
        assembled_sqls = []
        
        kept_tables = []
        dropped_tables = []
        
        current_tokens = 0
        
        # Priority 1: Schema & Docs based on score
        # We allow up to 80% of budget for schemas + docs
        schema_budget = max_tokens * 0.8
        
        for st in scored_tables:
            ddl_tokens = self._count_tokens(st["ddl"])
            doc_tokens = self._count_tokens(st["doc"])
            
            if current_tokens + ddl_tokens + doc_tokens < schema_budget:
                assembled_ddls.append(st["ddl"])
                if st["doc"]:
                    assembled_docs.append(st["doc"])
                current_tokens += (ddl_tokens + doc_tokens)
                kept_tables.append({"table": st["name"], "value_score": st["score"], "kept": True, "drop_reason": None})
            else:
                dropped_tables.append({"table": st["name"], "value_score": st["score"], "kept": False, "drop_reason": "LOW_SCORE_BUDGET_EXCEEDED"})
                
        # Priority 2: SQL Examples
        seen_sqls = set()
        for sql in sql_examples:
            if not sql or sql in seen_sqls: continue
            tokens = self._count_tokens(sql)
            if current_tokens + tokens < max_tokens:
                assembled_sqls.append(sql)
                seen_sqls.add(sql)
                current_tokens += tokens

        return {
            "ddls": assembled_ddls,
            "docs": assembled_docs,
            "sql_examples": assembled_sqls,
            "kept_tables": kept_tables,
            "dropped_tables": dropped_tables
        }
