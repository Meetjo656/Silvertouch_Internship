import os
import sys
import re
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.generation.column_selector import ColumnSelector
from src.schema.schema_registry import SchemaRegistry
from src.semantic.business_semantics_resolver import BusinessSemanticsResolver
from src.semantic.join_path_resolver import JoinPathResolver

class ContextOptimizer:
    """
    Filters high-recall retrieval output down to a high-precision generation context.
    """
    def __init__(self, registry: SchemaRegistry, data_dir: str):
        self.registry = registry
        self.resolver = BusinessSemanticsResolver(data_dir)
        self.join_resolver = JoinPathResolver(data_dir, registry)

    def _score_table(self, table_name: str, question: str, docs: List[str], sql_examples: List[str], retrieval_score: float) -> float:
        """
        Scores a table's relevance for generation.
        """
        t_lower = table_name.lower()
        q_lower = question.lower()
        
        # 1. Question overlap
        # Check if table name parts are in the question
        t_parts = t_lower.split('_')
        q_overlap_score = 0.5
        for part in t_parts:
            if part in q_lower and len(part) > 3:
                q_overlap_score += 0.5
                
        # 2. Schema signal overlap (Does the table have columns that match the question?)
        schema_overlap = 1.0
        if self.registry.table_exists(t_lower):
            cols = self.registry.tables[t_lower]["columns"]
            for col in cols:
                c_parts = col.lower().split('_')
                if any(p in q_lower for p in c_parts if len(p) > 3):
                    schema_overlap += 0.2
                    
        # 3. Join importance / Historical usage
        join_importance = 1.0
        sql_corpus = " ".join(sql_examples).lower()
        if t_lower in sql_corpus:
            join_importance += 2.0
            
        # 4. Documentation overlap
        doc_corpus = " ".join(docs).lower()
        if t_lower in doc_corpus:
            join_importance += 0.5
            
        return q_overlap_score * schema_overlap * join_importance * retrieval_score

    def optimize(self, question: str, retrieved_tables: List[str], retrieved_docs: List[str], retrieved_sql_examples: List[str], retrieved_scores: List[float] = None) -> Dict[str, Any]:
        if not retrieved_scores:
            # Fallback if scores not provided, distribute sequentially
            retrieved_scores = [1.0 - (i * 0.05) for i in range(len(retrieved_tables))]
            
        # 1. Score and select Top 3 tables
        scored_tables = []
        for i, t in enumerate(retrieved_tables):
            score = self._score_table(t, question, retrieved_docs, retrieved_sql_examples, retrieved_scores[i])
            scored_tables.append((t, score))
            
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        generation_tables = [t[0] for t in scored_tables[:3]]
        
        # 2. Extract and prune columns
        pruned_ddls = []
        semantic_rules = []
        total_retrieved_cols = 0
        total_gen_cols = 0
        
        for t in generation_tables:
            if self.registry.table_exists(t):
                all_cols = self.registry.tables[t.lower()]["columns"]
                total_retrieved_cols += len(all_cols)
                
                selected_cols = ColumnSelector.select(t, all_cols, question, " ".join(retrieved_docs), retrieved_sql_examples)
                total_gen_cols += len(selected_cols)
                
                # Fetch business semantics for these columns
                table_rules = self.resolver.resolve_for_table(t, selected_cols)
                for rule in table_rules:
                    if rule not in semantic_rules:
                        semantic_rules.append(rule)
                
                # Format pruned DDL
                pruned_ddl = f"Table: {t}\nRelevant Columns: {', '.join(selected_cols)}"
                pruned_ddls.append(pruned_ddl)
            else:
                pruned_ddls.append(f"Table: {t}\nRelevant Columns: Unknown")
                
        # Calculate unselected table columns for stats
        for t in retrieved_tables:
            if t not in generation_tables and self.registry.table_exists(t):
                total_retrieved_cols += len(self.registry.tables[t.lower()]["columns"])

        # 3. Enhance SQL Pattern Selection
        generation_sqls = []
        if retrieved_sql_examples:
            # Simple heuristic: prefer SQL examples that mention columns we extracted semantics for
            scored_sqls = []
            for sql in retrieved_sql_examples:
                sql_score = 0
                for rule in semantic_rules:
                    col_name = rule.split(':')[0].strip()
                    if col_name in sql.lower():
                        sql_score += 1
                scored_sqls.append((sql, sql_score))
                
            # Stable sort to preserve retriever's vector similarity as primary tie-breaker
            scored_sqls.sort(key=lambda x: x[1], reverse=True)
            generation_sqls = [scored_sqls[0][0]]
            
        generation_docs = retrieved_docs[:1] if retrieved_docs else []
        
        # 4. Resolve Join Paths
        join_paths = self.join_resolver.resolve_join_path(generation_tables)
        
        return {
            "generation_tables": generation_tables,
            "generation_ddls": pruned_ddls,
            "business_semantics": semantic_rules,
            "join_paths": join_paths,
            "generation_docs": generation_docs,
            "generation_sql_examples": generation_sqls,
            "metrics": {
                "retrieved_tables": len(retrieved_tables),
                "generation_tables": len(generation_tables),
                "retrieved_columns": total_retrieved_cols,
                "generation_columns": total_gen_cols,
                "retrieved_docs": len(retrieved_docs),
                "generation_docs": len(generation_docs),
                "retrieved_sql_examples": len(retrieved_sql_examples),
                "generation_sql_examples": len(generation_sqls)
            }
        }
