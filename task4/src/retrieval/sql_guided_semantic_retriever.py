import os
import json
from typing import List, Dict, Any
from src.utils.sql_structure_extractor import SQLStructureExtractor
from src.retrieval.schema_matcher import SchemaMatcher

class SQLGuidedSemanticRetriever:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.extractor = SQLStructureExtractor()
        
        catalog_path = os.path.join(data_dir, "processed", "schema_signature_catalog.json")
        self.matcher = SchemaMatcher(catalog_path)
        
        join_graph_path = os.path.join(data_dir, "processed", "join_graph.json")
        self.join_graph = {}
        if os.path.exists(join_graph_path):
            try:
                with open(join_graph_path, 'r', encoding='utf-8') as f:
                    self.join_graph = json.load(f)
            except Exception as e:
                print(f"Failed to load join graph: {e}")

    def _expand_candidates(self, candidates: List[str], min_join_frequency: int = 25) -> List[str]:
        expanded = set(candidates)
        for table in candidates:
            if table in self.join_graph:
                for joined_table, freq in self.join_graph[table].items():
                    if freq >= min_join_frequency:
                        expanded.add(joined_table)
        return list(expanded)

    def retrieve(self, question: str, retrieved_sql_examples: List[str], sql_similarity_scores: List[float], top_k: int = 5) -> Dict[str, Any]:
        """
        Orchestrates SQL-Guided Semantic Retrieval.
        Returns a dictionary containing candidate tables, confidence level, and metadata.
        """
        max_similarity = max(sql_similarity_scores) if sql_similarity_scores else 0.0
        candidate_tables_list = []
        metadata = {
            "confidence": "Low",
            "max_similarity": max_similarity,
            "patterns_extracted": False,
            "expanded": False
        }

        if max_similarity >= 0.85:
            metadata["confidence"] = "High"
        elif max_similarity >= 0.65:
            metadata["confidence"] = "Medium"
        else:
            metadata["confidence"] = "Low"

        if metadata["confidence"] in ["High", "Medium"]:
            # Extract Query Patterns from Retrieved SQL
            all_patterns = {
                "tables": [],
                "columns": [],
                "joins": [],
                "aggregations": [],
                "filters": []
            }
            for sql in retrieved_sql_examples:
                patterns = self.extractor.extract(sql)
                for key in all_patterns:
                    if key in patterns:
                        for item in patterns[key]:
                            if item not in all_patterns[key]:
                                all_patterns[key].append(item)
            
            metadata["patterns_extracted"] = True
            
            # Schema Matching against Current Authoritative Schema Signature Catalog
            matched_tables = self.matcher.match(question, all_patterns, top_k=top_k)
            for t in matched_tables:
                if t not in candidate_tables_list:
                    candidate_tables_list.append(t)
            
            # Join Graph Expansion for Medium Confidence
            if metadata["confidence"] == "Medium":
                expanded_tables = self._expand_candidates(list(candidate_tables_list), min_join_frequency=25)
                for t in expanded_tables:
                    if t not in candidate_tables_list:
                        candidate_tables_list.append(t)
                metadata["expanded"] = True
                
        else:
            # Low Confidence Pipeline: Fallback to pure Schema Signature Search
            matched_tables = self.matcher.match(question, {}, top_k=top_k)
            for t in matched_tables:
                if t not in candidate_tables_list:
                    candidate_tables_list.append(t)

        return {
            "candidate_tables": candidate_tables_list,
            "metadata": metadata
        }
