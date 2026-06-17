import os
import json
from typing import List, Dict, Tuple, Set

class JoinPathResolver:
    """
    Deduces exact SQL join column relationships (e.g., table_a.id = table_b.table_a_id)
    between a set of candidate tables based on historical join graph and schema structures.
    """
    def __init__(self, data_dir: str, registry):
        self.registry = registry
        self.graph_path = os.path.join(data_dir, "processed", "join_graph.json")
        self.join_graph = {}
        
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'r', encoding='utf-8') as f:
                self.join_graph = json.load(f)
                
    def resolve_join_path(self, tables: List[str]) -> List[str]:
        """
        Finds the valid join relationships among the provided tables.
        Returns a list of strings like: "table_a.column_x = table_b.column_y"
        """
        tables = [t.lower() for t in tables if self.registry.table_exists(t)]
        if len(tables) < 2:
            return []
            
        resolved_edges = set()
        relationships = []
        
        # O(N^2) over tables to find edges
        for i in range(len(tables)):
            for j in range(i + 1, len(tables)):
                t1 = tables[i]
                t2 = tables[j]
                
                # Check if there is historical evidence or just attempt schema match
                has_history = False
                if t1 in self.join_graph and t2 in self.join_graph[t1]:
                    has_history = True
                elif t2 in self.join_graph and t1 in self.join_graph[t2]:
                    has_history = True
                    
                # Deduce columns
                t1_cols = self.registry.tables[t1].get("columns", [])
                t2_cols = self.registry.tables[t2].get("columns", [])
                
                edge = self._find_joining_columns(t1, t1_cols, t2, t2_cols)
                if edge:
                    # If there's an edge, add it
                    resolved_edges.add(tuple(sorted([t1, t2])))
                    relationships.append(f"{edge[0]}.{edge[1]} = {edge[2]}.{edge[3]}")
                elif has_history:
                    # Historical graph says they join, but simple ID heuristic failed.
                    # Fallback to string matching IDs if they don't follow exact naming conventions.
                    fallback_edge = self._fallback_joining_columns(t1, t1_cols, t2, t2_cols)
                    if fallback_edge:
                        resolved_edges.add(tuple(sorted([t1, t2])))
                        relationships.append(f"{fallback_edge[0]}.{fallback_edge[1]} = {fallback_edge[2]}.{fallback_edge[3]}")
                        
        return list(set(relationships))

    def _find_joining_columns(self, t1: str, t1_cols: List[str], t2: str, t2_cols: List[str]) -> Tuple[str, str, str, str]:
        # Rule 1: t1.id = t2.t1_id (or similar suffix)
        t1_base = t1.replace("chatbot_", "").replace("_master", "").replace("_private", "")
        t2_base = t2.replace("chatbot_", "").replace("_master", "").replace("_private", "")
        
        # Check t1.id = t2.[t1]_id
        if "id" in t1_cols:
            for c in t2_cols:
                if c == f"{t1_base}_id" or c == f"{t1}_id":
                    return (t1, "id", t2, c)
                    
        # Check t2.id = t1.[t2]_id
        if "id" in t2_cols:
            for c in t1_cols:
                if c == f"{t2_base}_id" or c == f"{t2}_id":
                    return (t2, "id", t1, c)
                    
        # Rule 2: Shared exact _id column (e.g. both have district_id)
        # Avoid linking on just "id" = "id"
        shared_id_cols = set(t1_cols).intersection(set(t2_cols))
        for c in shared_id_cols:
            if c.endswith("_id") and c != "id":
                return (t1, c, t2, c)
                
        return None
        
    def _fallback_joining_columns(self, t1: str, t1_cols: List[str], t2: str, t2_cols: List[str]) -> Tuple[str, str, str, str]:
        # If history says they join, look for any _id in t2 that shares a token with t1
        t1_tokens = set(t1.split('_'))
        for c in t2_cols:
            if c.endswith("_id"):
                c_tokens = set(c.replace("_id", "").split('_'))
                if t1_tokens.intersection(c_tokens):
                    # Guess this is the key
                    if "id" in t1_cols:
                        return (t1, "id", t2, c)
                        
        t2_tokens = set(t2.split('_'))
        for c in t1_cols:
            if c.endswith("_id"):
                c_tokens = set(c.replace("_id", "").split('_'))
                if t2_tokens.intersection(c_tokens):
                    if "id" in t2_cols:
                        return (t2, "id", t1, c)
                        
        return None
