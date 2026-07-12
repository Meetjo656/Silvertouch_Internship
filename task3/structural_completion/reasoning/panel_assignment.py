from typing import List, Dict, Any, Tuple
from ..graph import PanelGraph
from ..panel import Panel
from .layout_rules import LayoutRules
from .matcher import Matcher

class PanelAssignment:
    def __init__(self, rules: LayoutRules):
        self.rules = rules

    def assign(self, graph: PanelGraph, candidates: List[Dict[str, Any]], missing_classes: List[str]) -> Tuple[List[Panel], List[Dict[str, Any]]]:
        """
        Assigns candidates to missing classes based on layout rules and matcher scores.
        Returns a tuple of (inferred_panels, unused_candidates).
        """
        inferred = []
        used_indices = set()
        
        for m_class in missing_classes:
            allowed_neighbors = self.rules.get_allowed_neighbors(m_class)
            
            # Find all anchors for this missing class
            anchors = []
            for n_class in allowed_neighbors:
                anchors.extend(graph.get_panels_by_class(n_class))
                
            best_score = 0
            best_cand_idx = -1
            best_evidence = []
            
            for i, cand in enumerate(candidates):
                if i in used_indices:
                    continue
                    
                # Score candidate against all anchors
                cand_score = 0
                for anchor in anchors:
                    # Very simple relation heuristic
                    relation = "left_of" if cand["features"]["width"] < cand["features"]["height"] else "above"
                    score = Matcher.calculate_match_score(cand, anchor, relation)
                    if score > cand_score:
                        cand_score = score
                        
                if cand_score > best_score:
                    best_score = cand_score
                    best_cand_idx = i
                    best_evidence = [f"Best match score: {cand_score:.1f}"]
                    
            if best_cand_idx != -1 and best_score > 30: # Minimum threshold
                used_indices.add(best_cand_idx)
                cand = candidates[best_cand_idx]
                
                # Expand confidence object
                confidence = {
                    "score": min(1.0, best_score / 100.0),
                    "geometry": 0.8 if cand["features"]["touches_artwork_edge"] else 0.5,
                    "layout": 1.0,
                    "adjacency": best_score / 100.0,
                    "shape": 1.0,
                    "evidence": ["rectangle detected"] + best_evidence
                }
                
                panel = Panel(
                    class_name=m_class,
                    bbox=cand["bbox"],
                    confidence=confidence,
                    source="geometry_reasoner"
                )
                inferred.append(panel)
                
        unused = [c for i, c in enumerate(candidates) if i not in used_indices]
        return inferred, unused
