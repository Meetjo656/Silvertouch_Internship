import os
import json

class ContextValueScorer:
    """
    Assigns an importance score to every retrieved table to intelligently manage context loss
    when the token budget is exceeded.
    """
    def __init__(self, data_dir: str):
        sig_path = os.path.join(data_dir, "processed", "schema_signature_catalog.json")
        self.signatures = {}
        if os.path.exists(sig_path):
            with open(sig_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    self.signatures[item["table_name"].lower()] = item.get("signals", [])

        
    def classify_table(self, table_name: str) -> str:
        """
        Uses schema_signature_catalog.json signals to classify the table.
        Strictly prohibits table-name suffix matching.
        """
        signals = [str(s).lower() for s in self.signatures.get(table_name.lower(), [])]
        
        # Geography Detection
        if any(w in signals for w in ["district", "state", "village", "region", "geography", "location"]):
            return "GEOGRAPHY"
            
        # Ranking Detection
        if any(w in signals for w in ["rank", "ranking", "score", "position"]):
            return "RANKING"
            
        # Lookup Detection
        if any(w in signals for w in ["master", "reference", "mapping"]):
            return "LOOKUP"
            
        return "GENERAL"

    def score(self, table_name: str, rank: int, doc_str: str, question: str, is_join_path: bool = False) -> float:
        """
        Calculates the base retention score for a table.
        """
        # Retrieval Score Approximation (Top result = 1.0, decays slightly)
        retrieval_score = max(0.1, 1.0 - (rank * 0.05))
        
        # Documentation Score
        doc_score = 1.0 if doc_str and len(doc_str.strip()) > 10 else 0.5
        
        # Question Overlap
        t_lower = table_name.lower()
        q_lower = question.lower()
        overlap_score = 1.0
        if any(part in q_lower for part in t_lower.split('_') if len(part) > 3):
            overlap_score = 1.2
            
        base_score = retrieval_score * doc_score * overlap_score
        
        # Critical Table Protection Boosts
        t_type = self.classify_table(table_name)
        if t_type in ["GEOGRAPHY", "RANKING"]:
            base_score += 0.15
            
        if is_join_path:
            base_score += 0.15
            
        return round(min(1.0, base_score), 3)
