from collections import defaultdict

class ModuleRouter:
    def __init__(self):
        # Direct intent keywords to boost specific modules
        self.intent_keywords = {
            'RANKINGS': ['rank', 'ranking', 'score', 'position'],
            'GP_COVERAGE': ['coverage', 'covered', 'gp', 'village coverage'],
            'AT_A_GLANCE': ['glance', 'summary', 'overview', 'dashboard']
        }
        
    def predict(self, query, extracted_raw_entities, threshold=0.4):
        """
        Predicts the target module based on extracted catalog entities and intent keywords.
        """
        module_scores = defaultdict(float)
        query_lower = query.lower()
        
        # 1. Score from entities
        for entity in extracted_raw_entities:
            modules = entity.get('modules', [])
            score_per_module = 2.0 / max(len(modules), 1) # distribute points if multiple
            for mod in modules:
                module_scores[mod] += score_per_module
                
        # 2. Score from intent keywords
        for mod, keywords in self.intent_keywords.items():
            for kw in keywords:
                if kw in query_lower:
                    module_scores[mod] += 3.0
                    
        if not module_scores:
            return [{"module": "LEGACY", "confidence": 1.0}] # Fallback
            
        # Normalize scores to confidence (0.0 to 1.0)
        max_score = max(module_scores.values())
        predictions = []
        for mod, score in module_scores.items():
            confidence = score / max_score
            if confidence >= threshold:
                predictions.append({"module": mod, "confidence": round(confidence, 2)})
                
        return sorted(predictions, key=lambda x: x['confidence'], reverse=True)
