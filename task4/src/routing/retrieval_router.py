from collections import Counter

class RetrievalModuleRouter:
    """
    Router B: Retrieval-Driven Module Router
    Leverages K-Nearest Neighbors of historical questions to vote on the module.
    """
    def __init__(self, embedding_retriever):
        # We inject a pre-fit EmbeddingRetriever to save GPU memory
        self.retriever = embedding_retriever

    def predict(self, query, top_k_neighbors=20, threshold=0.1):
        # Search the entire corpus without module filtering
        results = self.retriever.search(query, top_k=top_k_neighbors, filter_dict=None)
        
        if not results:
            return [{"module": "LEGACY", "confidence": 1.0}]
            
        # Tally the modules from the nearest neighbors
        # We can do a weighted tally based on similarity score, or just a raw count.
        # Let's do a weighted tally: sum of similarity scores for each module.
        module_scores = {}
        for r in results:
            mod = r['doc'].get('module', 'UNKNOWN')
            sim = r['score']
            if mod not in module_scores:
                module_scores[mod] = 0.0
            module_scores[mod] += sim
            
        # Normalize scores
        total_score = sum(module_scores.values())
        if total_score == 0:
            return [{"module": "LEGACY", "confidence": 1.0}]
            
        final_results = []
        for mod, total_sim in module_scores.items():
            conf = total_sim / total_score
            if conf >= threshold:
                final_results.append({"module": mod, "confidence": round(conf, 4)})
                
        # Sort descending
        final_results = sorted(final_results, key=lambda x: x['confidence'], reverse=True)
        
        # Fallback
        if not final_results:
            final_results = [{"module": "LEGACY", "confidence": 1.0}]
            
        return final_results
