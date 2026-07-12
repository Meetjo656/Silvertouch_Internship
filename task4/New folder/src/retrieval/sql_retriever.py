import os
import json
from src.retrieval.embedding_retriever import EmbeddingRetriever

class SQLRetriever:
    def __init__(self, raw_data_dir):
        legacy_dir = os.path.join(raw_data_dir, 'Private Data', 'qdrant_export_20260615_062842', 'LEGACY')
        sql_path = os.path.join(legacy_dir, 'sql.json')
        
        with open(sql_path, 'r', encoding='utf-8') as f:
            raw_sql = json.load(f)
            
        self.corpus = [item['payload'] for item in raw_sql]
        self.retriever = EmbeddingRetriever(model_name="all-MiniLM-L6-v2").fit(self.corpus, text_key='question')
        
    def retrieve(self, query, predicted_modules, top_k=3):
        """
        Uses EmbeddingRetriever to find semantically similar historical questions.
        Experiment 2: Filter by top 3 modules.
        """
        if not predicted_modules:
            return []
            
        # Get top 3 module names
        top_modules = [m['module'] for m in predicted_modules[:3]]
        
        # Search with filter targeting any of the top 3 modules
        results = self.retriever.search(query, top_k=top_k, filter_dict={"module": top_modules})
        
        formatted_results = []
        for r in results:
            formatted_results.append({
                "question": r['doc'].get('question', ''),
                "sql": r['doc'].get('sql', ''),
                "similarity_score": round(r['score'], 2)
            })
            
        return formatted_results
