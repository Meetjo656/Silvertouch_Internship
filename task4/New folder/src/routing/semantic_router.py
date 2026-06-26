import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

class SemanticModuleRouter:
    """
    Router A: Representative Embedding Router
    """
    def __init__(self, embeddings_dir, model_name="all-MiniLM-L6-v2"):
        npy_path = os.path.join(embeddings_dir, 'module_embeddings.npy')
        json_path = os.path.join(embeddings_dir, 'module_metadata.json')
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Load precomputed embeddings and metadata
        self.corpus_embeddings = torch.tensor(np.load(npy_path)).to(self.device)
        with open(json_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
    def predict(self, query, threshold=0.3):
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute cosine similarities
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # Aggregate maximum score per module
        module_scores = {}
        for idx, score in enumerate(cos_scores):
            mod = self.metadata[idx]['module']
            sim = score.item()
            if mod not in module_scores or sim > module_scores[mod]:
                module_scores[mod] = sim
                
        # Filter and format
        results = []
        for mod, score in module_scores.items():
            if score >= threshold:
                results.append({"module": mod, "confidence": round(score, 4)})
                
        # Sort descending
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        # Fallback if nothing passes threshold
        if not results:
            results = [{"module": "LEGACY", "confidence": 1.0}]
            
        return results
