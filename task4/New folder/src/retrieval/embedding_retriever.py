from src.retrieval.base import BaseRetriever
from sentence_transformers import SentenceTransformer, util
import torch

class EmbeddingRetriever(BaseRetriever):
    """
    Semantic Retriever using HuggingFace sentence-transformers.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Auto-detect GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.corpus_data = []
        self.corpus_embeddings = None

    def fit(self, corpus_dicts, text_key='text'):
        """
        Pre-computes embeddings for the corpus.
        """
        self.corpus_data = corpus_dicts
        corpus_texts = [doc.get(text_key, '') for doc in corpus_dicts]
        
        print(f"Encoding {len(corpus_texts)} documents into embeddings...")
        self.corpus_embeddings = self.model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)
        return self

    def search(self, query, top_k=5, filter_dict=None):
        if self.corpus_embeddings is None:
            raise ValueError("Retriever must be fitted before searching.")
            
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute cosine similarities
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # Sort scores in descending order
        top_results = torch.topk(cos_scores, k=len(self.corpus_data))
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            doc = self.corpus_data[idx]
            sim = score.item()
            
            # Apply metadata filters if provided (exact match)
            if filter_dict:
                match = True
                for k, v in filter_dict.items():
                    if isinstance(v, list) and isinstance(doc.get(k), list):
                        if not set(v).intersection(set(doc.get(k))):
                            match = False
                            break
                    elif isinstance(v, list):
                        if doc.get(k) not in v:
                            match = False
                            break
                    else:
                        if doc.get(k) != v:
                            match = False
                            break
                if not match:
                    continue
                    
            results.append({"doc": doc, "score": sim})
            if len(results) >= top_k:
                break
                
        return results
