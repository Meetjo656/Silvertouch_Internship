from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BaseRetriever(ABC):
    """
    Abstract base class for all semantic retrievers to ensure future-proofing.
    BM25, Embedding, or Hybrid retrievers must inherit from this.
    """
    
    @abstractmethod
    def fit(self, corpus):
        """Fit the retriever on a corpus of documents."""
        pass

    @abstractmethod
    def search(self, query, top_k=5, filter_dict=None):
        """Search the corpus for the query and return top_k results."""
        pass


class TFIDFRetriever(BaseRetriever):
    """
    Baseline TF-IDF Retriever for Phase 1.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.corpus_data = []

    def fit(self, corpus_dicts, text_key='text'):
        """
        Expects a list of dictionaries.
        text_key defines which key contains the text to be vectorized.
        """
        self.corpus_data = corpus_dicts
        corpus_texts = [doc.get(text_key, '') for doc in corpus_dicts]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
        return self

    def search(self, query, top_k=5, filter_dict=None):
        if self.tfidf_matrix is None:
            raise ValueError("Retriever must be fitted before searching.")
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get indices sorted by similarity (descending)
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            doc = self.corpus_data[idx]
            sim = float(similarities[idx])
            
            if sim == 0.0:
                continue
                
            # Apply metadata filters if provided (exact match)
            if filter_dict:
                match = True
                for k, v in filter_dict.items():
                    # Handle array overlaps (e.g. module filtering)
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
