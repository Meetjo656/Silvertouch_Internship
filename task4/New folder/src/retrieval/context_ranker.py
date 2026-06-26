import re
import torch
from sentence_transformers import util

class ContextRanker:
    def __init__(self, embedding_model, w1=0.50, w2=0.20, w3=0.15, w4=0.10, w5=0.05):
        self.embed_model = embedding_model
        self.w1 = w1 # semantic_similarity
        self.w2 = w2 # entity_match_score / business_term_overlap
        self.w3 = w3 # module_match_score
        self.w4 = w4 # geography_match_score
        self.w5 = w5 # role_relevance_score

    def rank_ddls(self, query, candidate_tables, predicted_modules, retrieved_sqls, entities, user_context=None):
        """
        Ranks candidate DDL table names based on Entity Matches, Module Matches, and SQL Table References.
        """
        ranked = []
        target_modules = [m['module'] for m in predicted_modules] if predicted_modules else []
        
        # Extract table references from retrieved SQLs
        sql_text = " ".join([sql.lower() for sql in retrieved_sqls])
        
        for table in candidate_tables:
            score = 0.0
            table_lower = table.lower()
            
            # 1. SQL Table References (Massive Boost)
            # If the table name is literally in the retrieved SQL examples, it's highly relevant.
            if table_lower in sql_text:
                score += 1.0
                
            # 2. Entity Matches (W2)
            # Check if any extracted entity is in the table name
            entity_match = 0.0
            for k, v_list in entities.items():
                for v in v_list:
                    if str(v).lower() in table_lower:
                        entity_match = 1.0
                        break
            score += entity_match * self.w2
            
            # 3. Geo Match (W4)
            if user_context and user_context.district:
                if user_context.district.lower() in table_lower:
                    score += self.w4
            if "district" in table_lower or "state" in table_lower:
                score += self.w4
                
            ranked.append({"table": table, "score": score})
            
        ranked = sorted(ranked, key=lambda x: x['score'], reverse=True)
        return [item['table'] for item in ranked]

    def rank_docs(self, query, candidate_docs_raw, predicted_modules, entities, user_context=None):
        """
        Implements Hybrid Ranking for Documentation:
        doc_score = (semantic_similarity * 0.5) + (business_term_overlap * 0.3) + (module_match * 0.2)
        Also incorporates User Context (Role & Geo) boosts.
        candidate_docs_raw is a list of dicts: {'content': ..., 'module': ..., 'business_terms': ...}
        """
        if not candidate_docs_raw:
            return []
            
        # 1. Compute Semantic Similarity for all docs
        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        doc_texts = [d.get('content', '') for d in candidate_docs_raw]
        doc_embs = self.embed_model.encode(doc_texts, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, doc_embs)[0].cpu().tolist()
        
        target_modules = [m['module'] for m in predicted_modules] if predicted_modules else []
        
        # Flatten entities for overlap check
        flat_entities = []
        for v_list in entities.values():
            flat_entities.extend([str(v).lower() for v in v_list])
            
        ranked = []
        for idx, doc in enumerate(candidate_docs_raw):
            content = doc.get('content', '').lower()
            doc_module = doc.get('module', '')
            doc_terms = doc.get('business_terms', [])
            if isinstance(doc_terms, str): doc_terms = [doc_terms]
            doc_terms = [str(t).lower() for t in doc_terms]
            
            semantic_score = max(0.0, cos_scores[idx]) # Relu to avoid negative scores pulling it down
            
            # Business Term / Entity Overlap
            term_overlap = 0.0
            for term in flat_entities:
                if term in doc_terms or term in content:
                    term_overlap = 1.0
                    break
                    
            module_match = 1.0 if doc_module in target_modules else 0.0
            
            base_score = (semantic_score * 0.5) + (term_overlap * 0.3) + (module_match * 0.2)
            
            # Role & Geo Boosts
            geo_boost = 0.0
            role_boost = 0.0
            
            if user_context:
                if user_context.district and user_context.district.lower() in content:
                    geo_boost = self.w4
                if user_context.role and "district" in user_context.role.lower() and "district" in content:
                    role_boost = self.w5
                if user_context.role and "state" in user_context.role.lower() and "state" in content:
                    role_boost = self.w5
                    
            final_score = base_score + geo_boost + role_boost
            
            ranked.append({
                "content": doc.get('content', ''),
                "score": final_score
            })
            
        ranked = sorted(ranked, key=lambda x: x['score'], reverse=True)
        return [item['content'] for item in ranked]
