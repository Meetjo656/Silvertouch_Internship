import os
import json

class DocRetriever:
    def __init__(self, raw_data_dir):
        # Load all docs
        legacy_dir = os.path.join(raw_data_dir, 'Private Data', 'qdrant_export_20260615_062842', 'LEGACY')
        doc_path = os.path.join(legacy_dir, 'doc.json')
        with open(doc_path, 'r', encoding='utf-8') as f:
            self.docs = json.load(f)
            
    def retrieve(self, predicted_modules, extracted_raw_entities):
        """
        Retrieves documentation relevant to the predicted modules and extracted entities.
        """
        # Get module names
        target_modules = [m['module'] for m in predicted_modules]
        
        relevant_docs = []
        for item in self.docs:
            payload = item['payload']
            doc_module = payload.get('module')
            
            if doc_module in target_modules:
                # Pass full payload data for ranking
                relevant_docs.append(payload)
                
        # Deduplicate and return
        unique_docs = []
        seen = set()
        for d in relevant_docs:
            c = d.get('content', d.get('documentation', ''))
            if c not in seen:
                seen.add(c)
                unique_docs.append(d)
        return unique_docs
