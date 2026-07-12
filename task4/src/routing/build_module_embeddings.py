import sys
import os
import json
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

def build_module_embeddings():
    raw_sql_path = os.path.join(project_root, 'data', 'raw', 'Private Data', 'qdrant_export_20260615_062842', 'LEGACY', 'sql.json')
    with open(raw_sql_path, 'r', encoding='utf-8') as f:
        raw_sql = json.load(f)
        
    # Group by module, then by primary business term
    module_subgroups = defaultdict(lambda: defaultdict(list))
    
    for item in raw_sql:
        payload = item['payload']
        module = payload.get('module', 'UNKNOWN')
        if not module:
            continue
            
        b_terms = payload.get('business_terms', [])
        if isinstance(b_terms, str):
            b_terms = [b_terms]
            
        # Determine primary theme
        primary_theme = "General"
        if b_terms and len(b_terms) > 0:
            primary_theme = b_terms[0].title()
            
        module_subgroups[module][primary_theme].append(payload)
        
    # Create representative texts
    metadata = []
    texts_to_embed = []
    
    idx = 0
    for module, themes in module_subgroups.items():
        for theme, payloads in themes.items():
            # Gather all questions and terms for this theme
            questions = [p.get('question', '') for p in payloads if p.get('question')]
            # Take a sample of up to 10 questions to avoid massive strings
            sample_questions = list(set(questions))[:10]
            
            all_terms = set()
            for p in payloads:
                terms = p.get('business_terms', [])
                if isinstance(terms, list):
                    all_terms.update(terms)
                elif isinstance(terms, str):
                    all_terms.add(terms)
                    
            representative_text = f"Module: {module}. Theme: {theme}. Terms: {', '.join(all_terms)}. Example questions: {' | '.join(sample_questions)}"
            
            texts_to_embed.append(representative_text)
            metadata.append({
                "index": idx,
                "module": module,
                "theme": theme,
                "text": representative_text,
                "payload_count": len(payloads)
            })
            idx += 1
            
    print(f"Generated {len(texts_to_embed)} semantic representatives across {len(module_subgroups)} modules.")
    
    print("Loading model all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    
    print("Encoding embeddings...")
    embeddings = model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=True)
    
    out_dir = os.path.join(project_root, 'data', 'embeddings')
    os.makedirs(out_dir, exist_ok=True)
    
    npy_path = os.path.join(out_dir, 'module_embeddings.npy')
    json_path = os.path.join(out_dir, 'module_metadata.json')
    
    np.save(npy_path, embeddings)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved {embeddings.shape} embeddings to {npy_path}")
    print(f"Saved metadata to {json_path}")

if __name__ == "__main__":
    build_module_embeddings()
