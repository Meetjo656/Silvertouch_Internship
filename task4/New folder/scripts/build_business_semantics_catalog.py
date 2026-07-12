import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.schema.schema_registry import SchemaRegistry
from src.semantic.enum_extractor import EnumExtractor

def build_catalog(data_dir: str, out_path: str):
    print("Building Business Semantics Catalog...")
    registry = SchemaRegistry(data_dir)
    extractor = EnumExtractor()
    
    catalog = {} 
    
    # 1. Extract exact mappings from DDLs and Docs
    for table_name, meta in registry.tables.items():
        if meta.get("source") == "AUTHORITATIVE":
            ddl = meta.get("ddl", "")
            table_semantics = extractor.extract_from_ddl(ddl)
            
            md = registry.get_table_metadata(table_name)
            doc = md.get("documentation", md.get("doc", ""))
            doc_semantics = extractor.extract_from_doc(doc)
            
            merged = {}
            for col, mapping in table_semantics.items():
                merged[col] = mapping
            for col, mapping in doc_semantics.items():
                if col not in merged:
                    merged[col] = mapping
                else:
                    merged[col].update(mapping)
                    
            if merged:
                catalog[table_name] = merged
                
    # 2. Add raw historical filter values for fallback (REMOVED due to hallucination risk)
    # filter_path = os.path.join(data_dir, "processed", "filter_pattern_library.json")
    # if os.path.exists(filter_path):
    #     with open(filter_path, 'r', encoding='utf-8') as f:
    #         mined = json.load(f)
    #     catalog["_GLOBAL_MINED_FILTERS"] = mined
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2)
        
    print(f"Built business semantics catalog covering {len(catalog) - 1} tables. Saved to {out_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    build_catalog(os.path.join(base_dir, "data"), os.path.join(base_dir, "data", "processed", "business_semantics_catalog.json"))
