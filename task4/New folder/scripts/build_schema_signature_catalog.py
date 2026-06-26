import os
import json
import re
from collections import defaultdict

def tokenize(text):
    if not text:
        return set()
    # Split by non-alphanumeric, convert to lower
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    # Stop words
    stop_words = {"the", "a", "an", "is", "of", "and", "in", "to", "for", "with", "on", "at", "by", "from", "chatbot", "private", "public", "master", "table", "id", "name"}
    return {t for t in tokens if t not in stop_words and len(t) > 2}

def load_json(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    return None

def build_catalog(data_dir):
    tables_info = defaultdict(lambda: {
        "signals": set(),
        "columns": set(),
        "modules": set()
    })

    # 1. Parse enterprise_semantic_catalog.json
    esc_path = os.path.join(data_dir, "enterprise_semantic_catalog.json")
    esc_data = load_json(esc_path)
    if esc_data:
        for concept_entry in esc_data:
            concept = concept_entry.get("concept", "")
            aliases = concept_entry.get("aliases", [])
            modules = concept_entry.get("modules", [])
            related_tables = concept_entry.get("related_tables", [])
            related_columns = concept_entry.get("related_columns", [])
            
            concept_tokens = tokenize(concept)
            for alias in aliases:
                concept_tokens.update(tokenize(alias))
                
            for table in related_tables:
                tables_info[table]["signals"].update(concept_tokens)
                for col in related_columns:
                    tables_info[table]["columns"].add(col)
                    tables_info[table]["signals"].update(tokenize(col))
                for mod in modules:
                    tables_info[table]["modules"].add(mod)
                    tables_info[table]["signals"].update(tokenize(mod))

    # 2. Parse DDL and DOC JSONs in raw data
    raw_dirs = [os.path.join(data_dir, "raw", "Private Data"), os.path.join(data_dir, "raw", "Public Data")]
    for raw_dir in raw_dirs:
        if not os.path.exists(raw_dir): continue
        for root, dirs, files in os.walk(raw_dir):
            module_name = os.path.basename(root)
            
            # ddl.json or sql.json might contain table schema
            for file in files:
                if file.endswith("ddl.json") or file.endswith("sql.json") or file.endswith("doc.json") or "ncd_public_chatbot_collection_ddl" in file or "docs" in file:
                    file_path = os.path.join(root, file)
                    data = load_json(file_path)
                    if not data: continue
                    
                    if isinstance(data, list):
                        for item in data:
                            payload = item.get("payload", {})
                            if not payload:
                                payload = item
                            
                            schema_text = payload.get("schema", "") or payload.get("ddl", "")
                            doc_text = payload.get("documentation", "") or payload.get("doc", "")
                            
                            # Extremely simple regex to find CREATE TABLE table_name
                            tables_found = re.findall(r'CREATE\s+TABLE\s+([a-zA-Z0-9_]+)', schema_text, re.IGNORECASE)
                            
                            for table in tables_found:
                                # Add table name tokens
                                tables_info[table]["signals"].update(tokenize(table))
                                # Add module
                                if module_name not in ["Public Data", "Private Data", "qdrant_exports", "qdrant_export_20260615_062842"]:
                                    tables_info[table]["modules"].add(module_name.upper())
                                    tables_info[table]["signals"].update(tokenize(module_name))
                                
                                # Add schema tokens and doc tokens
                                tables_info[table]["signals"].update(tokenize(schema_text))
                                tables_info[table]["signals"].update(tokenize(doc_text))
                                
                                # Extract columns (naive approach)
                                lines = schema_text.split('\n')
                                for line in lines:
                                    # Very basic column extraction
                                    match = re.match(r'\s+([a-zA-Z0-9_]+)\s+[A-Z]+', line)
                                    if match:
                                        col = match.group(1)
                                        tables_info[table]["columns"].add(col)
                                        tables_info[table]["signals"].update(tokenize(col))

    # Convert to catalog format
    catalog = []
    for table, info in tables_info.items():
        if not table: continue
        catalog.append({
            "table_name": table,
            "signals": list(info["signals"]),
            "columns": list(info["columns"]),
            "modules": list(info["modules"])
        })
        
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "schema_signature_catalog.json")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2)
        
    print(f"Generated schema_signature_catalog.json with {len(catalog)} tables.")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "data")
    build_catalog(data_dir)
