import sys
import os
import json
import random

# Add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.utils.file_ops import load_json, save_json_artifact

def generate_benchmark():
    legacy_dir = os.path.join(project_root, 'data', 'raw', 'Private Data', 'qdrant_export_20260615_062842', 'LEGACY')
    sql_data = load_json(os.path.join(legacy_dir, 'sql.json'))
    
    categories = {
        "Single-Module": [],
        "Location-Heavy": [],
        "Institution-Type": [],
        "Cross-Domain": [],
        "Multi-Table": []
    }
    
    for item in sql_data:
        payload = item['payload']
        sql = payload.get('sql', '').lower()
        q = payload.get('question', '').lower()
        mod = payload.get('module', '')
        tables = payload.get('tables', [])
        
        entry = {
            "question": payload.get('question'),
            "expected_module": mod,
            "expected_tables": tables,
            "expected_sql": payload.get('sql')
        }
        
        # 1. Multi-Table Join (>2 tables)
        if len(tables) >= 3:
            categories["Multi-Table"].append(entry)
            
        # 2. Cross-Domain
        elif mod == 'AT_A_GLANCE':
            categories["Cross-Domain"].append(entry)
            
        # 3. Location-Heavy (has district, state, block filters)
        elif 'district_name ilike' in sql or 'state_name ilike' in sql:
            categories["Location-Heavy"].append(entry)
            
        # 4. Institution-Type (has PACS, DCCB, etc)
        elif any(inst in sql for inst in ['pacs', 'dccb', 'lamps', 'scb']):
            categories["Institution-Type"].append(entry)
            
        # 5. Single-Module (Fallback: 1 table, not AT_A_GLANCE)
        elif len(tables) == 1 and mod != 'AT_A_GLANCE':
            categories["Single-Module"].append(entry)

    # Sample up to 10 from each category
    benchmark_dataset = []
    for cat, items in categories.items():
        sample_size = min(10, len(items))
        sampled = random.sample(items, sample_size)
        for s in sampled:
            s['category'] = cat
            benchmark_dataset.append(s)
            
    print(f"Generated benchmark with {len(benchmark_dataset)} questions across {len(categories)} categories.")
    
    benchmark_dir = os.path.join(project_root, 'data', 'benchmark')
    save_json_artifact(benchmark_dataset, benchmark_dir, 'retrieval_benchmark')

if __name__ == "__main__":
    generate_benchmark()
