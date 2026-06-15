import sys
import os
from collections import Counter
import json

# Add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.utils.file_ops import load_json, save_json_artifact

def run_profiler():
    legacy_dir = os.path.join(project_root, 'data', 'raw', 'Private Data', 'qdrant_export_20260615_062842', 'LEGACY')
    
    print(f"Loading data from {legacy_dir}...")
    try:
        sql_data = load_json(os.path.join(legacy_dir, 'sql.json'))
        ddl_data = load_json(os.path.join(legacy_dir, 'ddl.json'))
        doc_data = load_json(os.path.join(legacy_dir, 'doc.json'))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # 1. Dataset Profile
    profile = {
        "total_sql_queries": len(sql_data),
        "total_ddl_tables": len(ddl_data),
        "total_documents": len(doc_data)
    }
    
    # 2. Module Statistics
    module_sql = Counter([item['payload'].get('module', 'UNKNOWN') for item in sql_data])
    module_ddl = Counter([item['payload'].get('module', 'UNKNOWN') for item in ddl_data])
    module_doc = Counter([item['payload'].get('module', 'UNKNOWN') for item in doc_data])
    
    module_stats = {
        "sql_by_module": dict(module_sql),
        "ddl_by_module": dict(module_ddl),
        "doc_by_module": dict(module_doc)
    }

    # 3. Business Term Frequency
    business_terms = Counter()
    for item in sql_data + ddl_data + doc_data:
        terms = item['payload'].get('business_terms', [])
        if isinstance(terms, list):
            for t in terms:
                business_terms[t.title()] += 1
        elif isinstance(terms, str):
            business_terms[terms.title()] += 1
            
    # 4. Table Frequency
    table_usage = Counter()
    for item in sql_data:
        for t in item['payload'].get('tables', []):
            table_usage[t] += 1

    # Save all artifacts
    reports_dir = os.path.join(project_root, 'outputs', 'reports')
    
    save_json_artifact(profile, reports_dir, 'dataset_profile')
    save_json_artifact(module_stats, reports_dir, 'module_statistics')
    save_json_artifact(dict(business_terms.most_common(100)), reports_dir, 'business_term_frequency')
    save_json_artifact(dict(table_usage.most_common(50)), reports_dir, 'table_frequency')
    
    print("Profiling complete. Artifacts saved.")

if __name__ == "__main__":
    run_profiler()
