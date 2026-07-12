import sys
import os
import glob
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.utils.file_ops import load_json, save_json_artifact

def enrich_benchmark():
    benchmark_dir = os.path.join(project_root, 'data', 'benchmark')
    # Load original benchmark
    bench_files = glob.glob(os.path.join(benchmark_dir, 'retrieval_benchmark_*.json'))
    latest_bench_file = max(bench_files, key=os.path.getctime)
    benchmark = load_json(latest_bench_file)
    
    # Load raw data to map relationships
    raw_dir = os.path.join(project_root, 'data', 'raw', 'Private Data', 'qdrant_export_20260615_062842', 'LEGACY')
    sql_data = [item['payload'] for item in load_json(os.path.join(raw_dir, 'sql.json'))]
    doc_data = [item['payload'] for item in load_json(os.path.join(raw_dir, 'doc.json'))]
    
    # Build lookup maps for faster processing
    sql_lookup = {item.get('sql'): item for item in sql_data if item.get('sql')}
    
    enriched_benchmark = []
    
    for idx, item in enumerate(benchmark):
        expected_sql = item['expected_sql']
        expected_module = item['expected_module']
        expected_tables = set(item['expected_tables'])
        
        # 1. Fetch business_terms from the raw SQL payload
        original_payload = sql_lookup.get(expected_sql, {})
        b_terms = original_payload.get('business_terms', [])
        if isinstance(b_terms, str):
            b_terms = [b_terms]
        expected_business_terms = set(b_terms)
        
        # 2. Derive expected_docs from relationships
        expected_docs = []
        for doc in doc_data:
            doc_module = doc.get('module', '')
            if doc_module != expected_module:
                continue
                
            doc_tables = doc.get('tables', [])
            if isinstance(doc_tables, str):
                doc_tables = [doc_tables]
            doc_tables_set = set(doc_tables)
            
            doc_terms = doc.get('business_terms', [])
            if isinstance(doc_terms, str):
                doc_terms = [doc_terms]
            doc_terms_set = set(doc_terms)
            
            # Intersection check
            if (expected_tables.intersection(doc_tables_set) or 
                expected_business_terms.intersection(doc_terms_set)):
                expected_docs.append(doc.get('content', ''))
                
        # Append enriched fields
        item['expected_business_terms'] = list(expected_business_terms)
        item['expected_docs'] = expected_docs
        enriched_benchmark.append(item)
        
    # Save Enriched Benchmark
    save_json_artifact(enriched_benchmark, benchmark_dir, 'retrieval_benchmark_enriched')
    print(f"Successfully enriched {len(enriched_benchmark)} benchmark queries.")

if __name__ == "__main__":
    enrich_benchmark()
