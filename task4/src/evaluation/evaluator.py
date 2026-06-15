import sys
import os
import glob
import json

# Add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.retrieval.assembler import ContextAssembler
from src.utils.file_ops import load_json, save_json_artifact

def run_evaluator():
    benchmark_dir = os.path.join(project_root, 'data', 'benchmark')
    # Find the most recent benchmark file
    list_of_files = glob.glob(os.path.join(benchmark_dir, '*.json'))
    if not list_of_files:
        print("No benchmark file found.")
        return
        
    latest_benchmark_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading benchmark: {latest_benchmark_file}")
    
    benchmark_data = load_json(latest_benchmark_file)
    
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    catalog_path = os.path.join(project_root, 'data', 'semantic_catalog', 'semantic_catalog.json')
    
    assembler = ContextAssembler(raw_data_dir, catalog_path)
    
    results = []
    
    total_module_score = 0
    total_doc_score = 0
    total_ddl_score = 0
    total_sql_score = 0
    
    for i, item in enumerate(benchmark_data):
        q = item['question']
        expected_module = item['expected_module']
        expected_tables = set(item['expected_tables'])
        expected_sql = item['expected_sql']
        
        # Run retrieval
        context_pkg = assembler.assemble(q)
        
        predicted_modules = [m['module'] for m in context_pkg['predicted_modules']]
        retrieved_tables = set(context_pkg['relevant_tables'])
        retrieved_sqls = [s['sql'] for s in context_pkg['similar_sql_examples']]
        
        # 1. Module Score
        mod_score = 100 if expected_module in predicted_modules else 0
        total_module_score += mod_score
        
        # 2. Doc Score (Heuristic: did it retrieve docs if it got the module right?)
        doc_score = 100 if (mod_score == 100 and len(context_pkg['relevant_docs']) > 0) else 0
        total_doc_score += doc_score
        
        # 3. DDL Score (Recall of expected tables)
        if len(expected_tables) > 0:
            overlap = expected_tables.intersection(retrieved_tables)
            ddl_score = (len(overlap) / len(expected_tables)) * 100
        else:
            ddl_score = 100
        total_ddl_score += ddl_score
        
        # 4. SQL Score
        sql_score = 100 if expected_sql in retrieved_sqls else 0
        total_sql_score += sql_score
        
        results.append({
            "question": q,
            "category": item['category'],
            "scores": {
                "module_score": mod_score,
                "doc_score": doc_score,
                "ddl_score": ddl_score,
                "sql_example_score": sql_score
            },
            "context_package": context_pkg
        })
        
        print(f"Processed {i+1}/{len(benchmark_data)}")

    # Calculate final metrics
    num_q = len(benchmark_data)
    metrics = {
        "Module Accuracy": total_module_score / num_q,
        "Documentation Retrieval Accuracy": total_doc_score / num_q,
        "DDL Retrieval Accuracy": total_ddl_score / num_q,
        "SQL Example Retrieval Accuracy": total_sql_score / num_q,
        "Overall Retrieval Accuracy": (total_module_score + total_doc_score + total_ddl_score + total_sql_score) / (4 * num_q)
    }
    
    print("\n--- Final Metrics ---")
    print(json.dumps(metrics, indent=2))
    
    # Save Artifacts
    res_dir = os.path.join(project_root, 'outputs', 'retrieval_results')
    save_json_artifact(results, res_dir, 'retrieval_results')
    
    met_dir = os.path.join(project_root, 'outputs', 'benchmark_results')
    save_json_artifact(metrics, met_dir, 'retrieval_metrics')

if __name__ == "__main__":
    run_evaluator()
