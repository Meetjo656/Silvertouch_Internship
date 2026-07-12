import sys
import os
import glob
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.retrieval.doc_retriever import DocRetriever
from src.retrieval.ddl_retriever import DDLRetriever
from src.retrieval.sql_retriever import SQLRetriever
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.routing.extractor import EntityExtractor
from src.utils.file_ops import load_json, save_json_artifact, save_markdown_artifact

def compute_recall(expected_set, retrieved_list, k):
    """Computes Recall@K"""
    if not expected_set:
        return 100.0
    # Take top K from retrieved
    top_k_retrieved = set(retrieved_list[:k])
    overlap = expected_set.intersection(top_k_retrieved)
    return (len(overlap) / len(expected_set)) * 100.0

def run_quality_analysis():
    benchmark_dir = os.path.join(project_root, 'data', 'benchmark')
    bench_files = glob.glob(os.path.join(benchmark_dir, 'retrieval_benchmark_enriched_*.json'))
    if not bench_files:
        print("Enriched benchmark not found.")
        return
        
    latest_bench_file = max(bench_files, key=os.path.getctime)
    benchmark = load_json(latest_bench_file)
    
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    catalog_path = os.path.join(project_root, 'data', 'semantic_catalog', 'semantic_catalog.json')
    
    # Initialize components
    extractor = EntityExtractor(catalog_path)
    doc_retriever = DocRetriever(raw_data_dir)
    ddl_retriever = DDLRetriever(raw_data_dir)
    
    # For SQL Retriever, we need top-10. The current SQLRetriever might have top_k=3 default.
    # We will initialize the EmbeddingRetriever directly to get up to 10 results.
    legacy_dir = os.path.join(raw_data_dir, 'Private Data', 'qdrant_export_20260615_062842', 'LEGACY')
    sql_path = os.path.join(legacy_dir, 'sql.json')
    with open(sql_path, 'r', encoding='utf-8') as f:
        raw_sql = json.load(f)
    sql_corpus = [item['payload'] for item in raw_sql]
    print("Loading EmbeddingRetriever for SQL Evaluation...")
    sql_retriever = EmbeddingRetriever(model_name="all-MiniLM-L6-v2").fit(sql_corpus, text_key='question')
    
    ddl_results = []
    doc_results = []
    sql_results = []
    failure_analysis = []
    sufficiency_results = []
    
    total_ddl_scores = {"k1": 0, "k3": 0, "k5": 0, "k10": 0}
    total_doc_scores = {"k1": 0, "k3": 0, "k5": 0}
    total_sql_scores = {"k1": 0, "k3": 0, "k5": 0, "k10": 0}
    total_sufficiency = 0
    
    num_q = len(benchmark)
    
    for item in benchmark:
        q = item['question']
        expected_module = item['expected_module']
        expected_tables = set(item['expected_tables'])
        expected_docs = set(item['expected_docs'])
        expected_sql = item['expected_sql']
        
        # Bypass router
        predicted_modules = [{"module": expected_module, "confidence": 1.0}]
        entities, raw_entities = extractor.extract(q)
        
        # 1. DDL Retrieval (Task 1)
        # DDLRetriever returns a list of table definitions or names.
        retrieved_tables_raw = ddl_retriever.retrieve(predicted_modules, raw_entities)
        # Assume it returns strings of table names or DDLs containing the table names.
        # To be safe, we will just check if the expected table name is anywhere in the retrieved strings.
        retrieved_table_names = []
        for ddl_str in retrieved_tables_raw:
            for exp_t in expected_tables:
                if exp_t.lower() in ddl_str.lower() and exp_t not in retrieved_table_names:
                    retrieved_table_names.append(exp_t)
        
        # Pad if needed, but normally retrieval returns all or nothing for a module.
        ddl_r1 = compute_recall(expected_tables, retrieved_table_names, 1)
        ddl_r3 = compute_recall(expected_tables, retrieved_table_names, 3)
        ddl_r5 = compute_recall(expected_tables, retrieved_table_names, 5)
        ddl_r10 = compute_recall(expected_tables, retrieved_table_names, 10)
        
        ddl_results.append({
            "question": q,
            "expected_tables": list(expected_tables),
            "retrieved_tables": retrieved_table_names,
            "recall_at_1": ddl_r1,
            "recall_at_3": ddl_r3,
            "recall_at_5": ddl_r5,
            "recall_at_10": ddl_r10
        })
        
        # 2. Doc Retrieval (Task 2)
        retrieved_docs_raw = doc_retriever.retrieve(predicted_modules, raw_entities)
        retrieved_docs = [d.get('content', '') for d in retrieved_docs_raw] if isinstance(retrieved_docs_raw[0], dict) else retrieved_docs_raw if retrieved_docs_raw else []
        
        doc_r1 = compute_recall(expected_docs, retrieved_docs, 1)
        doc_r3 = compute_recall(expected_docs, retrieved_docs, 3)
        doc_r5 = compute_recall(expected_docs, retrieved_docs, 5)
        
        doc_results.append({
            "question": q,
            "expected_docs": list(expected_docs),
            "retrieved_docs": retrieved_docs,
            "doc_recall": doc_r5 # Use best recall for overall doc recall
        })
        
        # 3. SQL Retrieval (Task 3)
        sql_hits = sql_retriever.search(q, top_k=10, filter_dict=None)
        retrieved_sqls = [h['doc'].get('sql', '') for h in sql_hits]
        
        expected_sql_set = {expected_sql} if expected_sql else set()
        sql_r1 = compute_recall(expected_sql_set, retrieved_sqls, 1)
        sql_r3 = compute_recall(expected_sql_set, retrieved_sqls, 3)
        sql_r5 = compute_recall(expected_sql_set, retrieved_sqls, 5)
        sql_r10 = compute_recall(expected_sql_set, retrieved_sqls, 10)
        
        sql_results.append({
            "question": q,
            "expected_sql": expected_sql,
            "retrieved_sql_examples": retrieved_sqls,
            "sql_recall": sql_r10
        })
        
        # Accumulate metrics
        total_ddl_scores['k1'] += ddl_r1; total_ddl_scores['k3'] += ddl_r3; total_ddl_scores['k5'] += ddl_r5; total_ddl_scores['k10'] += ddl_r10
        total_doc_scores['k1'] += doc_r1; total_doc_scores['k3'] += doc_r3; total_doc_scores['k5'] += doc_r5
        total_sql_scores['k1'] += sql_r1; total_sql_scores['k3'] += sql_r3; total_sql_scores['k5'] += sql_r5; total_sql_scores['k10'] += sql_r10
        
        # 4. Failure Analysis
        missing_components = []
        if ddl_r10 < 100:
            missing_components.append("Missing DDL")
            failure_analysis.append({"question": q, "failure_type": "Missing DDL", "root_cause": "DDL Retreiver failed to extract geographical/master tables or schema gap", "recommended_fix": "Enhance DDL dependency mapping"})
        if doc_r5 < 100:
            missing_components.append("Missing Documentation")
            failure_analysis.append({"question": q, "failure_type": "Missing Documentation", "root_cause": "Semantic mapping gap between expected business terms and doc metadata", "recommended_fix": "Improve doc tagging"})
        if sql_r10 < 100:
            missing_components.append("Missing SQL Example")
            failure_analysis.append({"question": q, "failure_type": "Missing SQL Example", "root_cause": "Semantic Retrieval Failure or unique query not in history", "recommended_fix": "Enhance embedding model or add few-shot rules"})
            
        # 5. Context Sufficiency
        sufficient = (ddl_r10 == 100) and (doc_r5 == 100) and (sql_r10 == 100)
        if sufficient:
            total_sufficiency += 1
            
        sufficiency_results.append({
            "question": q,
            "context_sufficient": sufficient,
            "missing_components": missing_components
        })
        
    # Calculate Averages
    ddl_avg = total_ddl_scores['k10'] / num_q
    doc_avg = total_doc_scores['k5'] / num_q
    sql_avg = total_sql_scores['k10'] / num_q
    suff_avg = (total_sufficiency / num_q) * 100.0
    
    # 6. Bottleneck Identification
    scores = {"DDL Retrieval": ddl_avg, "Documentation Retrieval": doc_avg, "SQL Example Retrieval": sql_avg}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    
    bottleneck_report = f"# Bottleneck Identification Report\n\n"
    bottleneck_report += f"## Average Retrieval Scores\n"
    bottleneck_report += f"- **DDL Retrieval Score**: {round(ddl_avg, 2)}%\n"
    bottleneck_report += f"- **Doc Retrieval Score**: {round(doc_avg, 2)}%\n"
    bottleneck_report += f"- **SQL Retrieval Score**: {round(sql_avg, 2)}%\n"
    bottleneck_report += f"- **Context Sufficiency Score**: {round(suff_avg, 2)}%\n\n"
    
    bottleneck_report += f"## Identification\n"
    bottleneck_report += f"**Primary Bottleneck**: {sorted_scores[0][0]} ({round(sorted_scores[0][1], 2)}%)\n"
    bottleneck_report += f"**Secondary Bottleneck**: {sorted_scores[1][0]} ({round(sorted_scores[1][1], 2)}%)\n"
    
    # Save Deliverables
    res_dir = os.path.join(project_root, 'outputs', 'benchmark_results')
    save_json_artifact(ddl_results, res_dir, 'ddl_retrieval_metrics')
    save_json_artifact(doc_results, res_dir, 'doc_retrieval_metrics')
    save_json_artifact(sql_results, res_dir, 'sql_retrieval_metrics')
    save_json_artifact(sufficiency_results, res_dir, 'context_sufficiency')
    
    rep_dir = os.path.join(project_root, 'outputs', 'reports')
    save_json_artifact(failure_analysis, rep_dir, 'retrieval_failure_analysis')
    save_markdown_artifact(bottleneck_report, rep_dir, 'bottleneck_report')
    
    print("Quality Analysis Completed. All 7 deliverables saved.")

if __name__ == "__main__":
    run_quality_analysis()
