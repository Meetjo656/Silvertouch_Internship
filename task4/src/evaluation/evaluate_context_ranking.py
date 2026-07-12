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
from src.retrieval.context_ranker import ContextRanker
from src.utils.file_ops import load_json, save_json_artifact, save_markdown_artifact

def compute_recall(expected_set, retrieved_list, k):
    if not expected_set: return 100.0
    top_k_retrieved = set(retrieved_list[:k])
    overlap = expected_set.intersection(top_k_retrieved)
    return (len(overlap) / len(expected_set)) * 100.0

def evaluate_context_ranking():
    benchmark_dir = os.path.join(project_root, 'data', 'benchmark')
    bench_files = glob.glob(os.path.join(benchmark_dir, 'retrieval_benchmark_enriched_*.json'))
    if not bench_files:
        print("Enriched benchmark not found.")
        return
        
    benchmark = load_json(max(bench_files, key=os.path.getctime))
    
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    catalog_path = os.path.join(project_root, 'data', 'semantic_catalog', 'semantic_catalog.json')
    
    extractor = EntityExtractor(catalog_path)
    doc_retriever = DocRetriever(raw_data_dir)
    ddl_retriever = DDLRetriever(raw_data_dir)
    
    legacy_dir = os.path.join(raw_data_dir, 'Private Data', 'qdrant_export_20260615_062842', 'LEGACY')
    with open(os.path.join(legacy_dir, 'sql.json'), 'r', encoding='utf-8') as f:
        sql_corpus = [item['payload'] for item in json.load(f)]
        
    print("Loading Embedding Models...")
    embed_retriever = EmbeddingRetriever(model_name="all-MiniLM-L6-v2").fit(sql_corpus, text_key='question')
    
    # Initialize Ranker
    ranker = ContextRanker(embedding_model=embed_retriever.model)
    
    metrics = {
        "before": {"ddl1": 0, "ddl3": 0, "ddl5": 0, "doc1": 0, "doc3": 0, "doc5": 0, "suff": 0, "size": 0},
        "after":  {"ddl1": 0, "ddl3": 0, "ddl5": 0, "doc1": 0, "doc3": 0, "doc5": 0, "suff": 0, "size": 0}
    }
    
    failures = []
    num_q = len(benchmark)
    
    for item in benchmark:
        q = item['question']
        expected_module = item['expected_module']
        expected_tables = set(item['expected_tables'])
        expected_docs = set(item['expected_docs'])
        
        predicted_modules = [{"module": expected_module, "confidence": 1.0}]
        entities_dict, raw_entities = extractor.extract(q)
        
        # 1. Retrieve Raw Candidates
        sql_hits = embed_retriever.search(q, top_k=5, filter_dict=None)
        retrieved_sqls = [h['doc'].get('sql', '') for h in sql_hits]
        
        raw_tables = ddl_retriever.retrieve(predicted_modules, raw_entities) # List of table names
        raw_docs = doc_retriever.retrieve(predicted_modules, raw_entities) # List of dicts
        
        raw_docs_texts = [d.get('content', d.get('documentation', '')) for d in raw_docs]
        
        # 2. Compute Before Metrics
        b_ddl1 = compute_recall(expected_tables, raw_tables, 1)
        b_ddl3 = compute_recall(expected_tables, raw_tables, 3)
        b_ddl5 = compute_recall(expected_tables, raw_tables, 5)
        
        b_doc1 = compute_recall(expected_docs, raw_docs_texts, 1)
        b_doc3 = compute_recall(expected_docs, raw_docs_texts, 3)
        b_doc5 = compute_recall(expected_docs, raw_docs_texts, 5)
        
        b_suff = (b_ddl5 == 100) and (b_doc5 == 100)
        
        metrics['before']['ddl1'] += b_ddl1; metrics['before']['ddl3'] += b_ddl3; metrics['before']['ddl5'] += b_ddl5
        metrics['before']['doc1'] += b_doc1; metrics['before']['doc3'] += b_doc3; metrics['before']['doc5'] += b_doc5
        metrics['before']['suff'] += 1 if b_suff else 0
        metrics['before']['size'] += len(raw_tables) + len(raw_docs)
        
        # 3. Apply Deterministic Ranking (No UserContext as instructed)
        ranked_tables = ranker.rank_ddls(q, raw_tables, predicted_modules, retrieved_sqls, entities_dict, user_context=None)
        ranked_docs = ranker.rank_docs(q, raw_docs, predicted_modules, entities_dict, user_context=None)
        
        # 4. Compute After Metrics
        a_ddl1 = compute_recall(expected_tables, ranked_tables, 1)
        a_ddl3 = compute_recall(expected_tables, ranked_tables, 3)
        a_ddl5 = compute_recall(expected_tables, ranked_tables, 5)
        
        a_doc1 = compute_recall(expected_docs, ranked_docs, 1)
        a_doc3 = compute_recall(expected_docs, ranked_docs, 3)
        a_doc5 = compute_recall(expected_docs, ranked_docs, 5)
        
        a_suff = (a_ddl5 == 100) and (a_doc5 == 100)
        
        metrics['after']['ddl1'] += a_ddl1; metrics['after']['ddl3'] += a_ddl3; metrics['after']['ddl5'] += a_ddl5
        metrics['after']['doc1'] += a_doc1; metrics['after']['doc3'] += a_doc3; metrics['after']['doc5'] += a_doc5
        metrics['after']['suff'] += 1 if a_suff else 0
        
        # 5. Track Failures based on 'After' performance
        if not a_suff:
            missing = []
            if a_ddl5 < 100: missing.append("DDL")
            if a_doc5 < 100: missing.append("Docs")
            
            failures.append({
                "question": q,
                "expected_tables": list(expected_tables),
                "retrieved_tables": ranked_tables[:5],
                "expected_docs": list(expected_docs),
                "retrieved_docs": ranked_docs[:5],
                "missing_context": missing,
                "failure_reason": f"Ranker failed to push expected items into Top-5. Missing: {missing}"
            })
            
    # Calculate Averages
    for k in metrics['before']:
        metrics['before'][k] /= num_q
        metrics['after'][k] /= num_q
        
    metrics['before']['suff'] *= 100.0
    metrics['after']['suff'] *= 100.0
    
    # Save Deliverables
    rep_dir = os.path.join(project_root, 'outputs', 'reports')
    res_dir = os.path.join(project_root, 'outputs', 'benchmark_results')
    
    save_json_artifact(metrics, res_dir, 'context_ranking_metrics')
    save_json_artifact(failures, rep_dir, 'context_ranking_failures')
    
    # Generate MD Report
    md = "# Deterministic Context Ranking Evaluation\n\n"
    md += "## DDL Retrieval Recall\n"
    md += f"- **Recall@1**: {metrics['before']['ddl1']:.1f}% -> **{metrics['after']['ddl1']:.1f}%**\n"
    md += f"- **Recall@3**: {metrics['before']['ddl3']:.1f}% -> **{metrics['after']['ddl3']:.1f}%**\n"
    md += f"- **Recall@5**: {metrics['before']['ddl5']:.1f}% -> **{metrics['after']['ddl5']:.1f}%**\n\n"
    
    md += "## Documentation Retrieval Recall\n"
    md += f"- **Recall@1**: {metrics['before']['doc1']:.1f}% -> **{metrics['after']['doc1']:.1f}%**\n"
    md += f"- **Recall@3**: {metrics['before']['doc3']:.1f}% -> **{metrics['after']['doc3']:.1f}%**\n"
    md += f"- **Recall@5**: {metrics['before']['doc5']:.1f}% -> **{metrics['after']['doc5']:.1f}%**\n\n"
    
    md += "## Overall Context Quality\n"
    md += f"- **Context Sufficiency**: {metrics['before']['suff']:.1f}% -> **{metrics['after']['suff']:.1f}%**\n"
    md += f"- **Avg Total Context Items Supplied**: {metrics['before']['size']:.1f} items\n"
    
    save_markdown_artifact(md, rep_dir, 'context_ranking_report')
    print("Ranking Evaluation Completed. All deliverables saved.")

if __name__ == "__main__":
    evaluate_context_ranking()
