import sys
import os
import glob
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.routing.router import ModuleRouter
from src.routing.semantic_router import SemanticModuleRouter
from src.routing.retrieval_router import RetrievalModuleRouter
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.utils.file_ops import load_json, save_json_artifact, save_markdown_artifact

def evaluate_router(router, benchmark_data, name, extractor=None):
    print(f"\nEvaluating {name}...")
    top1_correct = 0
    top3_correct = 0
    num_q = len(benchmark_data)
    
    for item in benchmark_data:
        q = item['question']
        expected = item['expected_module']
        
        if name == "Regex Router" and extractor is not None:
            _, extracted_raw = extractor.extract(q)
            predictions = router.predict(q, extracted_raw)
        else:
            predictions = router.predict(q)
            
        predicted_modules = [m['module'] for m in predictions]
        
        if predicted_modules and predicted_modules[0] == expected:
            top1_correct += 1
            
        if expected in predicted_modules[:3]:
            top3_correct += 1
            
    metrics = {
        "Top-1 Accuracy": round((top1_correct / num_q) * 100, 2),
        "Top-3 Accuracy": round((top3_correct / num_q) * 100, 2)
    }
    print(f"Results for {name}: {metrics}")
    return metrics

def run_router_eval():
    benchmark_dir = os.path.join(project_root, 'data', 'benchmark')
    list_of_files = glob.glob(os.path.join(benchmark_dir, '*.json'))
    if not list_of_files:
        print("No benchmark file found.")
        return
        
    latest_benchmark_file = max(list_of_files, key=os.path.getctime)
    benchmark_data = load_json(latest_benchmark_file)
    
    # Initialize Regex Router
    catalog_path = os.path.join(project_root, 'data', 'semantic_catalog', 'semantic_catalog.json')
    from src.routing.extractor import EntityExtractor
    extractor = EntityExtractor(catalog_path)
    legacy_router = ModuleRouter()
    
    # Initialize Router A
    embeddings_dir = os.path.join(project_root, 'data', 'embeddings')
    semantic_router = SemanticModuleRouter(embeddings_dir)
    
    # Initialize Router B
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    legacy_dir = os.path.join(raw_data_dir, 'Private Data', 'qdrant_export_20260615_062842', 'LEGACY')
    sql_path = os.path.join(legacy_dir, 'sql.json')
    with open(sql_path, 'r', encoding='utf-8') as f:
        raw_sql = json.load(f)
    corpus = [item['payload'] for item in raw_sql]
    
    print("Loading EmbeddingRetriever for Router B...")
    base_retriever = EmbeddingRetriever(model_name="all-MiniLM-L6-v2").fit(corpus, text_key='question')
    retrieval_router = RetrievalModuleRouter(base_retriever)
    
    # Evaluate
    results = {
        "Experiment A (Regex Router)": evaluate_router(legacy_router, benchmark_data, "Regex Router", extractor=extractor),
        "Experiment B (Representative Embedding Router)": evaluate_router(semantic_router, benchmark_data, "Representative Embedding Router"),
        "Experiment C (Retrieval-Driven Router)": evaluate_router(retrieval_router, benchmark_data, "Retrieval-Driven Router")
    }
    
    # Save Metrics
    met_dir = os.path.join(project_root, 'outputs', 'benchmark_results')
    save_json_artifact(results, met_dir, 'router_metrics')
    
    # Generate MD Report
    md_content = "# Module Router Architecture Comparison\n\n"
    md_content += "This report benchmarks the legacy regex-based routing against two semantic embedding architectures.\n\n"
    
    for exp, mets in results.items():
        md_content += f"## {exp}\n"
        md_content += f"- **Top-1 Accuracy**: {mets['Top-1 Accuracy']}%\n"
        md_content += f"- **Top-3 Accuracy**: {mets['Top-3 Accuracy']}%\n\n"
        
    md_content += "## Conclusion\n"
    md_content += "The highest performing architecture will be integrated into the `ContextAssembler` to permanently unlock massive Text-to-SQL performance.\n"
    
    rep_dir = os.path.join(project_root, 'outputs', 'reports')
    save_markdown_artifact(md_content, rep_dir, 'router_comparison')

if __name__ == "__main__":
    run_router_eval()
