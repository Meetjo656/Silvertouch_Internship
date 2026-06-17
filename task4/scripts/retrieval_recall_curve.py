import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.retrieval.sql_guided_semantic_retriever import SQLGuidedSemanticRetriever

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    benchmark_path = os.path.join(data_dir, "benchmark", "retrieval_benchmark_enriched_20260615_181102.json")
    if not os.path.exists(benchmark_path):
        benchmark_path = os.path.join(data_dir, "benchmark", "retrieval_benchmark_20260615_145857.json")
        
    benchmark_data = load_json(benchmark_path)
    if not benchmark_data:
        print("No benchmark data found.")
        return

    retriever = SQLGuidedSemanticRetriever(data_dir)
    
    k_values = [1, 3, 5, 10, 20]
    recalls = {k: 0 for k in k_values}
    total_queries = len(benchmark_data)
    
    misses = []

    print(f"Running recall curve on {total_queries} queries...")
    for idx, item in enumerate(benchmark_data):
        question = item.get("question", "")
        expected_sql = item.get("expected_sql", "")
        expected_tables = set(t.lower() for t in item.get("expected_tables", []))
        
        # We retrieve up to 20 to check the max recall limit.
        retrieval_res = retriever.retrieve(question, [expected_sql], [0.9], top_k=20)
        candidate_tables = [t.lower() for t in retrieval_res["candidate_tables"]]
        
        # Check recall at different K
        for k in k_values:
            k_candidates = set(candidate_tables[:k])
            if expected_tables.issubset(k_candidates):
                recalls[k] += 1
                
        # If it fails at K=5, record the miss
        if not expected_tables.issubset(set(candidate_tables[:5])):
            for expected_t in expected_tables:
                if expected_t not in candidate_tables[:5]:
                    rank = -1
                    if expected_t in candidate_tables:
                        rank = candidate_tables.index(expected_t) + 1
                        
                    misses.append({
                        "query": question,
                        "expected_table": expected_t,
                        "rank": rank
                    })

    results = {}
    for k in k_values:
        val = (recalls[k] / total_queries) * 100
        results[f"Recall@{k}"] = f"{val:.2f}%"
        
    out_path = os.path.join(out_dir, "retrieval_recall_curve.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": results,
            "misses_at_5": misses
        }, f, indent=2)

    print("Recall Curve Results:")
    print(json.dumps(results, indent=2))
    print(f"\nSaved detailed analysis to {out_path}")

if __name__ == "__main__":
    main()
