import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.retrieval.sql_guided_semantic_retriever import SQLGuidedSemanticRetriever
from src.schema.schema_registry import SchemaRegistry
from src.generation.context_assembler import ContextAssembler

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
    registry = SchemaRegistry(data_dir)
    assembler = ContextAssembler(data_dir)
    
    total_drops = 0
    critical_drops = 0
    non_critical_drops = 0
    total_critical_tables = 0
    retained_critical_tables = 0
    
    report_lines = ["# Context Loss Analysis Report", ""]
    
    for item in benchmark_data:
        question = item.get("question", "")
        expected_sql = item.get("expected_sql", "")
        expected_tables = {t.lower() for t in item.get("expected_tables", [])}
        
        # Retrieval
        retrieval_res = retriever.retrieve(question, [expected_sql], [0.9])
        candidate_tables = retrieval_res["candidate_tables"]
        
        ddls = []
        docs = []
        for t in candidate_tables:
            if registry.table_exists(t):
                md = registry.get_table_metadata(t)
                ddls.append(registry.tables[t.lower()]["ddl"])
                docs.append(md.get("documentation", md.get("doc", "")))
            else:
                ddls.append("")
                docs.append("")
                
        # Assemble
        assembled = assembler.assemble(candidate_tables, ddls, docs, [expected_sql], question, expected_tables=expected_tables)
        
        kept = assembled.get("kept_tables", [])
        dropped = assembled.get("dropped_tables", [])
        
        total_drops += len(dropped)
        
        retrieved_expected = expected_tables.intersection({t.lower() for t in candidate_tables})
        total_critical_tables += len(retrieved_expected)
        
        for d in dropped:
            t_name = d["table"].lower()
            if t_name in expected_tables:
                critical_drops += 1
            else:
                non_critical_drops += 1
                
        for k in kept:
            t_name = k["table"].lower()
            if t_name in expected_tables:
                retained_critical_tables += 1
                
    critical_drop_rate = (critical_drops / len(benchmark_data)) if len(benchmark_data) > 0 else 0 
    context_retention_efficiency = (retained_critical_tables / total_critical_tables * 100) if total_critical_tables > 0 else 0
    
    report_lines.append(f"## Metrics")
    report_lines.append(f"* **Total Drops**: {total_drops}")
    report_lines.append(f"* **Critical Drops**: {critical_drops}")
    report_lines.append(f"* **Non-Critical Drops**: {non_critical_drops}")
    report_lines.append(f"* **Context Retention Efficiency**: {context_retention_efficiency:.2f}%")
    report_lines.append(f"* **Critical Drop Rate (per query avg)**: {critical_drop_rate:.2f}")
    
    with open(os.path.join(out_dir, "context_loss_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print("Context Loss Analysis complete. Saved to outputs/context_loss_report.md")

if __name__ == "__main__":
    main()
