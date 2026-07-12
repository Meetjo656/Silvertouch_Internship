import os
import json
import sys
from collections import defaultdict

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
    assembler = ContextAssembler()
    
    missing_tables = defaultdict(int)
    missing_docs = defaultdict(int)
    missing_signatures = defaultdict(int)
    
    gap_audit = []

    print(f"Analyzing retrieval gaps for {len(benchmark_data)} queries...")
    for idx, item in enumerate(benchmark_data):
        question = item.get("question", "")
        expected_sql = item.get("expected_sql", "")
        expected_tables = {t.lower() for t in item.get("expected_tables", [])}
        
        # 1. Simulate Retrieval Guidance
        retrieval_res = retriever.retrieve(question, [expected_sql], [0.9])
        retrieved_tables = {t.lower() for t in retrieval_res["candidate_tables"]}
        
        # 2. Check Missing Tables
        missed = expected_tables - retrieved_tables
        
        for mt in missed:
            missing_tables[mt] += 1
            
            # Check if it's in registry
            if not registry.table_exists(mt):
                missing_signatures[mt] += 1
            else:
                md = registry.get_table_metadata(mt)
                doc = md.get("documentation", md.get("doc", ""))
                if not doc or len(doc.strip()) < 10:
                    missing_docs[mt] += 1
                    
        # Check Context Assembly Losses
        # Could happen if assembler drops a retrieved table due to token budget.
        # But we pass all DDLs up to 60%. Let's see if we actually lost any here.
        ddls = []
        docs = []
        for t in retrieved_tables:
            if registry.table_exists(t):
                md = registry.get_table_metadata(t)
                ddls.append(registry.tables[t.lower()]["ddl"])
                docs.append(md.get("documentation", md.get("doc", "")))
        
        assembled = assembler.assemble(ddls, docs, [expected_sql])
        assembled_ddls_text = "\n".join(assembled["ddls"])
        
        # Check if retrieved expected tables made it through assembly
        assembly_losses = []
        for t in (expected_tables.intersection(retrieved_tables)):
            if t not in assembled_ddls_text.lower():
                assembly_losses.append(t)
                
        gap_audit.append({
            "question": question,
            "expected_tables": list(expected_tables),
            "retrieved_tables": list(retrieved_tables),
            "missed_tables": list(missed),
            "assembly_losses": assembly_losses
        })

    # Sort gaps
    sorted_missing_tables = dict(sorted(missing_tables.items(), key=lambda x: x[1], reverse=True))
    sorted_missing_docs = dict(sorted(missing_docs.items(), key=lambda x: x[1], reverse=True))
    sorted_missing_sigs = dict(sorted(missing_signatures.items(), key=lambda x: x[1], reverse=True))
    
    with open(os.path.join(out_dir, "top_missing_tables.json"), "w", encoding="utf-8") as f:
        json.dump(sorted_missing_tables, f, indent=2)
        
    with open(os.path.join(out_dir, "top_missing_docs.json"), "w", encoding="utf-8") as f:
        json.dump(sorted_missing_docs, f, indent=2)

    # Markdown Report
    report = f"""# Retrieval Gap Analysis Report

## Summary
* **Total Queries Analyzed**: {len(benchmark_data)}
* **Queries with Retrieval Misses**: {len([g for g in gap_audit if g['missed_tables']])}
* **Total Distinct Missing Tables**: {len(missing_tables)}

## Top Missing Tables
"""
    for t, c in list(sorted_missing_tables.items())[:10]:
        report += f"- **{t}**: {c} misses\n"
        
    report += "\n## Missing Schema Signatures (Not in DDL/Registry)\n"
    if sorted_missing_sigs:
        for t, c in list(sorted_missing_sigs.items())[:10]:
            report += f"- **{t}**: {c} missing signatures\n"
    else:
        report += "No schema signatures are missing for the expected tables.\n"

    report += "\n## Missing Documentation (In Registry, but poor/missing docs)\n"
    if sorted_missing_docs:
        for t, c in list(sorted_missing_docs.items())[:10]:
            report += f"- **{t}**: {c} times missed due to poor docs\n"
    else:
        report += "No major documentation gaps identified for the expected tables.\n"

    total_assembly_losses = sum(len(g["assembly_losses"]) for g in gap_audit)
    report += f"\n## Context Assembly Losses\n"
    report += f"* **Tables dropped due to token limits**: {total_assembly_losses}\n"
    
    with open(os.path.join(out_dir, "retrieval_gap_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
        
    print("Gap analysis complete.")
    print("Outputs generated:")
    print(" - outputs/top_missing_tables.json")
    print(" - outputs/top_missing_docs.json")
    print(" - outputs/retrieval_gap_report.md")

if __name__ == "__main__":
    main()
