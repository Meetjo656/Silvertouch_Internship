import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.retrieval.sql_guided_semantic_retriever import SQLGuidedSemanticRetriever
from src.schema.schema_registry import SchemaRegistry
from src.generation.context_assembler import ContextAssembler
from src.generation.prompt_builder import PromptBuilder

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "data")
    
    # Try enriched benchmark first, fallback to original
    benchmark_path = os.path.join(data_dir, "benchmark", "retrieval_benchmark_enriched_20260615_181102.json")
    if not os.path.exists(benchmark_path):
        benchmark_path = os.path.join(data_dir, "benchmark", "retrieval_benchmark_20260615_145857.json")
        
    benchmark_data = load_json(benchmark_path)
    if not benchmark_data:
        print("No benchmark data found.")
        return

    retriever = SQLGuidedSemanticRetriever(data_dir)
    schema_registry = SchemaRegistry(data_dir)
    assembler = ContextAssembler()
    prompt_builder = PromptBuilder()
    
    ft_dataset = []
    
    print(f"Building FT dataset from {len(benchmark_data)} queries...")
    for item in benchmark_data:
        question = item.get("question", "")
        expected_sql = item.get("expected_sql", "")
        
        # 1. Retrieval
        retrieval_res = retriever.retrieve(question, [expected_sql], [0.9])
        candidate_tables = retrieval_res["candidate_tables"]
        
        # 2. Extract DDLs & Docs
        ddls = []
        docs = []
        for t in candidate_tables:
            if schema_registry.table_exists(t):
                md = schema_registry.get_table_metadata(t)
                ddls.append(schema_registry.tables[t.lower()]["ddl"])
                docs.append(md.get("documentation", md.get("doc", "")))
                
        # 3. Context Assembly
        assembled_context = assembler.assemble(ddls, docs, [expected_sql])
        
        # 4. Prompt Builder
        prompt = prompt_builder.build(question, assembled_context)
        
        ft_dataset.append({
            "question": question,
            "retrieved_sql_examples": assembled_context.get("sql_examples", []),
            "retrieved_ddl": assembled_context.get("ddls", []),
            "retrieved_docs": assembled_context.get("docs", []),
            "prompt": prompt,
            "expected_sql": expected_sql
        })

    out_dir = os.path.join(data_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "qwen_ft_dataset.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(ft_dataset, f, indent=2)
        
    print(f"Successfully built FT dataset with {len(ft_dataset)} examples at {out_path}")

if __name__ == "__main__":
    main()
