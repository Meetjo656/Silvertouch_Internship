import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.retrieval.sql_guided_semantic_retriever import SQLGuidedSemanticRetriever
from src.generation.context_assembler import ContextAssembler
from src.generation.prompt_builder import PromptBuilder
from src.schema.schema_registry import SchemaRegistry

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "outputs")
    
    results_path = os.path.join(out_dir, "generation_results.json")
    if not os.path.exists(results_path):
        print("Run benchmark first.")
        return
        
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)[:10]
        
    retriever = SQLGuidedSemanticRetriever(data_dir)
    assembler = ContextAssembler(data_dir)
    prompt_builder = PromptBuilder()
    registry = SchemaRegistry(data_dir)
    
    debug_samples = []
    
    for item in results:
        question = item["question"]
        expected_sql = item["expected_sql"]
        generated_sql = item["generated_sql"]
        validation_errors = item.get("hallucinated_tables", []) + item.get("hallucinated_columns", [])
        if not item.get("valid"):
            validation_errors.append("INVALID_SYNTAX")
            
        retrieval_res = retriever.retrieve(question, [expected_sql], [0.9], top_k=10)
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
                
        assembled_context = assembler.assemble(
            candidate_tables, ddls, docs, [expected_sql], question, set()
        )
        
        prompt = prompt_builder.build(question, assembled_context)
        
        debug_samples.append({
            "question": question,
            "prompt": prompt,
            "generated_sql": generated_sql,
            "expected_sql": expected_sql,
            "validation_errors": validation_errors,
            "failure_type": item.get("failure_type")
        })
        
    out_path = os.path.join(out_dir, "prompt_debug_samples.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(debug_samples, f, indent=2)
        
    print(f"Saved {len(debug_samples)} debug samples to {out_path}")

if __name__ == "__main__":
    main()
