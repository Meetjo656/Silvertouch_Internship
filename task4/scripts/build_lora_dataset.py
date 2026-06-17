import os
import json
import sys
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.schema.schema_registry import SchemaRegistry
from src.retrieval.sql_guided_semantic_retriever import SQLGuidedSemanticRetriever
from src.generation.context_assembler import ContextAssembler
from src.generation.context_optimizer import ContextOptimizer
from src.generation.prompt_builder import PromptBuilder
from src.evaluation.complexity_evaluator import QueryComplexityClassifier

def load_all_historical_queries(data_dir: str) -> List[Dict]:
    queries = []
    sql_dir = os.path.join(data_dir, "raw", "Private Data", "qdrant_export_20260615_062842")
    if not os.path.exists(sql_dir):
        print(f"SQL Dir not found: {sql_dir}")
        return queries
        
    for root, dirs, files in os.walk(sql_dir):
        if "sql.json" in files:
            path = os.path.join(root, "sql.json")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        payload = item.get("payload", {})
                        question = payload.get("question", "")
                        sql = payload.get("sql", "")
                        module = payload.get("module", "UNKNOWN")
                        expected_tables = payload.get("tables", [])
                        if question and sql:
                            queries.append({
                                "question": question,
                                "sql": sql,
                                "expected_tables": expected_tables,
                                "module": module
                            })
            except Exception as e:
                pass
    return queries

def build_dataset(data_dir: str, out_path: str):
    registry = SchemaRegistry(data_dir)
    retriever = SQLGuidedSemanticRetriever(data_dir)
    assembler = ContextAssembler(data_dir)
    optimizer = ContextOptimizer(registry, data_dir)
    prompt_builder = PromptBuilder()
    classifier = QueryComplexityClassifier()
    
    queries = load_all_historical_queries(data_dir)
    print(f"Loaded {len(queries)} historical queries.")
    
    dataset = []
    
    for i, item in enumerate(queries):
        if i % 100 == 0:
            print(f"Processing {i}/{len(queries)}...")
            
        question = item["question"]
        expected_sql = item["sql"]
        expected_tables = set(t.lower() for t in item.get("expected_tables", []))
        module = item.get("module", "UNKNOWN")
        
        complexity_labels = classifier.classify(expected_sql)
        primary_complexity = complexity_labels[0] if complexity_labels else "SIMPLE_SELECT"
        
        # 1. Retrieve
        retrieval_res = retriever.retrieve(question, [expected_sql], [0.9], top_k=10)
        candidate_tables = retrieval_res["candidate_tables"]
        
        # Extract basic schema for assembler
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
                
        # 2. Assemble Context
        assembled_context = assembler.assemble(
            candidate_tables,
            ddls,
            docs,
            [expected_sql],
            question,
            expected_tables=expected_tables
        )
        
        # 3. Optimize Context for Generation
        optimized_context = optimizer.optimize(
            question,
            candidate_tables,
            assembled_context.get("docs", []),
            assembled_context.get("sql_examples", []),
            retrieval_res.get("scores", None)
        )
        
        # 4. Format the final Prompt
        final_context = {
            "ddls": optimized_context["generation_ddls"],
            "docs": optimized_context["generation_docs"],
            "sql_examples": optimized_context["generation_sql_examples"],
            "business_semantics": optimized_context.get("business_semantics", []),
            "join_paths": optimized_context.get("join_paths", [])
        }
        
        prompt = prompt_builder.build(question, final_context)
        
        # 5. Format ChatML
        dataset_item = {
            "messages": [
                {"role": "system", "content": "You are an enterprise Text-to-SQL assistant. Generate strictly valid postgres SQL. Respond with ONLY the raw SQL."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": expected_sql}
            ],
            "metadata": {
                "question": question,
                "complexity": primary_complexity,
                "module": module,
                "expected_tables": list(expected_tables),
                "contains_join": "JOIN" in expected_sql.upper(),
                "contains_filter": "WHERE" in expected_sql.upper() or "HAVING" in expected_sql.upper(),
                "contains_business_semantics": len(optimized_context.get("business_semantics", [])) > 0
            }
        }
        
        dataset.append(dataset_item)
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
            
    print(f"Saved {len(dataset)} examples to {out_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    build_dataset(os.path.join(base_dir, "data"), os.path.join(base_dir, "data", "training", "raw_qwen_lora_dataset.jsonl"))
