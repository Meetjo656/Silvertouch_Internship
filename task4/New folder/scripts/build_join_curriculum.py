import os
import json
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.schema.schema_registry import SchemaRegistry
from src.semantic.join_path_resolver import JoinPathResolver
from src.generation.context_optimizer import ContextOptimizer
from src.generation.prompt_builder import PromptBuilder
from src.evaluation.query_complexity_classifier import QueryComplexityClassifier

TABLE_CONCEPTS = {
    "cooperative": "cooperatives",
    "district": "districts",
    "state": "states",
    "scb": "state cooperative banks",
    "dccb": "district central cooperative banks",
    "pac": "PACS",
    "federation": "federations",
    "dairy": "dairy societies",
    "fishery": "fishery societies",
    "education": "training institutes",
    "rankings": "rankings",
    "members": "members"
}

def get_business_name(t_name: str) -> str:
    for k, v in TABLE_CONCEPTS.items():
        if k in t_name.lower():
            return v
    return t_name.replace("chatbot_", "").replace("_master", "").replace("_private", "").replace("_", " ")

def build_synthetic_joins(data_dir: str):
    registry = SchemaRegistry(data_dir)
    join_resolver = JoinPathResolver(data_dir, registry)
    optimizer = ContextOptimizer(registry, data_dir)
    prompt_builder = PromptBuilder()
    
    tables = list(registry.tables.keys())
    
    # Pre-compute adjacency
    adj = {t: [] for t in tables}
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            path = join_resolver.resolve_join_path([tables[i], tables[j]])
            if path:
                adj[tables[i]].append((tables[j], path[0]))
                adj[tables[j]].append((tables[i], path[0]))
                
    synthetic_samples = []
    
    # 1. Generate 3-Table Joins (MULTI_JOIN)
    # 2. Generate Ranking Joins
    
    random.seed(42)
    
    for t1 in tables:
        neighbors = adj[t1]
        if len(neighbors) < 1:
            continue
            
        for t2, edge1 in neighbors:
            for t3, edge2 in adj[t2]:
                if t3 == t1:
                    continue
                    
                # We have t1 - t2 - t3
                concept1 = get_business_name(t1)
                concept2 = get_business_name(t2)
                concept3 = get_business_name(t3)
                
                # Make sure we don't just say "rankings rankings"
                if "ranking" in t1 or "ranking" in t2 or "ranking" in t3:
                    question = f"Show the top ranked {concept1} and their associated {concept3} information."
                else:
                    question = f"List all {concept1} including their {concept2} details and connected {concept3}."
                    
                sql = f"SELECT {t1}.*, {t2}.*, {t3}.* FROM {t1} JOIN {t2} ON {edge1} JOIN {t3} ON {edge2}"
                if "ranking" in sql:
                    sql += " ORDER BY rank_value LIMIT 10" # Mock ranking
                    
                candidate_tables = [t1, t2, t3]
                
                # Mock retriever docs/sqls
                retrieved_docs = [registry.metadata.get(t, {}).get("documentation", "") for t in candidate_tables]
                
                optimized_context = optimizer.optimize(
                    question,
                    candidate_tables,
                    retrieved_docs,
                    [sql], # Put SQL in history to ensure it's selected as reference
                    [0.99, 0.95, 0.90]
                )
                
                final_context = {
                    "ddls": optimized_context["generation_ddls"],
                    "docs": optimized_context["generation_docs"],
                    "sql_examples": optimized_context["generation_sql_examples"],
                    "business_semantics": optimized_context.get("business_semantics", []),
                    "join_paths": optimized_context.get("join_paths", [])
                }
                
                prompt = prompt_builder.build(question, final_context)
                
                complexity = QueryComplexityClassifier.classify(sql)
                
                item = {
                    "messages": [
                        {"role": "system", "content": "You are an enterprise Text-to-SQL assistant. Generate strictly valid postgres SQL. Respond with ONLY the raw SQL wrapped in ```sql\n...\n```."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": f"```sql\n{sql}\n```"}
                    ],
                    "metadata": {
                        "question": question,
                        "complexity": complexity,
                        "module": "SYNTHETIC",
                        "expected_tables": candidate_tables,
                        "contains_join": True,
                        "contains_filter": False,
                        "contains_business_semantics": len(final_context["business_semantics"]) > 0
                    }
                }
                
                synthetic_samples.append(item)
                
                if len(synthetic_samples) >= 800:
                    break
            if len(synthetic_samples) >= 800:
                break
        if len(synthetic_samples) >= 800:
            break
            
    out_path = os.path.join(data_dir, "training", "synthetic_join_dataset.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for s in synthetic_samples:
            f.write(json.dumps(s) + '\n')
            
    print(f"Generated {len(synthetic_samples)} synthetic join queries.")
    
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    build_synthetic_joins(os.path.join(base_dir, "data"))
