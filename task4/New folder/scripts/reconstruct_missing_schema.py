import os
import json
import sqlglot
import sqlglot.expressions as exp
from collections import defaultdict
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.process_sql_knowledge import load_sql_examples

def infer_schemas(examples, targets):
    inferred = defaultdict(lambda: {"columns": defaultdict(int), "usage_count": 0})
    
    for ex in examples:
        sql = ex.get("sql", "")
        if not sql: continue
        
        try:
            ast = sqlglot.parse_one(sql, read="postgres")
            
            # Extract mapping of table names to their aliases
            tables_in_query = {t.name.lower(): t.alias_or_name.lower() for t in ast.find_all(exp.Table) if t.name}
            
            for t in targets:
                t_lower = t.lower()
                if t_lower in tables_in_query:
                    alias = tables_in_query[t_lower]
                    inferred[t_lower]["usage_count"] += 1
                    
                    # Extract columns belonging to this table
                    for col in ast.find_all(exp.Column):
                        col_table = col.table.lower() if col.table else ""
                        if col_table == alias or col_table == t_lower:
                            inferred[t_lower]["columns"][col.name.lower()] += 1
                        elif not col_table and len(tables_in_query) == 1:
                            # If no table qualifier and it's the only table in query
                            inferred[t_lower]["columns"][col.name.lower()] += 1
        except Exception:
            pass
            
    return inferred

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    targets = ["chatbot_dccb_master_private", "chatbot_cooperative_rankings", "chatbot_scb_members_private"]
    
    print("Loading historical SQL for schema inference...")
    examples = load_sql_examples(data_dir)
    print(f"Loaded {len(examples)} SQL examples.")
    
    inferred = infer_schemas(examples, targets)
    
    results = []
    for t in targets:
        if t in inferred and inferred[t]["usage_count"] > 0:
            cols = [c for c, count in sorted(inferred[t]["columns"].items(), key=lambda x: x[1], reverse=True)]
            # Simple heuristic: Cap confidence at 0.95
            conf = min(0.95, 0.5 + (inferred[t]["usage_count"] * 0.05))
            results.append({
                "table": t,
                "schema_status": "INFERRED",
                "confidence": round(conf, 2),
                "inferred_columns": cols
            })
            print(f"Successfully inferred {len(cols)} columns for {t}")
        else:
            results.append({
                "table": t,
                "schema_status": "NOT_FOUND",
                "confidence": 0.0,
                "inferred_columns": []
            })
            print(f"Could not infer schema for {t}")
            
    out_path = os.path.join(out_dir, "inferred_schema_catalog.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Inferred schema catalog written to {out_path}")

if __name__ == "__main__":
    main()
