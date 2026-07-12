import os
import json
import sys
import sqlglot
import sqlglot.expressions as exp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.schema.schema_registry import SchemaRegistry
from src.evaluation.hallucination_detector import HallucinationDetector

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
        
    registry = SchemaRegistry(data_dir)
    detector = HallucinationDetector()
    
    out_path = os.path.join(out_dir, "validator_debug_log.txt")
    with open(out_path, 'w', encoding='utf-8') as logf:
        def log(msg):
            print(msg)
            logf.write(str(msg) + "\n")
            
        for i, item in enumerate(results):
            log(f"=== Sample {i+1} ===")
            log("Expected SQL:")
            log(item["expected_sql"])
            log("\nGenerated SQL:")
            log(item["generated_sql"])
            
            gen_sql = item["generated_sql"]
            if not gen_sql:
                log("No SQL generated.")
                log("="*40 + "\n")
                continue
                
            try:
                expressions = sqlglot.parse(gen_sql, read="postgres")
                ast = expressions[0]
                
                tables_extracted = [t.name for t in ast.find_all(exp.Table) if t.name]
                cols_extracted = [c.name for c in ast.find_all(exp.Column) if c.name and c.name != "*"]
                
                log(f"\nExtracted Tables: {tables_extracted}")
                log(f"Extracted Columns: {cols_extracted}")
                
                halls = detector.detect(gen_sql, registry)
                log(f"Hallucinated Tables Detected: {halls['hallucinated_tables']}")
                log(f"Hallucinated Columns Detected: {halls['hallucinated_columns']}")
                
                log("\nAlias Map Check:")
                aliases = {}
                for table in ast.find_all(exp.Table):
                    table_name = table.name
                    if table_name:
                        table_name_lower = table_name.lower()
                        alias = table.alias if table.alias else table_name
                        aliases[alias.lower()] = table_name_lower
                        log(f"  {alias} -> {table_name_lower} (Exists: {registry.table_exists(table_name_lower)})")
                        
                log("\nColumn Map Check:")
                for column in ast.find_all(exp.Column):
                    col_name = column.name
                    if not col_name or col_name == "*": continue
                    
                    table_ref = column.table
                    log(f"  Column: {col_name}, Ref: {table_ref}")
                    if table_ref:
                        table_ref_lower = table_ref.lower()
                        if table_ref_lower in aliases:
                            tname = aliases[table_ref_lower]
                            exists = registry.column_exists(tname, col_name)
                            log(f"    -> Resolves to {tname}. Exists: {exists}")
                        else:
                            log(f"    -> UNKNOWN ALIAS '{table_ref}'")
                    else:
                        log(f"    -> Unqualified. Target tables: {list(aliases.values())}")
                        found = any(registry.column_exists(t, col_name) for t in aliases.values())
                        log(f"    -> Found in any? {found}")
                
            except Exception as e:
                log(f"\nParse Error: {e}")
                
            log("="*40 + "\n")
            
    print(f"Full log written to {out_path}")

if __name__ == "__main__":
    main()
