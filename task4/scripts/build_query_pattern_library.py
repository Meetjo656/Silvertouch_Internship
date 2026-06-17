import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation.query_complexity_classifier import QueryComplexityClassifier
from src.utils.sql_structure_extractor import SQLStructureExtractor
from scripts.process_sql_knowledge import load_sql_examples

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "data")
    
    print("Loading historical SQL data...")
    train_data = load_sql_examples(data_dir)
    if not train_data:
        print("No historical SQL data found to build patterns.")
        return

    extractor = SQLStructureExtractor()
    patterns = {
        "RANKING": [],
        "AGGREGATION": [],
        "TREND_ANALYSIS": [],
        "MULTI_JOIN": [],
        "WINDOW_FUNCTION": [],
        "FILTER_QUERY": []
    }

    print(f"Processing {len(train_data)} historical queries for pattern extraction...")
    for item in train_data:
        sql = item.get("sql", "")
        if not sql: continue
        
        complexity = QueryComplexityClassifier.classify(sql)
        
        if complexity in patterns and len(patterns[complexity]) < 50: # Cap at 50 to avoid massive files
            struct = extractor.extract(sql)
            patterns[complexity].append({
                "pattern_type": complexity,
                "sql": sql,
                "joins": struct.get("joins", []),
                "aggregations": struct.get("aggregations", []),
                "filters": struct.get("filters", [])
            })

    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "query_pattern_library.json")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(patterns, f, indent=2)
        
    print(f"Generated query_pattern_library.json with {sum(len(v) for v in patterns.values())} total patterns.")

if __name__ == "__main__":
    main()
