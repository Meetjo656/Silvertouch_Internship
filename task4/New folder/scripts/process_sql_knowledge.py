import os
import json
import uuid
from collections import defaultdict
from tqdm import tqdm
import sys

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.sql_structure_extractor import SQLStructureExtractor

def load_sql_examples(data_dir):
    """
    Loads SQL examples from train.jsonl and qdrant exports.
    Returns a list of dicts: [{"id": str, "sql": str, ...}]
    """
    examples = []
    
    # Load from processed train.jsonl if it exists
    train_file = os.path.join(data_dir, "processed", "train.jsonl")
    if os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if "response" in data:
                    sql = data["response"]
                    examples.append({"id": str(uuid.uuid4()), "sql": sql})
    
    # Load from Public Data
    public_sql_file = os.path.join(data_dir, "raw", "Public Data", "qdrant_exports", "ncd_public_chatbot_collection_question_sql_v2.json")
    if os.path.exists(public_sql_file):
        with open(public_sql_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if "payload" in item and "sql" in item["payload"]:
                    examples.append({"id": item.get("id", str(uuid.uuid4())), "sql": item["payload"]["sql"]})

    # Load from Private Data
    private_data_dir = os.path.join(data_dir, "raw", "Private Data")
    if os.path.exists(private_data_dir):
        for root, dirs, files in os.walk(private_data_dir):
            for file in files:
                if file == "sql.json":
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            for item in data:
                                if "payload" in item and "sql" in item["payload"]:
                                    examples.append({"id": item.get("id", str(uuid.uuid4())), "sql": item["payload"]["sql"]})
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    return examples

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "data")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading SQL examples...")
    examples = load_sql_examples(data_dir)
    print(f"Loaded {len(examples)} SQL examples.")

    extractor = SQLStructureExtractor()

    sql_metadata = []
    table_usage_stats = defaultdict(lambda: {"usage_count": 0})
    join_graph = defaultdict(lambda: defaultdict(int))
    query_pattern_stats = {
        "aggregations": defaultdict(int),
        "filters": defaultdict(int),
        "patterns": []
    }

    print("Extracting SQL structures...")
    for ex in tqdm(examples):
        sql = ex["sql"]
        metadata = extractor.extract(sql)
        
        meta_entry = {
            "query_id": ex["id"],
            "tables": metadata["tables"],
            "joins": metadata["joins"],
            "aggregations": metadata["aggregations"],
            "filters": metadata["filters"],
            "group_by": metadata["group_by"],
            "order_by": metadata["order_by"]
        }
        sql_metadata.append(meta_entry)

        # Table stats
        for table in metadata["tables"]:
            table_usage_stats[table]["usage_count"] += 1
        
        # Join graph
        for join in metadata["joins"]:
            ta = join["table_a"]
            tb = join["table_b"]
            join_graph[ta][tb] += 1
            join_graph[tb][ta] += 1

        # Query patterns
        for agg in metadata["aggregations"]:
            query_pattern_stats["aggregations"][agg] += 1
            
        pattern = {
            "query_id": ex["id"],
            "aggregations": metadata["aggregations"],
            "filters": metadata["filters"]
        }
        query_pattern_stats["patterns"].append(pattern)

    # Save sql_metadata.json
    with open(os.path.join(processed_dir, "sql_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(sql_metadata, f, indent=2)

    # Save table_usage_stats.json
    with open(os.path.join(processed_dir, "table_usage_stats.json"), "w", encoding="utf-8") as f:
        json.dump(table_usage_stats, f, indent=2)

    # Save join_graph.json
    with open(os.path.join(processed_dir, "join_graph.json"), "w", encoding="utf-8") as f:
        json.dump(join_graph, f, indent=2)

    # Save query_pattern_stats.json
    with open(os.path.join(processed_dir, "query_pattern_stats.json"), "w", encoding="utf-8") as f:
        json.dump(query_pattern_stats, f, indent=2)

    print("Knowledge extraction complete. Files saved to data/processed/")

if __name__ == "__main__":
    main()
