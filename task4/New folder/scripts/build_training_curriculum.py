import os
import json
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation.query_complexity_classifier import QueryComplexityClassifier
from src.schema.schema_registry import SchemaRegistry
from src.semantic.join_path_resolver import JoinPathResolver

def build_curriculum(data_dir: str):
    raw_path = os.path.join(data_dir, "training", "raw_qwen_lora_dataset.jsonl")
    synthetic_path = os.path.join(data_dir, "training", "synthetic_join_dataset.jsonl")
    
    if not os.path.exists(raw_path):
        print(f"Raw dataset not found at {raw_path}")
        return
        
    train_path = os.path.join(data_dir, "training", "qwen_lora_train_v2.jsonl")
    val_path = os.path.join(data_dir, "training", "qwen_lora_validation_v2.jsonl")
    join_path = os.path.join(data_dir, "training", "join_reasoning_subset.jsonl")
    filter_path = os.path.join(data_dir, "training", "filter_reasoning_subset.jsonl")
    
    weights = {
        "SIMPLE_SELECT": 1,
        "FILTER_QUERY": 3,
        "AGGREGATION": 2,
        "SUBQUERY": 3,
        "MULTI_JOIN": 5,
        "WINDOW_FUNCTION": 5,
        "RANKING": 5
    }
    
    val_modules = ["EDUCATION", "BANKS"]
    
    train_data = []
    val_data = []
    join_subset = []
    filter_subset = []
    
    registry = SchemaRegistry(data_dir)
    resolver = JoinPathResolver(data_dir, registry)
    
    print("Reading raw historical dataset and computing curriculum weights...")
    
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            meta = item.get("metadata", {})
            module = meta.get("module", "UNKNOWN")
            
            expected_sql = item["messages"][2]["content"].replace("```sql\n", "").replace("\n```", "")
            complexity = QueryComplexityClassifier.classify(expected_sql)
            item["metadata"]["complexity"] = complexity
            
            # Inject KNOWN JOIN RELATIONSHIPS to prompt string
            prompt = item["messages"][1]["content"]
            if "---\nDOCUMENTATION" in prompt and "KNOWN JOIN RELATIONSHIPS" not in prompt:
                tables = meta.get("expected_tables", [])
                join_paths = resolver.resolve_join_path(tables)
                joins_str = "\n\n".join(join_paths) if join_paths else "No join relationships known."
                new_block = f"---\nKNOWN JOIN RELATIONSHIPS\n\n{joins_str}\n\n---\nDOCUMENTATION"
                prompt = prompt.replace("---\nDOCUMENTATION", new_block)
                item["messages"][1]["content"] = prompt
            
            if meta.get("contains_join"):
                join_subset.append(item)
                
            if meta.get("contains_business_semantics"):
                filter_subset.append(item)
                
            if module in val_modules:
                val_data.append(item)
            else:
                weight = weights.get(complexity, 1)
                for _ in range(weight):
                    train_data.append(item)
                    
    print("Merging synthetic join dataset (1x weight)...")
    if os.path.exists(synthetic_path):
        with open(synthetic_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                train_data.append(item)
                join_subset.append(item)
                
    random.seed(42)
    random.shuffle(train_data)
    
    def write_jsonl(path, data):
        with open(path, 'w', encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')
                
    write_jsonl(train_path, train_data)
    write_jsonl(val_path, val_data)
    write_jsonl(join_path, join_subset)
    write_jsonl(filter_path, filter_subset)
    
    print(f"Curriculum Build Complete.")
    print(f"Train samples (Oversampled + Synthetic): {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Join subset samples: {len(join_subset)}")
    print(f"Filter subset samples: {len(filter_subset)}")
    
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    build_curriculum(os.path.join(base_dir, "data"))
