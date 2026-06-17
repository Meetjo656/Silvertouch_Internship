import os
import json
from collections import Counter

def analyze_dataset(data_dir: str, output_path: str):
    train_path = os.path.join(data_dir, "training", "qwen_lora_train_v2.jsonl")
    val_path = os.path.join(data_dir, "training", "qwen_lora_validation_v2.jsonl")
    
    report = {
        "Total Training Samples": 0,
        "Total Validation Samples": 0,
        "Unique Tables Used": set(),
        "Unique Domains": set(),
        "Average Context Length (Characters)": 0,
        "SQL Complexity Distribution": Counter(),
        "Join Distribution": Counter(),
        "Filter Distribution": Counter()
    }
    
    total_prompt_length = 0
    total_samples = 0
    
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            total_samples += 1
            meta = item.get("metadata", {})
            report["SQL Complexity Distribution"][meta.get("complexity", "UNKNOWN")] += 1
            report["Unique Domains"].add(meta.get("module", "UNKNOWN"))
            for t in meta.get("expected_tables", []):
                report["Unique Tables Used"].add(t)
                
            report["Join Distribution"]["HAS_JOIN" if meta.get("contains_join") else "NO_JOIN"] += 1
            report["Filter Distribution"]["HAS_FILTER" if meta.get("contains_filter") else "NO_FILTER"] += 1
            
            prompt = item["messages"][1]["content"]
            total_prompt_length += len(prompt)
            
    if os.path.exists(val_path):
        with open(val_path, 'r', encoding='utf-8') as f:
            for _ in f:
                report["Total Validation Samples"] += 1
            
    report["Total Training Samples"] = total_samples
    if total_samples > 0:
        report["Average Context Length (Characters)"] = int(total_prompt_length / total_samples)
        
    report["Unique Tables Used"] = len(report["Unique Tables Used"])
    report["Unique Domains"] = len(report["Unique Domains"])
    
    report["SQL Complexity Distribution"] = dict(report["SQL Complexity Distribution"])
    report["Join Distribution"] = dict(report["Join Distribution"])
    report["Filter Distribution"] = dict(report["Filter Distribution"])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    print(f"Dataset Analysis complete. Report saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    analyze_dataset(os.path.join(base_dir, "data"), os.path.join(base_dir, "outputs", "training_dataset_v2_report.json"))
