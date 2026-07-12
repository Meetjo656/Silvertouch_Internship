import os
import json
import random

try:
    import tiktoken
except ImportError:
    tiktoken = None

def audit(data_dir: str):
    train_path = os.path.join(data_dir, "training", "qwen_lora_train.jsonl")
    raw_path = os.path.join(data_dir, "training", "raw_qwen_lora_dataset.jsonl")
    
    # Audit 1
    raw_counts = {"MULTI_JOIN": 0, "RANKING": 0, "WINDOW_FUNCTION": 0, "SUBQUERY": 0}
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            c = item.get("metadata", {}).get("complexity")
            if c in raw_counts:
                raw_counts[c] += 1
                
    print("--- AUDIT 1: Raw Counts (Before Oversampling) ---")
    for k, v in raw_counts.items():
        print(f"{k}: {v}")
        
    # Audit 2
    samples = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
            
    random.seed(42)
    sample_50 = random.sample(samples, min(50, len(samples)))
    
    valid_prompts = 0
    has_business_semantics = 0
    has_joins = 0 
    
    print("\n--- AUDIT 2: Inspection of 50 Samples ---")
    for item in sample_50:
        prompt = item["messages"][1]["content"]
        
        # Check structure
        if "QUESTION" in prompt and "AVAILABLE TABLES" in prompt and "REFERENCE SQL PATTERN" in prompt:
            valid_prompts += 1
            
        if "BUSINESS SEMANTICS" in prompt and "No business semantics needed" not in prompt:
            has_business_semantics += 1
            
        if "KNOWN JOIN PATHS" in prompt: 
            has_joins += 1
            
    print(f"Valid Prompt Structures: {valid_prompts}/50")
    print(f"Contains explicit Business Semantics: {has_business_semantics}/50")
    print(f"Contains explicit Join Paths: {has_joins}/50")
    
    # Audit 3
    print("\n--- AUDIT 3: Token Lengths ---")
    total_chars = 0
    total_tokens = 0
    for item in samples:
        prompt = item["messages"][1]["content"]
        target = item["messages"][2]["content"]
        total_chars += len(prompt) + len(target)
        if tiktoken:
            enc = tiktoken.get_encoding("cl100k_base")
            total_tokens += len(enc.encode(prompt + target))
        else:
            total_tokens += int(len(prompt + target) / 4) 
            
    avg_chars = total_chars / len(samples)
    avg_tokens = total_tokens / len(samples)
    print(f"Average Character Length: {avg_chars:.2f}")
    print(f"Average Token Length: {avg_tokens:.2f} tokens")
    
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    audit(os.path.join(base_dir, "data"))
