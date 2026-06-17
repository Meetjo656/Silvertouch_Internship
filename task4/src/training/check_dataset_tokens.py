import json
import numpy as np
from transformers import AutoTokenizer

def calculate_token_stats(file_path, tokenizer):
    token_counts = []
    
    print(f"Auditing tokens for: {file_path}")
    
    # Open and read the JSONL file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            # CRITICAL: Apply chat template exactly as it will be used in inference
            # Ensure tokenize=False here so we can pass it clean or let the tokenizer handle it,
            # or tokenize=True to get raw token IDs immediately.
            tokenized_chat = tokenizer.apply_chat_template(
                data["messages"], 
                tokenize=True, 
                add_generation_prompt=False
            )
            
            token_counts.append(len(tokenized_chat))
            
    # Calculate statistics
    max_length = max(token_counts)
    mean_length = np.mean(token_counts)
    p95 = np.percentile(token_counts, 95)
    p99 = np.percentile(token_counts, 99)
    
    print(f"--- Results for {file_path} ---")
    print(f"Max Length:       {max_length}")
    print(f"Mean Length:      {mean_length:.2f}")
    print(f"95th Percentile:  {p95:.2f}")
    print(f"99th Percentile:  {p99:.2f}\n")
    
    return token_counts

def main():
    # Load tokenizer from the target model
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # We only need the tokenizer, avoiding Unsloth's GPU requirement
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Audit both training and validation sets
    train_lengths = calculate_token_stats("data/qwen_lora_train_v2.jsonl", tokenizer)
    val_lengths = calculate_token_stats("data/qwen_lora_validation.jsonl", tokenizer)

if __name__ == "__main__":
    main()
