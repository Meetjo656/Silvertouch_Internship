import os
import json
import random
import numpy as np
from collections import Counter

def get_base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATASET_PATH = os.path.join(get_base_dir(), "data", "training", "qwen_lora_train_v2.jsonl")
AUDIT_JSON = os.path.join(get_base_dir(), "outputs", "lora_dataset_audit.json")
AUDIT_MD = os.path.join(get_base_dir(), "outputs", "lora_dataset_audit.md")

REQUIRED_SECTIONS = [
    "QUESTION",
    "AVAILABLE TABLES",
    "Relevant Columns", # This is outputted by ContextOptimizer within AVAILABLE TABLES
    "KNOWN JOIN RELATIONSHIPS",
    "BUSINESS SEMANTICS",
    "DOCUMENTATION",
    "REFERENCE SQL PATTERN"
]

def load_dataset(path):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def audit_structure(samples):
    stats = {
        "total_samples": len(samples),
        "missing_messages": 0,
        "invalid_message_count": 0,
        "invalid_roles": 0
    }
    
    for sample in samples:
        if "messages" not in sample:
            stats["missing_messages"] += 1
            continue
            
        messages = sample["messages"]
        if len(messages) != 3:
            stats["invalid_message_count"] += 1
            continue
            
        if messages[0].get("role") != "system" or messages[1].get("role") != "user" or messages[2].get("role") != "assistant":
            stats["invalid_roles"] += 1
            
    return stats

def audit_prompt_sections(samples):
    section_counts = Counter()
    missing_section_counts = Counter()
    
    for sample in samples:
        try:
            user_prompt = sample["messages"][1]["content"]
            for section in REQUIRED_SECTIONS:
                if section in user_prompt:
                    section_counts[section] += 1
                else:
                    missing_section_counts[section] += 1
        except Exception:
            pass
            
    return {"present": dict(section_counts), "missing": dict(missing_section_counts)}

def audit_sql_targets(samples):
    stats = {
        "empty_sql": 0,
        "markdown_sql": 0,
        "non_sql": 0,
        "valid_sql": 0
    }
    
    for sample in samples:
        try:
            sql = sample["messages"][2]["content"].strip()
            if not sql:
                stats["empty_sql"] += 1
                continue
                
            # The Assistant response SHOULD be wrapped in markdown as per our prompt builder. 
            # Wait, the pseudo-code says: Must not contain markdown. 
            # Let's check how build_lora_dataset built it:
            # {"role": "assistant", "content": f"```sql\n{expected_sql}\n```"}
            # The Assistant DOES wrap it in Markdown. The user pseudo-code says `if ``` exists: flag markdown_sql`. 
            # Let's flag it, but consider it standard for our current dataset format unless instructed otherwise.
            # I will count how many are markdown.
            is_markdown = "```" in sql
            if is_markdown:
                stats["markdown_sql"] += 1
                
            cleaned_sql = sql.replace("```sql", "").replace("```", "").strip().upper()
            
            if cleaned_sql.startswith("HERE IS"):
                stats["non_sql"] += 1
            elif not (cleaned_sql.startswith("SELECT") or cleaned_sql.startswith("WITH") or cleaned_sql.startswith("INSERT") or cleaned_sql.startswith("UPDATE") or cleaned_sql.startswith("DELETE")):
                stats["non_sql"] += 1
            else:
                stats["valid_sql"] += 1
                
        except Exception:
            pass
            
    return stats

def audit_metadata(samples):
    complexity_counter = Counter()
    domain_counter = Counter()
    join_counter = Counter()
    
    for sample in samples:
        metadata = sample.get("metadata", {})
        complexity = metadata.get("complexity", "UNKNOWN")
        domain = metadata.get("module", "UNKNOWN")
        
        complexity_counter[complexity] += 1
        domain_counter[domain] += 1
        
        if metadata.get("contains_join"):
            join_counter["HAS_JOIN"] += 1
            
    return {
        "complexity": dict(complexity_counter),
        "domain": dict(domain_counter),
        "joins": dict(join_counter)
    }

def audit_token_lengths(samples):
    # Using a fast approximation since we may not have Qwen Tokenizer downloaded and it would take forever
    # Qwen tokenizer is very efficient. We will estimate with len / 3.5 if the tokenizer isn't available
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True)
        has_tokenizer = True
    except Exception:
        has_tokenizer = False
        
    lengths = []
    for sample in samples:
        try:
            text = "\n".join([m["content"] for m in sample["messages"]])
            if has_tokenizer:
                tokens = tokenizer.encode(text)
                lengths.append(len(tokens))
            else:
                # rough approximation for Qwen tokens
                lengths.append(int(len(text) / 3.5))
        except Exception:
            pass
            
    if not lengths:
        return {}
        
    return {
        "min": float(np.min(lengths)),
        "max": float(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99))
    }

def sample_examples(samples):
    examples = []
    random.seed(42)
    sample_10 = random.sample(samples, min(10, len(samples)))
    
    for s in sample_10:
        examples.append({
            "metadata": s.get("metadata", {}),
            "user_prompt": s["messages"][1]["content"][:500] + "...", # Truncated for report
            "target_sql": s["messages"][2]["content"]
        })
    return examples

def build_report(samples):
    return {
        "dataset_size": len(samples),
        "structure": audit_structure(samples),
        "prompt_sections": audit_prompt_sections(samples),
        "sql_targets": audit_sql_targets(samples),
        "metadata": audit_metadata(samples),
        "token_stats": audit_token_lengths(samples),
        "sample_examples": sample_examples(samples)
    }

def save_json(report):
    os.makedirs(os.path.dirname(AUDIT_JSON), exist_ok=True)
    with open(AUDIT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

def save_markdown(report):
    md = f"""# LoRA Dataset Audit

**Total Samples**: {report['dataset_size']}

## Structure
- Valid: {report['dataset_size'] - report['structure']['invalid_message_count'] - report['structure']['missing_messages']}
- Invalid Messages: {report['structure']['invalid_message_count']}
- Missing Messages: {report['structure']['missing_messages']}
- Invalid Roles: {report['structure']['invalid_roles']}

## Prompt Sections
"""
    for section, count in report['prompt_sections']['present'].items():
        md += f"- {section}: {count}\n"
    
    md += "\n**Missing Sections:**\n"
    for section, count in report['prompt_sections']['missing'].items():
        md += f"- {section}: {count}\n"
        
    md += f"""
## SQL Targets
- Valid SQL: {report['sql_targets']['valid_sql']}
- Markdown SQL: {report['sql_targets']['markdown_sql']}
- Empty SQL: {report['sql_targets']['empty_sql']}
- Non SQL: {report['sql_targets']['non_sql']}

## Token Statistics
- Min: {report['token_stats']['min']:.0f}
- Mean: {report['token_stats']['mean']:.0f}
- Median: {report['token_stats']['median']:.0f}
- P95: {report['token_stats']['p95']:.0f}
- P99: {report['token_stats']['p99']:.0f}
- Max: {report['token_stats']['max']:.0f}

## Complexity Distribution
"""
    for comp, count in report['metadata']['complexity'].items():
        md += f"- {comp}: {count}\n"
        
    md += "\n## Sample Metadata\n"
    for idx, ex in enumerate(report['sample_examples']):
        md += f"\n### Sample {idx+1}\n"
        md += f"**Complexity**: {ex['metadata'].get('complexity')}\n\n"
        md += f"**Target SQL**:\n{ex['target_sql']}\n\n"
        
    os.makedirs(os.path.dirname(AUDIT_MD), exist_ok=True)
    with open(AUDIT_MD, 'w', encoding='utf-8') as f:
        f.write(md)

def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    samples = load_dataset(DATASET_PATH)
    
    print("Auditing structure...")
    report = build_report(samples)
    
    print("Saving reports...")
    save_json(report)
    save_markdown(report)
    
    print("\n--- AUDIT SUMMARY ---")
    print(f"Total Samples: {report['dataset_size']}")
    print(f"Invalid Structure: {report['structure']['invalid_message_count']}")
    print(f"P95 Tokens: {report['token_stats']['p95']:.0f}")
    
    missing_joins = report['prompt_sections']['missing'].get('KNOWN JOIN RELATIONSHIPS', 0)
    print(f"Missing Joins: {missing_joins}")
    
    if report['structure']['invalid_message_count'] == 0 and missing_joins == 0 and report['token_stats']['p95'] < 2048:
        print("\n[SUCCESS] DATASET APPROVED. Ready for Phase 8.3 QLoRA Training.")
    else:
        print("\n[FAILURE] DATASET FAILED AUDIT.")
        
if __name__ == "__main__":
    main()
