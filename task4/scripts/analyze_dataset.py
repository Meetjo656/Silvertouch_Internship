import json
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
data_path = base_dir / "data/processed/train.jsonl"

instruction_lengths = []
sql_lengths = []

with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        instruction_lengths.append(len(record["instruction"]))
        sql_lengths.append(len(record["response"]))

num_samples = len(instruction_lengths)

if num_samples > 0:
    avg_instruction_len = sum(instruction_lengths) / num_samples
    max_instruction_len = max(instruction_lengths)
    avg_sql_len = sum(sql_lengths) / num_samples
    max_sql_len = max(sql_lengths)
else:
    avg_instruction_len = 0
    max_instruction_len = 0
    avg_sql_len = 0
    max_sql_len = 0

print("Dataset Analysis:")
print("-" * 30)
print(f"Number of samples:          {num_samples}")
print(f"Average instruction length: {avg_instruction_len:.2f} characters")
print(f"Max instruction length:     {max_instruction_len} characters")
print(f"Average SQL length:         {avg_sql_len:.2f} characters")
print(f"Max SQL length:             {max_sql_len} characters")
