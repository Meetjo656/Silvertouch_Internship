import pandas as pd
import json
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]

# Load raw data
sql_path = base_dir / "data/raw/banks_sample_csv/pgviews_banks_sql_train.csv"
ddl_path = base_dir / "data/raw/banks_sample_csv/pgviews_banks_ddl_train.csv"
docs_path = base_dir / "data/raw/banks_sample_csv/banks_documentation_train.csv"

sql_df = pd.read_csv(sql_path)
ddl_df = pd.read_csv(ddl_path)
docs_df = pd.read_csv(docs_path)

# Join all DDL and documentation strings
full_ddl = "\n\n".join(ddl_df["ddl"].dropna().astype(str))
full_docs = "\n\n".join(docs_df["documentation"].dropna().astype(str))

# Create processed directory
processed_dir = base_dir / "data/processed"
processed_dir.mkdir(parents=True, exist_ok=True)
out_path = processed_dir / "train.jsonl"

with open(out_path, "w", encoding="utf-8") as f:
    for _, row in sql_df.iterrows():
        prompt = str(row["prompt"]).strip()
        response_sql = str(row["sql"]).strip()
        
        instruction = f"Schema:\n{full_ddl}\n\nDocumentation:\n{full_docs}\n\nQuestion:\n{prompt}"
        
        record = {
            "instruction": instruction,
            "response": response_sql
        }
        
        f.write(json.dumps(record) + "\n")

print(f"Generated train.jsonl at {out_path} with {len(sql_df)} samples.")
