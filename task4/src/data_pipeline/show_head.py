import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]

sql = pd.read_csv(
    base_dir / "data/raw/banks_sample_csv/pgviews_banks_sql_train.csv"
)

ddl = pd.read_csv(
    base_dir / "data/raw/banks_sample_csv/pgviews_banks_ddl_train.csv"
)

docs = pd.read_csv(
    base_dir / "data/raw/banks_sample_csv/banks_documentation_train.csv"
)

print(sql.columns)
print(sql.head())
print(ddl.columns)
print(ddl.head())
print(docs.columns)
print(docs.head())