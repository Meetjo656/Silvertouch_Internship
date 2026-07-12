import os
import sys

sys.path.append(os.path.dirname(__file__))
from experiment_runner import run_experiment

if __name__ == "__main__":
    run_experiment("strict_schema_prompt_test.json", max_tables=10, max_docs=3, max_sqls=3, strict_prompt=True)
