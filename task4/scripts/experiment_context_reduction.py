import os
import sys

sys.path.append(os.path.dirname(__file__))
from experiment_runner import run_experiment

if __name__ == "__main__":
    run_experiment("context_reduction_test.json", max_tables=3, max_docs=1, max_sqls=1, strict_prompt=False)
