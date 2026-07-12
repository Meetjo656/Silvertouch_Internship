import os
import json
import sys
from collections import defaultdict

class BenchmarkReportGenerator:
    """
    Generates the final markdown report for Phase 7.5.
    Aggregates exact match, structural match, validation, hallucination, failure breakdowns, and complexity metrics.
    """
    @staticmethod
    def generate(results_filepath: str, complexity_filepath: str, output_filepath: str):
        if not os.path.exists(results_filepath):
            print("No generation results found.")
            return
            
        with open(results_filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        total = len(results)
        if total == 0:
            print("No results to report.")
            return
            
        exact_match_count = 0
        struct_match_count = 0
        validation_pass_count = 0
        
        table_hallucinations = 0
        col_hallucinations = 0
        
        failures = defaultdict(int)
        
        for res in results:
            if res.get("exact_match"):
                exact_match_count += 1
                
            if res.get("structural_match", 0) >= 0.99:
                struct_match_count += 1
                
            if res.get("valid"):
                validation_pass_count += 1
                
            if res.get("hallucinated_tables"):
                table_hallucinations += 1
                
            if res.get("hallucinated_columns"):
                col_hallucinations += 1
                
            failure = res.get("failure_type", "UNKNOWN")
            failures[failure] += 1
            
        exact_match_rate = (exact_match_count / total) * 100
        struct_match_rate = (struct_match_count / total) * 100
        validation_pass_rate = (validation_pass_count / total) * 100
        table_hallucination_rate = (table_hallucinations / total) * 100
        col_hallucination_rate = (col_hallucinations / total) * 100
        
        complexity_str = ""
        if os.path.exists(complexity_filepath):
            with open(complexity_filepath, 'r', encoding='utf-8') as f:
                comp_data = json.load(f).get("complexity_accuracy", {})
                for comp, acc in comp_data.items():
                    complexity_str += f"* **{comp}**: {acc}\n"
                    
        failure_str = ""
        for failure_type, count in sorted(failures.items(), key=lambda x: x[1], reverse=True):
            if failure_type != "NONE":
                failure_str += f"* **{failure_type}**: {(count/total)*100:.2f}% ({count}/{total})\n"

        report = f"""# Phase 7.5 Report: Live Qwen Generation Benchmark

## Overall Metrics
* **Exact Match**: {exact_match_rate:.2f}%
* **Structural Match**: {struct_match_rate:.2f}%
* **Validation Pass Rate**: {validation_pass_rate:.2f}%

## Hallucination Metrics
* **Table Hallucination Rate**: {table_hallucination_rate:.2f}%
* **Column Hallucination Rate**: {col_hallucination_rate:.2f}%

## Failure Breakdown
{failure_str if failure_str else "* None!\n"}
## Complexity Breakdown
{complexity_str if complexity_str else "* Not available\n"}
"""
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(report)
        print(f"Report saved to {output_filepath}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_path = os.path.join(base_dir, "outputs", "generation_results.json")
    complexity_path = os.path.join(base_dir, "outputs", "complexity_accuracy_report.json")
    out_path = os.path.join(base_dir, "outputs", "qwen_benchmark_report.md")
    
    BenchmarkReportGenerator.generate(results_path, complexity_path, out_path)
