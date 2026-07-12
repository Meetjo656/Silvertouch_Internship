import os
import json
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.evaluation.query_complexity_classifier import QueryComplexityClassifier

class ComplexityEvaluator:
    """
    Evaluates the generation results and breaks down accuracy by SQL query complexity.
    """
    @staticmethod
    def evaluate(results_filepath: str, output_filepath: str):
        if not os.path.exists(results_filepath):
            print(f"Results file {results_filepath} not found.")
            return
            
        with open(results_filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        complexity_totals = defaultdict(int)
        complexity_correct = defaultdict(int)
        
        for res in results:
            expected_sql = res.get("expected_sql", "")
            failure_type = res.get("failure_type", "UNKNOWN")
            
            complexity = QueryComplexityClassifier.classify(expected_sql)
            
            complexity_totals[complexity] += 1
            if failure_type == "NONE":
                complexity_correct[complexity] += 1
                
        metrics = {}
        for comp, total in sorted(complexity_totals.items(), key=lambda item: item[1], reverse=True):
            correct = complexity_correct[comp]
            accuracy = (correct / total) * 100
            metrics[comp] = f"{accuracy:.2f}% ({correct}/{total})"
            
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "complexity_accuracy": metrics
            }, f, indent=2)
            
        print("Complexity Accuracy Report:")
        print(json.dumps(metrics, indent=2))
        print(f"\nSaved to {output_filepath}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_path = os.path.join(base_dir, "outputs", "generation_results.json")
    out_path = os.path.join(base_dir, "outputs", "complexity_accuracy_report.json")
    ComplexityEvaluator.evaluate(results_path, out_path)
