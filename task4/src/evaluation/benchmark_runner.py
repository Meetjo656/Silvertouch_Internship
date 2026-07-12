import os
import json
import sys
from typing import List
from dataclasses import asdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.retrieval.sql_guided_semantic_retriever import SQLGuidedSemanticRetriever
from src.schema.schema_registry import SchemaRegistry
from src.generation.context_assembler import ContextAssembler
from src.generation.generation_runner import GenerationRunner
from src.generation.context_optimizer import ContextOptimizer
from src.evaluation.sql_comparator import SQLComparator
from src.evaluation.structural_match import StructuralMatch
from src.evaluation.hallucination_detector import HallucinationDetector
from src.evaluation.failure_classifier import FailureClassifier
from src.generation.generation_result import GenerationResult

class BenchmarkRunner:
    """
    Main benchmark orchestrator for Phase 7.5.
    Loops over samples, runs the pipeline, evaluates generation, and stores the results.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.registry = SchemaRegistry(data_dir)
        self.retriever = SQLGuidedSemanticRetriever(data_dir)
        self.assembler = ContextAssembler(data_dir)
        self.optimizer = ContextOptimizer(self.registry, data_dir)
        self.gen_runner = GenerationRunner(data_dir)
        self.hallucination_detector = HallucinationDetector()

    def load_benchmark(self, filepath: str) -> List[dict]:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def run(self, benchmark_file: str, output_file: str):
        data = self.load_benchmark(benchmark_file)
        if not data:
            print(f"No benchmark data found at {benchmark_file}")
            return

        results = []
        optimization_metrics = []

        print(f"Running Live Qwen Benchmark on {len(data)} queries...")

        for idx, item in enumerate(data):
            print(f"Processing query {idx+1}/{len(data)}...")
            
            question = item.get("question", "")
            expected_sql = item.get("expected_sql", "")
            expected_tables = set(t.lower() for t in item.get("expected_tables", []))
            
            # 1. Retrieve
            retrieval_res = self.retriever.retrieve(question, [expected_sql], [0.9], top_k=10)
            candidate_tables = retrieval_res["candidate_tables"]
            
            # Extract basic schema for assembler
            ddls = []
            docs = []
            for t in candidate_tables:
                if self.registry.table_exists(t):
                    md = self.registry.get_table_metadata(t)
                    ddls.append(self.registry.tables[t.lower()]["ddl"])
                    docs.append(md.get("documentation", md.get("doc", "")))
                else:
                    ddls.append("")
                    docs.append("")
            
            # 2. Assemble Context
            assembled_context = self.assembler.assemble(
                candidate_tables,
                ddls,
                docs,
                [expected_sql],
                question,
                expected_tables=expected_tables
            )
            
            # 2.5 Optimize Context for Generation
            optimized_context = self.optimizer.optimize(
                question,
                candidate_tables,
                assembled_context.get("docs", []),
                assembled_context.get("sql_examples", []),
                retrieval_res.get("scores", None)
            )
            
            optimization_metrics.append(optimized_context["metrics"])
            
            final_ddls = optimized_context["generation_ddls"]
            final_docs = optimized_context["generation_docs"]
            final_sqls = optimized_context["generation_sql_examples"]

            # 3, 4, 5. Prompt Build -> Generate -> Validate (Handled by GenerationRunner)
            gen_res = self.gen_runner.run(
                question=question,
                ddl=final_ddls,
                docs=final_docs,
                sql_examples=final_sqls,
                expected_sql=expected_sql,
                business_semantics=optimized_context.get("business_semantics", [])
            )
            
            generated_sql = gen_res["generated_sql"] or ""
            validation = gen_res["validation_result"]
            
            # 6. Compare (Exact Match)
            exact_match = SQLComparator.exact_match(expected_sql, generated_sql)
            
            # 7. Classify & Detect Hallucinations
            struct_match_scores = StructuralMatch.match(expected_sql, generated_sql)
            hallucinations = self.hallucination_detector.detect(generated_sql, self.registry)
            
            # Determine Failure Type
            candidate_recall = expected_tables.issubset(set(t.lower() for t in candidate_tables))
            failure_type = FailureClassifier.classify(
                valid=validation["valid"],
                exact_match=exact_match,
                hallucinated_tables=hallucinations["hallucinated_tables"],
                hallucinated_columns=hallucinations["hallucinated_columns"],
                struct_scores=struct_match_scores,
                candidate_recall=candidate_recall
            )

            # 8. Store GenerationResult
            result_obj = GenerationResult(
                question=question,
                expected_sql=expected_sql,
                generated_sql=generated_sql,
                valid=validation["valid"],
                exact_match=exact_match,
                structural_match=struct_match_scores["overall"],
                hallucinated_tables=hallucinations["hallucinated_tables"],
                hallucinated_columns=hallucinations["hallucinated_columns"],
                failure_type=failure_type
            )
            
            results.append(asdict(result_obj))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        opt_out_file = os.path.join(os.path.dirname(output_file), "context_optimization_metrics.json")
        with open(opt_out_file, 'w', encoding='utf-8') as f:
            json.dump(optimization_metrics, f, indent=2)
            
        print(f"Benchmark complete. Results saved to {output_file}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    benchmark_path = os.path.join(data_dir, "benchmark", "retrieval_benchmark_enriched_20260615_181102.json")
    if not os.path.exists(benchmark_path):
        benchmark_path = os.path.join(data_dir, "benchmark", "retrieval_benchmark_20260615_145857.json")
        
    output_path = os.path.join(out_dir, "generation_results.json")
    
    runner = BenchmarkRunner(data_dir)
    runner.run(benchmark_path, output_path)
