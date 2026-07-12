import os
import json
import sys
from dataclasses import asdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.generation.prompt_builder_strict import PromptBuilderStrict
from src.evaluation.sql_comparator import SQLComparator
from src.evaluation.structural_match import StructuralMatch
from src.evaluation.failure_classifier import FailureClassifier
from src.generation.generation_result import GenerationResult

def run_experiment(out_filename: str, max_tables: int, max_docs: int, max_sqls: int, strict_prompt: bool):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "outputs")
    
    benchmark_path = os.path.join(data_dir, "benchmark", "retrieval_benchmark_enriched_20260615_181102.json")
    if not os.path.exists(benchmark_path):
        benchmark_path = os.path.join(data_dir, "benchmark", "retrieval_benchmark_20260615_145857.json")
        
    runner = BenchmarkRunner(data_dir)
    
    if strict_prompt:
        runner.gen_runner.prompt_builder = PromptBuilderStrict()
        
    data = runner.load_benchmark(benchmark_path)[:20]
    results = []
    
    for idx, item in enumerate(data):
        print(f"Running {out_filename} {idx+1}/20...")
        question = item.get("question", "")
        expected_sql = item.get("expected_sql", "")
        expected_tables = set(t.lower() for t in item.get("expected_tables", []))
        
        # Override top_k
        retrieval_res = runner.retriever.retrieve(question, [expected_sql], [0.9], top_k=max_tables)
        candidate_tables = retrieval_res["candidate_tables"]
        
        ddls = []
        docs = []
        for t in candidate_tables:
            if runner.registry.table_exists(t):
                md = runner.registry.get_table_metadata(t)
                ddls.append(runner.registry.tables[t.lower()]["ddl"])
                docs.append(md.get("documentation", md.get("doc", "")))
            else:
                ddls.append("")
                docs.append("")
                
        assembled_context = runner.assembler.assemble(
            candidate_tables, ddls, docs, [expected_sql], question, expected_tables=expected_tables
        )
        
        final_ddls = assembled_context.get("ddls", [])
        final_docs = assembled_context.get("docs", [])[:max_docs]
        final_sqls = assembled_context.get("sql_examples", [])[:max_sqls]

        gen_res = runner.gen_runner.run(
            question=question,
            ddl=final_ddls,
            docs=final_docs,
            sql_examples=final_sqls,
            expected_sql=expected_sql
        )
        
        generated_sql = gen_res["generated_sql"] or ""
        validation = gen_res["validation_result"]
        
        exact_match = SQLComparator.exact_match(expected_sql, generated_sql)
        struct_match_scores = StructuralMatch.match(expected_sql, generated_sql)
        hallucinations = runner.hallucination_detector.detect(generated_sql, runner.registry)
        
        candidate_recall = expected_tables.issubset(set(t.lower() for t in candidate_tables))
        failure_type = FailureClassifier.classify(
            valid=validation["valid"],
            exact_match=exact_match,
            hallucinated_tables=hallucinations["hallucinated_tables"],
            hallucinated_columns=hallucinations["hallucinated_columns"],
            struct_scores=struct_match_scores,
            candidate_recall=candidate_recall
        )

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
        
    output_path = os.path.join(out_dir, out_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    # Print metrics
    table_halls = sum(1 for r in results if r["hallucinated_tables"]) / 20 * 100
    col_halls = sum(1 for r in results if r["hallucinated_columns"]) / 20 * 100
    valid_pass = sum(1 for r in results if r["valid"]) / 20 * 100
    struct_match = sum(r["structural_match"] for r in results) / 20 * 100
    
    print(f"--- Results for {out_filename} ---")
    print(f"Table Hallucination Rate: {table_halls:.2f}%")
    print(f"Column Hallucination Rate: {col_halls:.2f}%")
    print(f"Validation Pass Rate: {valid_pass:.2f}%")
    print(f"Avg Structural Match: {struct_match:.2f}%")
        
if __name__ == "__main__":
    run_type = sys.argv[1] if len(sys.argv) > 1 else ""
    if run_type == "context":
        run_experiment("context_reduction_test.json", max_tables=3, max_docs=1, max_sqls=1, strict_prompt=False)
    elif run_type == "strict":
        run_experiment("strict_schema_prompt_test.json", max_tables=10, max_docs=3, max_sqls=3, strict_prompt=True)
    elif run_type == "combined":
        run_experiment("combined_prompt_test.json", max_tables=3, max_docs=1, max_sqls=1, strict_prompt=True)
