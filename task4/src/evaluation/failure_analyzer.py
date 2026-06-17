import sqlglot
import sqlglot.expressions as exp
from typing import Dict, Any, List

class FailureAnalyzer:
    """
    Classifies every failed query into a specific failure category to definitively
    isolate Retrieval, Prompting, or Generation as the bottleneck.
    """
    @classmethod
    def analyze(cls, 
                expected_sql: str, 
                generated_sql: str, 
                expected_tables: List[str], 
                retrieved_tables: List[str], 
                validation_res: Dict[str, Any]) -> str:
        
        # 1. Retrieval Failure (Required tables not in retrieved context)
        expected_tables_lower = {t.lower() for t in expected_tables}
        retrieved_tables_lower = {t.lower() for t in retrieved_tables}
        if not expected_tables_lower.issubset(retrieved_tables_lower):
            return "RETRIEVAL_FAILURE"

        # 2. Syntax/Validation Failures
        if not validation_res.get("valid", False):
            err_str = " ".join(validation_res.get("errors", [])).lower()
            if "syntax" in err_str or "parse" in err_str:
                return "SYNTAX_FAILURE"
            hallucinations = validation_res.get("hallucinations", {})
            if hallucinations.get("tables"):
                return "TABLE_HALLUCINATION_FAILURE"
            if hallucinations.get("columns"):
                return "COLUMN_HALLUCINATION_FAILURE"
            if hallucinations.get("joins"):
                return "JOIN_HALLUCINATION_FAILURE"
            return "VALIDATION_FAILURE"

        # 3. Structural Failures (Compare expected vs generated ASTs)
        try:
            exp_ast = sqlglot.parse_one(expected_sql, read="postgres")
            gen_ast = sqlglot.parse_one(generated_sql, read="postgres")
            
            # Table Selection Failure
            exp_tables = {t.name.lower() for t in exp_ast.find_all(exp.Table) if t.name}
            gen_tables = {t.name.lower() for t in gen_ast.find_all(exp.Table) if t.name}
            if exp_tables != gen_tables:
                return "TABLE_SELECTION_FAILURE"
                
            # Join Failure
            exp_joins = len(list(exp_ast.find_all(exp.Join)))
            gen_joins = len(list(gen_ast.find_all(exp.Join)))
            if exp_joins != gen_joins:
                return "JOIN_FAILURE"
                
            # Aggregation Failure
            exp_groups = len(list(exp_ast.find_all(exp.Group)))
            gen_groups = len(list(gen_ast.find_all(exp.Group)))
            exp_aggs = len(list(exp_ast.find_all(exp.AggFunc)))
            gen_aggs = len(list(gen_ast.find_all(exp.AggFunc)))
            if exp_groups != gen_groups or exp_aggs != gen_aggs:
                return "AGGREGATION_FAILURE"
                
            # Filter Failure
            exp_wheres = len(list(exp_ast.find_all(exp.Where)))
            gen_wheres = len(list(gen_ast.find_all(exp.Where)))
            if exp_wheres != gen_wheres:
                return "FILTER_FAILURE"
                
            # Window Function Failure
            exp_windows = len(list(exp_ast.find_all(exp.Window)))
            gen_windows = len(list(gen_ast.find_all(exp.Window)))
            if exp_windows != gen_windows:
                return "WINDOW_FUNCTION_FAILURE"

        except Exception:
            pass # Fallback to prompt failure if we can't parse structure successfully
            
        # 4. Prompt Failure (Everything retrieved, no validation error, structurally similar but exact match failed)
        return "PROMPT_FAILURE"
