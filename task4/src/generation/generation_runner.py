import os
import sys
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.generation.prompt_builder import PromptBuilder
from src.generation.qwen_generator import QwenGenerator
from src.validation.sql_validator import SQLValidator
from src.schema.schema_registry import SchemaRegistry

class GenerationRunner:
    """
    Runs a single benchmark sample through the generation pipeline:
    Prompt Builder -> Qwen Generator -> SQL Validator.
    """
    def __init__(self, data_dir: str):
        self.prompt_builder = PromptBuilder()
        self.qwen = QwenGenerator()
        
        # Initialize schema registry for the validator
        self.registry = SchemaRegistry(data_dir)
        self.validator = SQLValidator(self.registry)

    def run(self, 
            question: str, 
            ddl: List[str], 
            docs: List[str], 
            sql_examples: List[str], 
            expected_sql: str,
            business_semantics: List[str] = None) -> Dict[str, Any]:
        """
        Executes a single query generation and validation pipeline.
        """
        
        # 1. Format the context for PromptBuilder
        assembled_context = {
            "ddls": ddl,
            "docs": docs,
            "sql_examples": sql_examples,
            "business_semantics": business_semantics or []
        }
        
        # 2. Build the prompt
        prompt = self.prompt_builder.build(question, assembled_context)
        
        # 3. Generate the SQL using Qwen
        generated_sql = self.qwen.generate(prompt)
        
        # 4. Validate the generated SQL
        if generated_sql:
            validation_result = self.validator.validate(generated_sql)
        else:
            validation_result = {
                "valid": False, 
                "errors": ["Failed to generate SQL."],
                "hallucinations": {"tables": [], "columns": [], "joins": []}
            }
            
        # 5. Execution result (Mocked for benchmark since no live DB is attached)
        execution_result = {
            "executed": False,
            "error": "Execution not currently supported in isolated benchmark."
        }
        
        return {
            "generated_sql": generated_sql,
            "validation_result": validation_result,
            "execution_result": execution_result
        }
