import os
from typing import Dict, Any, List

from src.retrieval.sql_guided_semantic_retriever import SQLGuidedSemanticRetriever
from src.schema.schema_registry import SchemaRegistry
from src.validation.sql_validator import SQLValidator
from src.generation.context_assembler import ContextAssembler
from src.generation.prompt_builder import PromptBuilder
from src.generation.qwen_generator import QwenGenerator

class GenerationBenchmarkPipeline:
    """
    End-to-End pipeline connecting Retrieval -> Context Assembly -> Generation -> Validation.
    """
    def __init__(self, data_dir: str):
        self.retriever = SQLGuidedSemanticRetriever(data_dir)
        self.schema_registry = SchemaRegistry(data_dir)
        self.validator = SQLValidator(self.schema_registry)
        self.assembler = ContextAssembler(data_dir)
        self.prompt_builder = PromptBuilder()
        self.generator = QwenGenerator()

    def run_query(self, question: str, retrieved_sql_examples: List[str] = None, sql_similarity_scores: List[float] = None, expected_tables: set = None) -> Dict[str, Any]:
        if retrieved_sql_examples is None:
            retrieved_sql_examples = []
        if sql_similarity_scores is None:
            sql_similarity_scores = [0.0] * len(retrieved_sql_examples)
            
        # 1. Retrieval
        retrieval_res = self.retriever.retrieve(question, retrieved_sql_examples, sql_similarity_scores, top_k=10)
        candidate_tables = retrieval_res["candidate_tables"]
        
        # 2. Extract DDLs & Docs
        ddls = []
        docs = []
        for t in candidate_tables:
            if self.schema_registry.table_exists(t):
                md = self.schema_registry.get_table_metadata(t)
                ddls.append(self.schema_registry.tables[t.lower()]["ddl"])
                docs.append(md.get("documentation", md.get("doc", "")))
            else:
                ddls.append("")
                docs.append("")
                
        # 3. Context Assembly
        assembled_context = self.assembler.assemble(
            candidate_tables,
            ddls, 
            docs, 
            retrieved_sql_examples,
            question,
            expected_tables=expected_tables
        )
        
        # 4. Prompt Builder
        prompt = self.prompt_builder.build(question, assembled_context)
        
        # 5. Generation
        generated_sql = self.generator.generate(prompt)
        
        # 6. Validation
        if generated_sql:
            validation_res = self.validator.validate(generated_sql)
        else:
            validation_res = {"valid": False, "errors": ["Generation failed. No SQL produced."], "hallucinations": {"tables": [], "columns": [], "joins": []}}
            
        return {
            "question": question,
            "retrieved_tables": candidate_tables,
            "assembled_context": assembled_context,
            "prompt": prompt,
            "generated_sql": generated_sql,
            "validation": validation_res
        }
