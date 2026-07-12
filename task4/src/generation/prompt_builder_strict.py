import os
import sys
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.generation.prompt_builder import PromptBuilder

class PromptBuilderStrict(PromptBuilder):
    """
    Extremely strict variant of the prompt builder explicitly forbidding schema inference.
    """
    def build(self, question: str, assembled_context: Dict[str, List[str]]) -> str:
        ddls = "\n\n".join(assembled_context.get("ddls", []))
        docs = "\n\n".join(assembled_context.get("docs", []))
        sqls = "\n\n".join(assembled_context.get("sql_examples", []))
        
        prompt = f"""---
SYSTEM

You may ONLY use tables and columns listed below.

If a required table or column does not exist:
Return:
NO_VALID_SQL

Do NOT invent tables.
Do NOT invent columns.
Do NOT infer schema.
Do NOT guess schema.
Use only the provided DDL.
Return SQL only.

---
QUESTION

{question}

---
DDL CONTEXT

{ddls if ddls else "No DDL provided."}

---
DOCUMENTATION CONTEXT

{docs if docs else "No documentation provided."}

---
SIMILAR SQL EXAMPLES

{sqls if sqls else "No similar examples provided."}

---
Generate SQL.
"""
        return prompt.strip()
