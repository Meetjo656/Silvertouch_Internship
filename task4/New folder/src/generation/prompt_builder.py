from typing import Dict, List

class PromptBuilder:
    """
    Constructs a production-style Text-to-SQL prompt from the optimized generation context.
    """
    def __init__(self, dialect: str = "postgres"):
        self.dialect = dialect

    def build(self, question: str, assembled_context: Dict[str, List[str]]) -> str:
        ddls = "\n\n".join(assembled_context.get("ddls", []))
        docs = "\n\n".join(assembled_context.get("docs", []))
        sqls = "\n\n".join(assembled_context.get("sql_examples", []))
        semantics = "\n\n".join(assembled_context.get("business_semantics", []))
        joins = "\n\n".join(assembled_context.get("join_paths", []))
        
        prompt = f"""---
QUESTION

{question}

---
AVAILABLE TABLES

{ddls if ddls else "No tables provided.\\nRelevant Columns: None"}

---
KNOWN JOIN RELATIONSHIPS

{joins if joins else "No join relationships known."}

---
DOCUMENTATION

{docs if docs else "No documentation provided."}

---
BUSINESS SEMANTICS

{semantics if semantics else "No business semantics needed."}

---
REFERENCE SQL PATTERN

{sqls if sqls else "No similar examples provided."}

---
INSTRUCTIONS

Generate valid {self.dialect.upper()} SQL only.
Use only the provided tables and columns.
Do NOT invent tables.
Do NOT invent columns.
Do NOT infer schema.
Return ONLY the raw SQL query, without any markdown formatting or explanation.

---
Generate SQL.
"""
        return prompt.strip()
