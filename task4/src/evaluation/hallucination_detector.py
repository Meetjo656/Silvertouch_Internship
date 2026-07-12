import sqlglot
import sqlglot.expressions as exp
from typing import Dict, Any

class HallucinationDetector:
    """
    Detects hallucinated tables and columns in generated SQL.
    A hallucination occurs when the model invents schema elements not present in the SchemaRegistry.
    """
    
    @staticmethod
    def detect(generated_sql: str, registry, dialect: str = "postgres") -> Dict[str, Any]:
        result = {
            "hallucinated_tables": [],
            "hallucinated_columns": []
        }
        
        if not generated_sql:
            return result
            
        try:
            expressions = sqlglot.parse(generated_sql, read=dialect)
            if not expressions:
                return result
            ast = expressions[0]
            if not ast:
                return result
        except Exception:
            # If it doesn't parse, we can't reliably detect hallucinations
            return result
            
        aliases = {}
        tables_used = set()
        
        # 1. Detect Table Hallucinations
        for table in ast.find_all(exp.Table):
            table_name = table.name
            if table_name:
                table_name_lower = table_name.lower()
                tables_used.add(table_name_lower)
                alias = table.alias if table.alias else table_name
                aliases[alias.lower()] = table_name_lower
                
                if not registry.table_exists(table_name_lower):
                    result["hallucinated_tables"].append(table_name)
                    
        # 2. Detect Column Hallucinations
        for column in ast.find_all(exp.Column):
            col_name = column.name
            if not col_name or col_name == "*":
                continue
            table_ref = column.table
            target_tables = []
            
            if table_ref:
                table_ref_lower = table_ref.lower()
                if table_ref_lower in aliases:
                    target_tables.append(aliases[table_ref_lower])
                else:
                    # If a column references an unknown table alias, that alias is a hallucination
                    result["hallucinated_tables"].append(table_ref)
                    continue
            else:
                # If no table prefix, the column must exist in at least one of the queried tables
                target_tables = list(tables_used)
                
            found = False
            for t in target_tables:
                if registry.column_exists(t, col_name):
                    found = True
                    break
                    
            if not found:
                result["hallucinated_columns"].append(col_name)
                
        # Deduplicate
        result["hallucinated_tables"] = list(set(result["hallucinated_tables"]))
        result["hallucinated_columns"] = list(set(result["hallucinated_columns"]))
        
        return result
