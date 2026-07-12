import sqlglot
import sqlglot.expressions as exp
from typing import Dict, Any, List

class SQLValidator:
    """
    Validates a generated SQL string against the authoritative SchemaRegistry.
    Checks for:
    - Successful parsing
    - Table existence
    - Column existence
    - Invalid aliases
    - Basic Join validity
    - Hallucinated schema elements
    """
    def __init__(self, registry, dialect="postgres"):
        self.registry = registry
        self.dialect = dialect

    def validate(self, sql: str) -> Dict[str, Any]:
        result = {
            "valid": True,
            "errors": [],
            "hallucinations": {
                "tables": [],
                "columns": [],
                "joins": []
            }
        }
        
        if not sql:
            result["valid"] = False
            result["errors"].append("SQL is empty.")
            return result

        try:
            expressions = sqlglot.parse(sql, read=self.dialect)
            if not expressions:
                result["valid"] = False
                result["errors"].append("SQL parsed to empty expression.")
                return result
            ast = expressions[0]
            if not ast:
                result["valid"] = False
                result["errors"].append("SQL AST is empty.")
                return result
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Syntax error: {e}")
            return result

        aliases = {}
        tables_used = set()

        # Check Table Existence and gather aliases
        for table in ast.find_all(exp.Table):
            table_name = table.name
            if table_name:
                table_name_lower = table_name.lower()
                tables_used.add(table_name_lower)
                alias = table.alias if table.alias else table_name
                aliases[alias.lower()] = table_name_lower
                
                if not self.registry.table_exists(table_name_lower):
                    result["valid"] = False
                    result["errors"].append(f"Table '{table_name}' does not exist in schema.")
                    result["hallucinations"]["tables"].append(table_name)

        # Check Column Existence
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
                    # Could be a hallucinated table reference
                    result["valid"] = False
                    result["errors"].append(f"Invalid table alias or reference '{table_ref}' for column '{col_name}'.")
                    continue
            else:
                target_tables = list(tables_used)

            # Check if col_name exists in ANY of the target tables
            found = False
            for t in target_tables:
                if self.registry.column_exists(t, col_name):
                    found = True
                    break
            
            if not found:
                result["valid"] = False
                result["errors"].append(f"Column '{col_name}' does not exist in referenced tables.")
                result["hallucinations"]["columns"].append(col_name)

        # Basic Join Validity (verifying joined tables exist)
        for join in ast.find_all(exp.Join):
            right_table_node = join.this
            if isinstance(right_table_node, exp.Table):
                table_name = right_table_node.name
                if table_name and not self.registry.table_exists(table_name):
                    result["valid"] = False
                    result["errors"].append(f"Joined table '{table_name}' does not exist in schema.")
                    result["hallucinations"]["joins"].append(table_name)

        # Deduplicate hallucinations
        for k in result["hallucinations"]:
            result["hallucinations"][k] = list(set(result["hallucinations"][k]))

        return result
