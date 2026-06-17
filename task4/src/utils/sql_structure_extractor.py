import sqlglot
import sqlglot.expressions as exp
from typing import Dict, List, Any

class SQLStructureExtractor:
    """
    Extracts structural metadata from SQL queries using sqlglot.
    Extracts:
    - tables
    - joins
    - columns
    - filters (WHERE conditions)
    - aggregations (SUM, COUNT, etc.)
    - group by
    - order by
    """

    def __init__(self, dialect: str = "postgres"):
        self.dialect = dialect

    def extract(self, sql: str) -> Dict[str, Any]:
        result = {
            "tables": [],
            "joins": [],
            "columns": [],
            "filters": [],
            "aggregations": [],
            "group_by": [],
            "order_by": []
        }
        if not sql or not isinstance(sql, str):
            return result

        try:
            # Parse the SQL statement
            expressions = sqlglot.parse(sql, read=self.dialect)
            if not expressions:
                return result
            ast = expressions[0]
            if not ast:
                return result

            # Extract tables and their aliases
            aliases_to_tables = {}
            for table in ast.find_all(exp.Table):
                table_name = table.name
                alias = table.alias if table.alias else table_name
                if table_name:
                    result["tables"].append(table_name)
                    aliases_to_tables[alias] = table_name

            # Extract joins
            for join in ast.find_all(exp.Join):
                right_table_node = join.this
                if isinstance(right_table_node, exp.Table):
                    right_table = right_table_node.name
                    on_clause = join.args.get("on")
                    if on_clause:
                        for column in on_clause.find_all(exp.Column):
                            table_ref = column.table
                            if table_ref and table_ref in aliases_to_tables:
                                left_table = aliases_to_tables[table_ref]
                                if left_table != right_table:
                                    # Create a deterministic pair representation
                                    pair = sorted([left_table, right_table])
                                    pair_dict = {"table_a": pair[0], "table_b": pair[1]}
                                    if pair_dict not in result["joins"]:
                                        result["joins"].append(pair_dict)

            # Extract columns
            for column in ast.find_all(exp.Column):
                col_name = column.name
                if col_name and col_name != "*":
                    result["columns"].append(col_name)

            # Extract aggregations
            for agg in ast.find_all(exp.AggFunc):
                agg_name = agg.key.upper()
                result["aggregations"].append(agg_name)

            # Extract filters (WHERE clause)
            where = ast.args.get("where")
            if where:
                for condition in where.find_all(exp.Condition):
                    if not isinstance(condition, (exp.And, exp.Or)):
                        # Just grab basic condition expressions (e.g. col = val)
                        result["filters"].append(condition.sql(dialect=self.dialect))

            # Extract GROUP BY
            group = ast.args.get("group")
            if group:
                for expr in group.expressions:
                    if isinstance(expr, exp.Column):
                        result["group_by"].append(expr.name)
                    else:
                        result["group_by"].append(expr.sql(dialect=self.dialect))

            # Extract ORDER BY
            order = ast.args.get("order")
            if order:
                for expr in order.expressions:
                    if isinstance(expr.this, exp.Column):
                        result["order_by"].append(expr.this.name)
                    else:
                        result["order_by"].append(expr.this.sql(dialect=self.dialect))

            # Deduplicate primitives
            result["tables"] = list(set(result["tables"]))
            result["columns"] = list(set(result["columns"]))
            result["aggregations"] = list(set(result["aggregations"]))
            result["group_by"] = list(set(result["group_by"]))
            result["order_by"] = list(set(result["order_by"]))
            result["filters"] = list(set(result["filters"]))

        except Exception as e:
            # If parsing fails, just return whatever we have or empty
            pass

        return result
