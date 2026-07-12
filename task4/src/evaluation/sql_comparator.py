import sqlglot

class SQLComparator:
    """
    Utility class to compare SQL queries for exact match
    by normalizing them through sqlglot parsing to eliminate formatting differences.
    """
    @staticmethod
    def exact_match(expected_sql: str, generated_sql: str, dialect: str = "postgres") -> bool:
        """
        Determines if two SQL queries are an exact match after normalization.
        """
        if not expected_sql or not generated_sql:
            return False
            
        try:
            # Parse and generate back to normalize formatting (casing, whitespace)
            expected_ast = sqlglot.parse_one(expected_sql, read=dialect)
            expected_norm = expected_ast.sql(dialect=dialect)
            
            generated_ast = sqlglot.parse_one(generated_sql, read=dialect)
            generated_norm = generated_ast.sql(dialect=dialect)
            
            return expected_norm == generated_norm
        except Exception:
            # Fallback to simple string comparison if parsing fails (e.g. invalid SQL)
            # Remove basic formatting differences
            exp_clean = " ".join(expected_sql.strip().lower().split())
            gen_clean = " ".join(generated_sql.strip().lower().split())
            return exp_clean == gen_clean
