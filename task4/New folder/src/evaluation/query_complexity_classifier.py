import sqlglot
import sqlglot.expressions as exp

class QueryComplexityClassifier:
    """
    Classifies benchmark queries by structural complexity.
    Helps isolate which type of queries the LLM fails on.
    """
    @classmethod
    def classify(cls, sql: str, dialect: str = "postgres") -> str:
        if not sql:
            return "UNKNOWN"
            
        try:
            expressions = sqlglot.parse(sql, read=dialect)
            if not expressions:
                return "UNKNOWN"
            ast = expressions[0]
            if not ast:
                return "UNKNOWN"
        except Exception:
            return "UNKNOWN"
            
        # Check CTE
        if list(ast.find_all(exp.With)):
            return "CTE"
            
        # Check Window Functions
        if list(ast.find_all(exp.Window)):
            return "WINDOW_FUNCTION"
            
        # Check Subquery
        # Looking for Subquery instances or multiple Selects
        selects = list(ast.find_all(exp.Select))
        if len(selects) > 1:
            # Check if there is an ORDER BY + LIMIT combined in a way that suggests ranking
            has_order = list(ast.find_all(exp.Order))
            has_limit = list(ast.find_all(exp.Limit))
            if has_order and has_limit:
                 pass # Might still be ranking within subquery
            return "SUBQUERY"
            
        # Multi-Join
        joins = list(ast.find_all(exp.Join))
        if len(joins) >= 2:
            return "MULTI_JOIN"
            
        # Ranking
        has_order = list(ast.find_all(exp.Order))
        has_limit = list(ast.find_all(exp.Limit))
        if has_order and has_limit:
            return "RANKING"
            
        # Aggregation
        if list(ast.find_all(exp.Group)) or list(ast.find_all(exp.AggFunc)):
            return "AGGREGATION"
            
        # Trend Analysis (heuristic based on time functions)
        sql_lower = sql.lower()
        if "date(" in sql_lower or "extract(" in sql_lower or "date_trunc" in sql_lower or "year(" in sql_lower or "month(" in sql_lower:
            return "TREND_ANALYSIS"
            
        # Filter Query
        if list(ast.find_all(exp.Where)):
            return "FILTER_QUERY"
            
        return "SIMPLE_SELECT"
