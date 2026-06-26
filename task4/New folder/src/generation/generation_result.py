from dataclasses import dataclass, field
from typing import List

@dataclass
class GenerationResult:
    """
    Standardized dataclass to store the output metrics of a single generation benchmark run.
    """
    question: str
    expected_sql: str
    generated_sql: str
    valid: bool
    exact_match: bool
    structural_match: float
    hallucinated_tables: List[str] = field(default_factory=list)
    hallucinated_columns: List[str] = field(default_factory=list)
    failure_type: str = "NONE"
