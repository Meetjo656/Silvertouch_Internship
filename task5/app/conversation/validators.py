import re
from typing import Any, Tuple, Optional, Callable

class EntityValidator:
    """Validates extracted entity values against various constraints."""

    def __init__(self):
        pass

    def validate(self, entity_type: str, value: Any, constraints: Optional[dict] = None) -> Tuple[bool, Any]:
        """Main entry point to dispatch validation based on type."""
        constraints = constraints or {}
        
        if entity_type == "regex":
            return self.validate_regex(value, constraints.get("pattern", ""))
        elif entity_type == "length":
            return self.validate_length(value, constraints.get("min", 0), constraints.get("max", float("inf")))
        elif entity_type == "type":
            return self.validate_type(value, constraints.get("expected_type", str))
        elif entity_type == "custom":
            return self.validate_custom(value, constraints.get("validator_func"))
            
        return True, value

    def validate_regex(self, value: Any, pattern: str) -> Tuple[bool, Any]:
        """Validate if value matches a regex pattern."""
        if not pattern:
            return True, value
        match = re.match(pattern, str(value))
        return bool(match), value

    def validate_length(self, value: Any, min_len: int, max_len: float) -> Tuple[bool, Any]:
        """Validate string length."""
        val_str = str(value)
        is_valid = min_len <= len(val_str) <= max_len
        return is_valid, val_str

    def validate_custom(self, value: Any, validator_func: Optional[Callable]) -> Tuple[bool, Any]:
        """Run a custom validation function."""
        if not validator_func:
            return True, value
        try:
            result = validator_func(value)
            return bool(result), value
        except Exception:
            return False, value

    def validate_type(self, value: Any, expected_type: type) -> Tuple[bool, Any]:
        """Validate value matches expected Python type and cast it if possible."""
        if isinstance(value, expected_type):
            return True, value
        try:
            return True, expected_type(value)
        except (ValueError, TypeError):
            return False, value