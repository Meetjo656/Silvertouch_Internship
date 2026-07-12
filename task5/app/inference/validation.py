"""
app/inference/validation.py — Validates extracted entities against rules.
"""

import re
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger("mybot-nlu")

def validate_entities(
    extracted_entities: Dict[str, str],
    entity_list: List[Dict[str, Any]],
    entity_configs: List[Dict[str, Any]]
) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, str]]:
    """
    Validates extracted entities against constraints in entity_configs.
    
    Returns:
        (valid_entities, valid_entity_list, invalid_entities)
        where invalid_entities maps entity_name -> error_reason
    """
    valid_entities = {}
    invalid_entities = {}
    valid_entity_list = []
    
    # Create a lookup for configs
    config_map = {cfg.get("name"): cfg for cfg in entity_configs if cfg.get("name")}
    
    for name, value in extracted_entities.items():
        cfg = config_map.get(name, {})
        
        is_valid = True
        error_reason = ""
        
        # Check minLength
        min_len = cfg.get("minLength")
        if min_len is not None and len(value) < min_len:
            is_valid = False
            error_reason = f"must be at least {min_len} characters long"
            
        # Check maxLength
        max_len = cfg.get("maxLength")
        if is_valid and max_len is not None and len(value) > max_len:
            is_valid = False
            error_reason = f"must be no more than {max_len} characters long"
            
        # Check validationRegex
        val_regex = cfg.get("validationRegex")
        if is_valid and val_regex:
            try:
                if not re.fullmatch(val_regex, value, re.IGNORECASE):
                    is_valid = False
                    error_reason = "is in an invalid format"
            except Exception as e:
                logger.warning(f"Invalid validationRegex for entity '{name}': {e}")
                
        if is_valid:
            valid_entities[name] = value
        else:
            invalid_entities[name] = error_reason
            
    # Filter entityList
    for item in entity_list:
        if item.get("name") in valid_entities:
            valid_entity_list.append(item)
            
    if invalid_entities:
        logger.info(f"Validation failed for entities: {invalid_entities}")
            
    return valid_entities, valid_entity_list, invalid_entities
