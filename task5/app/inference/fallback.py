"""
app/inference/fallback.py — Fallback Manager and Smart Slot-Filling logic.

Intercepts intent and entity results to determine the next best action for the user
based on confidence tiers and missing required entities.
"""

from typing import Dict, Any, List

HIGH_THRESHOLD = 0.70
LOW_THRESHOLD = 0.30

def manage_fallback(
    intent_result: Dict[str, Any],
    extracted_entities: Dict[str, Any],
    required_entities: List[Any],
    invalid_entities: Dict[str, str]
) -> Dict[str, Any]:
    """
    Evaluates the results of intent detection and entity extraction.
    
    Returns a resolution dict with:
      - status: 'success', 'missing_entity', 'invalid_entity', 'clarification_needed', or 'fallback'
      - prompt: (Optional) A string prompting the user for missing info.
      - missingEntity: (Optional) The name of the missing entity.
      - clarificationOptions: (Optional) List of top intents if clarification is needed.
    """
    invalid_entities = invalid_entities or {}
    confidence = intent_result.get("confidence", 0.0)
    intent = intent_result.get("intent", "fallback")
    all_scores = intent_result.get("allScores", [])

    if confidence >= HIGH_THRESHOLD:
        # Check for missing or invalid entities
        missing_entity = None
        invalid_reason = None
        
        for req_ent in required_entities:
            ent_name = req_ent.get("name") if isinstance(req_ent, dict) else req_ent
            if ent_name in invalid_entities:
                missing_entity = ent_name
                invalid_reason = invalid_entities[ent_name]
                break
            elif ent_name not in extracted_entities:
                missing_entity = ent_name
                break
        
        if missing_entity and invalid_reason:
            return {
                "status": "invalid_entity",
                "prompt": f"The {missing_entity} you provided is invalid ({invalid_reason}). Please try again.",
                "missingEntity": missing_entity,
                "clarificationOptions": []
            }
        elif missing_entity:
            return {
                "status": "missing_entity",
                "prompt": f"Please provide your {missing_entity}.",
                "missingEntity": missing_entity,
                "clarificationOptions": []
            }
        
        return {
            "status": "success",
            "prompt": None,
            "missingEntity": None,
            "clarificationOptions": []
        }
        
    elif confidence >= LOW_THRESHOLD:
        # Clarification needed: Get top 4 intents (excluding the fallback/unknown ones)
        # all_scores is already sorted by score descending
        top_4 = []
        for score_info in all_scores:
            name = score_info.get("name")
            if name and name not in ("fallback", "unknown"):
                top_4.append(name)
            if len(top_4) >= 4:
                break
                
        return {
            "status": "clarification_needed",
            "prompt": "I'm not entirely sure what you mean. Did you mean one of these?",
            "missingEntity": None,
            "clarificationOptions": top_4
        }
        
    else:
        # Complete fallback
        return {
            "status": "fallback",
            "prompt": "I'm sorry, I didn't understand that. Could you please rephrase?",
            "missingEntity": None,
            "clarificationOptions": []
        }
