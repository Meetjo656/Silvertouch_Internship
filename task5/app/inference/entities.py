"""
app/inference/entities.py — Entity extraction pipeline (3 stages).

Stage 1: spaCy NER for general entities (PER, ORG, LOC, DATE, …)
Stage 2: Regex patterns (built-in + custom)
Stage 3: Trained custom entities — exact substring match first,
         then embedding similarity optionally gated by a presence classifier.
"""

import re
import logging
import numpy as np
from typing import Optional, Any, Dict
from app.conversation.session import ConversationSession
from app.conversation.manager import ConversationManager

from sentence_transformers import util

import app.state as state

logger = logging.getLogger("mybot-nlu")


# ─── Constants ────────────────────────────────────────────────────────────────

BUILTIN_PATTERNS: dict[str, str] = {
    "AppID":        r"\b\d{4,12}\b",
    "NumericID":    r"\b\d{4,12}\b",
    "PhoneNumber":  r"\b[6-9]\d{9}\b",
    "EmailAddress": r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
    "Date":         r"\b(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{2}[\/\-]\d{2})\b",
}

SPACY_TAG_MAP: dict[str, str] = {
    "PERSON": "PER",   "ORG": "ORG",          "GPE": "LOC",
    "LOC":    "LOC",   "PRODUCT": "MISC",      "EVENT": "MISC",
    "WORK_OF_ART": "MISC", "LAW": "MISC",      "LANGUAGE": "MISC",
    "DATE":   "DATE",  "TIME": "TIME",          "PERCENT": "PERCENT",
    "MONEY":  "MONEY", "QUANTITY": "QUANTITY",  "CARDINAL": "CARDINAL",
    "ORDINAL": "ORDINAL",
}

ENTITY_CLASSIFIER_PRESENCE_THRESHOLD  = 0.5
ENTITY_SEMANTIC_MATCH_THRESHOLD        = 0.80
ENTITY_SEMANTIC_MATCH_THRESHOLD_GATED  = 0.70


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_ngrams(text: str, max_n: int = 5) -> list[str]:
    words   = text.split()
    n_grams = []
    for n in range(1, min(max_n, len(words) + 1)):
        for i in range(len(words) - n + 1):
            n_grams.append(" ".join(words[i: i + n]))
    return n_grams


def _best_semantic_match(text: str, examples: list[str]) -> tuple[Optional[str], float]:
    """Find best n-gram ↔ example match via cosine similarity."""
    n_grams = _build_ngrams(text)
    if not n_grams or not state.intent_model:
        return None, 0.0

    n_gram_embeddings = state.intent_model.encode(n_grams, convert_to_tensor=True)
    ex_embeddings     = state.intent_model.encode(examples, convert_to_tensor=True)

    cosine_scores = util.cos_sim(n_gram_embeddings, ex_embeddings)
    max_idx       = np.unravel_index(cosine_scores.cpu().numpy().argmax(), cosine_scores.shape)
    best_score    = float(cosine_scores[max_idx[0]][max_idx[1]].item())

    return examples[max_idx[1]], best_score


# ─── Main extraction function ─────────────────────────────────────────────────

def extract_entities(
    text: str,
    custom_patterns: Optional[dict[str, str]] = None,
    entity_configs:  Optional[list] = None,
    workflow_id:     Optional[str]  = None,
) -> dict:
    """
    Extract entities from `text` using three staged techniques.

    Returns: { "entities": {name: value}, "entityList": [...] }
    """
    custom_patterns = custom_patterns or {}
    entity_configs  = entity_configs  or []

    entities:        dict = {}
    entity_list:     list = []
    regex_patterns:  dict = {}
    trained_entities: list = []

    for ent in entity_configs:
        ent_name = ent.get("name")
        if not ent_name:
            continue
        ent_type = ent.get("type", "regex")
        if ent_type == "regex" and ent.get("regex"):
            regex_patterns[ent_name] = ent["regex"]
        elif ent_type == "train":
            trained_entities.append(ent)

    for name, pattern in custom_patterns.items():
        if name not in regex_patterns:
            regex_patterns[name] = pattern

    # ── Stage 1: spaCy NER ───────────────────────────────────────────────────
    if state.spacy_nlp:
        try:
            doc = state.spacy_nlp(text)
            for ent in doc.ents:
                entity_type  = SPACY_TAG_MAP.get(ent.label_, ent.label_)
                entity_value = ent.text.strip()
                if not entity_value:
                    continue
                if entity_type not in entities:
                    entities[entity_type] = entity_value
                entity_list.append({
                    "name": entity_type, "value": entity_value,
                    "confidence": 1.0, "source": "spacy-general",
                })
            logger.info(f"spaCy extracted {sum(1 for e in entity_list if e['source'] == 'spacy-general')} entities")
        except Exception as e:
            logger.warning(f"spaCy NER error: {e}")

    # ── Stage 2: Regex patterns (built-in + custom) ───────────────────────────
    all_regexes = {**BUILTIN_PATTERNS, **regex_patterns}
    for name, pattern in all_regexes.items():
        try:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if not matches:
                continue
            val = matches[0]
            if isinstance(val, tuple):
                val = next((v for v in val if v), "")
            val = val.strip()
            if not val:
                continue
            if name not in entities:
                entities[name] = val
            entity_list.append({
                "name": name, "value": val, "allMatches": matches,
                "confidence": 1.0, "source": "regex",
            })
        except Exception as e:
            logger.warning(f"Regex matching error for '{name}': {e}")

    # ── Stage 3: Trained custom entities ─────────────────────────────────────
    if trained_entities:
        entity_clf_map = state.trained_entity_classifiers.get(workflow_id, {}) if workflow_id else {}

        for ent in trained_entities:
            ent_name = ent.get("name")
            examples = ent.get("examples") or []
            if not ent_name or not examples:
                continue

            matched_value: Optional[str] = None
            confidence    = 0.0
            source        = ""

            # Step 1: Exact substring match (highest confidence)
            for ex in sorted(examples, key=len, reverse=True):
                ex_str = str(ex).strip()
                if re.search(r"\b" + re.escape(ex_str) + r"\b", text, re.IGNORECASE):
                    matched_value = ex_str
                    confidence    = 1.0
                    source        = "trained-entity:substring"
                    break

            # Step 2: Embedding similarity (optionally gated by presence classifier)
            if not matched_value and state.intent_model:
                try:
                    presence_score = 1.0
                    has_classifier = ent_name in entity_clf_map
                    threshold      = ENTITY_SEMANTIC_MATCH_THRESHOLD

                    if has_classifier:
                        text_embedding = state.intent_model.encode([text], convert_to_tensor=False)
                        entity_proba   = entity_clf_map[ent_name].predict_proba(text_embedding)[0]
                        presence_score = float(entity_proba[1])
                        if presence_score <= ENTITY_CLASSIFIER_PRESENCE_THRESHOLD:
                            continue
                        threshold = ENTITY_SEMANTIC_MATCH_THRESHOLD_GATED

                    best_example, best_score = _best_semantic_match(text, examples)
                    if best_example and best_score > threshold:
                        matched_value = best_example
                        confidence    = best_score * presence_score
                        source        = (
                            "trained-entity:classifier-gated-embedding"
                            if has_classifier else "trained-entity:embedding"
                        )
                except Exception as e:
                    logger.warning(f"Trained entity embedding match error for '{ent_name}': {e}")

            if matched_value:
                entities[ent_name] = matched_value
                entity_list.append({
                    "name": ent_name, "value": matched_value,
                    "confidence": round(confidence, 4), "source": source,
                })

    return {"entities": entities, "entityList": entity_list}

def process_session_entities(
    text: str,
    session: ConversationSession,
    conv_manager: ConversationManager,
    custom_patterns: Optional[dict[str, str]] = None,
    entity_configs: Optional[list] = None,
    workflow_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the full conversational entity lifecycle:
    Extract Entity -> Validate -> Store in Session -> Check Completeness.
    """
    from app.inference.validation import validate_entities
    
    # 1. Extract Entity
    entity_result = extract_entities(text, custom_patterns, entity_configs, workflow_id)
    
    # 2. Validate
    valid_ents, valid_ent_list, invalid_ents = validate_entities(
        entity_result["entities"], 
        entity_result["entityList"], 
        entity_configs or []
    )
    
    # 3. Store in Session
    target = session.waiting_for
    if target and target in valid_ents:
        # User provided the specific entity we were waiting for
        conv_manager.store_entity(session.conversation_id, target, valid_ents[target])
    elif not target:
        # Opportunistically store any valid entities we found
        for k, v in valid_ents.items():
            conv_manager.store_entity(session.conversation_id, k, v)
            
    # Check if the target was invalid
    if target and target in invalid_ents:
        return {
            "status": "invalid_entity",
            "action": "prompt_again",
            "prompt": f"The {target} provided is invalid. Please try again.",
            "missingEntity": target,
            "entities": valid_ents,
            "invalid": invalid_ents
        }

    # 4. Entity Complete?
    if session.is_complete():
        # YES -> Resume Workflow
        return {
            "status": "success",
            "action": "resume_workflow",
            "entities": valid_ents,
            "invalid": invalid_ents
        }
    else:
        # NO -> Prompt Again
        missing = conv_manager.get_missing_entity(session.conversation_id)
        if missing:
            session.waiting_for = missing
            conv_manager.update_session(session)
            return {
                "status": "missing_entity",
                "action": "prompt_again",
                "prompt": f"Please provide your {missing}.",
                "missingEntity": missing,
                "entities": valid_ents,
                "invalid": invalid_ents
            }
            
        return {"status": "unknown"}
