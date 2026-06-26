"""
app/core/fingerprint.py — MD5 fingerprinting for incremental training.

Two subintents with the same fingerprint have identical training data and
do NOT need to be re-encoded. This is the heart of incremental training.
"""

import hashlib


def intent_fingerprint(sub: dict) -> str:
    """Stable MD5 fingerprint of a subintent's training data."""
    examples = sorted(sub.get("examples", []))
    label    = sub.get("label", "")
    desc     = sub.get("description", "")
    raw      = f"{label}||{desc}||{'|'.join(examples)}"
    return hashlib.md5(raw.encode()).hexdigest()


def compute_intent_fingerprints(subIntents: list) -> dict[str, str]:
    """
    Build a name → fingerprint map from a list of subintents.
    Accepts Pydantic SubIntent objects or plain dicts.
    """
    result = {}
    for s in subIntents:
        name = getattr(s, "name", None) or (s.get("name") if isinstance(s, dict) else None)
        if not name:
            continue
        as_dict = {
            "examples":    list(getattr(s, "examples",    s.get("examples",    []) if isinstance(s, dict) else [])),
            "label":       getattr(s, "label",       s.get("label",       "") if isinstance(s, dict) else ""),
            "description": getattr(s, "description", s.get("description", "") if isinstance(s, dict) else ""),
        }
        result[name] = intent_fingerprint(as_dict)
    return result
