"""
app/routers/analyze.py — NLU analysis endpoints.

  POST /analyze          → Full pipeline (intent + entities)
  POST /detect-intent    → Intent detection only
  POST /extract-entities → Entity extraction only
"""

import json
import time
import logging
from fastapi import APIRouter, HTTPException, Depends

import app.state as state
from app.config import TRAINING_DATA_DIR
from app.models import AnalyzeRequest, IntentRequest, EntityRequest
from app.inference.intent   import detect_intent
from app.inference.entities import extract_entities
from app.inference.validation import validate_entities
from app.inference.fallback import manage_fallback
from app.auth import get_api_key

logger = logging.getLogger("mybot-nlu")
router = APIRouter()


@router.post("/analyze")
async def analyze(request: AnalyzeRequest, api_key: str = Depends(get_api_key)):
    """Full NLU pipeline: entity extraction + validation + intent detection."""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Models are still loading.")

    request.text = request.text or ""

    start       = time.time()
    custom_pats = {k: v for k, v in (request.customPatterns or {}).items() if k and v}
    intent_required_entities_map = {}
    entity_configs = []

    if request.workflowId:
        wf_json = TRAINING_DATA_DIR / f"{request.workflowId}.json"
        if wf_json.exists():
            try:
                with open(wf_json, "r", encoding="utf-8") as f:
                    wf_data = json.load(f)
                entity_configs = wf_data.get("entities", [])
                for ent in entity_configs:
                    if ent.get("name") and ent.get("type", "regex") == "regex" and ent.get("regex"):
                        custom_pats[ent["name"]] = ent["regex"]
                for sub in wf_data.get("subIntents", []):
                    if sub.get("name") and sub.get("requiredEntities"):
                        intent_required_entities_map[sub["name"]] = sub["requiredEntities"]
            except Exception as e:
                logger.warning(f"Failed to load workflow data from {wf_json}: {e}")

    # ─── 1. Extract Entities ───────────────────────────────────────────────
    entity_result = extract_entities(request.text, custom_pats, entity_configs, request.workflowId)
    
    # ─── 2. Validate Entities ──────────────────────────────────────────────
    valid_ents, valid_ent_list, invalid_ents = validate_entities(
        entity_result["entities"], 
        entity_result["entityList"], 
        entity_configs
    )

    # ─── 3. Detect Intent ──────────────────────────────────────────────────
    intent_result = detect_intent(
        request.text,
        request.workflowId or "welcome-intent",
        request.subIntents or [],
        request.confidenceThreshold if request.confidenceThreshold is not None else 0.30,
    )
    
    # ─── 4. Fallback Manager ───────────────────────────────────────────────
    required_entities = intent_required_entities_map.get(intent_result["subintent"], [])
    resolution = manage_fallback(intent_result, valid_ents, required_entities, invalid_ents)
    
    elapsed = round(time.time() - start, 3)

    wf_metadata             = state.trained_metadata_store.get(request.workflowId, {}) if request.workflowId is not None else {}
    entity_training_metrics = wf_metadata.get("entityMetrics", {})

    return {
        "success":          True,
        "query":            request.text,
        "sessionId":        request.sessionId,
        "workflowId":       request.workflowId,
        "intent":           intent_result["intent"],
        "subintent":        intent_result["subintent"],
        "confidence":       intent_result["confidence"],
        "resolution":       resolution,
        "intentMode":       intent_result.get("mode", "unknown"),
        "requiredEntities": required_entities,
        "entities":         valid_ents,
        "invalidEntities":  invalid_ents,
        "entityList":       valid_ent_list,
        "allScores":        intent_result.get("allScores", []),
        "processingTimeMs": elapsed * 1000,
        "models": {
            "entityExtraction": "spacy en_core_web_sm + trained-classifiers"
                                if entity_training_metrics else "spacy en_core_web_sm",
            "intentDetection":  f"LogisticRegression (trained:{request.workflowId})"
                                if request.workflowId in state.trained_classifiers
                                else "cosine-similarity (zeroshot)",
        },
        "entityTrainingMetrics": entity_training_metrics,
    }


@router.post("/detect-intent")
async def detect_intent_endpoint(request: IntentRequest, api_key: str = Depends(get_api_key)):
    """Intent/subintent detection only."""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Models are still loading.")
    
    request.text = request.text or ""
    result = detect_intent(
        request.text,
        request.workflowId or "welcome-intent",
        request.subIntents or [],
        request.confidenceThreshold if request.confidenceThreshold is not None else 0.30,
    )
    
    # Run through fallback manager with empty entities (since this is intent-only)
    resolution = manage_fallback(result, {}, [], {})
    
    return {
        "success": True, 
        "query": request.text, 
        "workflowId": request.workflowId, 
        "resolution": resolution,
        **result
    }


@router.post("/extract-entities")
async def extract_entities_endpoint(request: EntityRequest, api_key: str = Depends(get_api_key)):
    """Entity extraction only."""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Models are still loading.")
    
    request.text = request.text or ""
    result = extract_entities(request.text, request.customPatterns, request.entityConfigs)
    return {"success": True, "query": request.text, **result}
