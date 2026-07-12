"""
app/routers/train.py — Training endpoints.

  POST /train             → Full retrain
  POST /train/incremental → Incremental retrain (skips unchanged intents)
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
import app.state as state
from app.models import TrainRequest, IncrementalTrainRequest
from app.core.cache import clear_embedding_cache
from app.training.intent import train_intent_classifier, train_intent_classifier_incremental, train_global_classifiers
from app.auth import get_api_key
logger = logging.getLogger("mybot-nlu")
router = APIRouter()


@router.post("/train")
async def train(request: TrainRequest, api_key: str = Depends(get_api_key)):
    """
    Full retrain — re-encodes every subintent from scratch.
    Includes 80/20 train/test split evaluation.
    Response fields: cvAccuracy (training data) and testAccuracy (held-out 20%).
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Base models are still loading. Retry in a moment.")
    if not request.workflowId:
        raise HTTPException(status_code=400, detail="workflowId is required.")
    if len(request.subIntents) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 subintents with examples to train.")

    try:
        logger.info(f"[FULL] Training started for '{request.workflowId}' with {len(request.subIntents)} subintents ...")
        clear_embedding_cache(request.workflowId)
        metadata = train_intent_classifier(
            request.workflowId,
            request.subIntents,
            request.augmentationFactor,
            request.entities,
        )
        
        # Trigger global shared training (Task 2)
        try:
            train_global_classifiers()
        except Exception as global_e:
            logger.error(f"Global training failed after workflow train: {global_e}")

        return {
            "success":  True,
            "message":  f"Full retrain complete for workflowId='{request.workflowId}'.",
            "training": metadata,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training failed for '{request.workflowId}': {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/train/incremental")
async def train_incremental(request: IncrementalTrainRequest, api_key: str = Depends(get_api_key)):
    """
    Incremental retrain — only re-encodes changed intents.
    Also runs 80/20 train/test split evaluation on the full combined dataset.
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Base models are still loading. Retry in a moment.")
    if not request.workflowId:
        raise HTTPException(status_code=400, detail="workflowId is required.")
    if len(request.subIntents) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 subintents with examples to train.")

    try:
        logger.info(f"[INCREMENTAL] Training started for '{request.workflowId}' with {len(request.subIntents)} subintents ...")
        metadata   = train_intent_classifier_incremental(
            request.workflowId,
            request.subIntents,
            request.augmentationFactor,
            request.entities,
        )
        
        # Trigger global shared training (Task 2)
        try:
            train_global_classifiers()
        except Exception as global_e:
            logger.error(f"Global training failed after incremental workflow train: {global_e}")

        changed    = metadata.get("changedIntents") or []
        cache_hits = metadata.get("cacheHits", 0)
        return {
            "success": True,
            "message": (
                f"Incremental retrain complete for workflowId='{request.workflowId}'. "
                f"{len(changed)} intent(s) re-encoded, {cache_hits} reused from cache."
            ),
            "training": metadata,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Incremental training failed for '{request.workflowId}': {e}")
        raise HTTPException(status_code=500, detail=f"Incremental training failed: {str(e)}")
