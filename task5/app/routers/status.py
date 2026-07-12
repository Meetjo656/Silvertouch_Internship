"""
app/routers/status.py — Model status and health endpoints.

  GET    /health           → Service health + loaded models summary
  GET    /training-status  → Per-workflow training metadata
  DELETE /training-status  → Remove a trained model (revert to zero-shot)
  GET    /versions         → List all versions for a workflow
  POST   /activate-version → Activate a specific version
  DELETE /version         → Delete a specific version
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends

import app.state as state
from app.core.persistence import model_path
from app.core.persistence import list_versions, activate_version, delete_version
from app.auth import get_api_key

logger = logging.getLogger("mybot-nlu")
router = APIRouter()


@router.get("/health")
async def health(api_key: str = Depends(get_api_key)):
    return {
        "status":       "ok" if state.models_loaded else "loading",
        "modelsLoaded": state.models_loaded,
        "baseModels": {
            "ner":     "spacy en_core_web_sm (CNN-based)" if state.spacy_nlp else "not loaded",
            "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "trainedWorkflows": {
            wf_id: {
                "mode":                 "trained",
                "intents":              state.trained_metadata_store.get(wf_id, {}).get("intents", []),
                "trainedAt":            state.trained_metadata_store.get(wf_id, {}).get("trainedAt"),
                "testAccuracy":         state.trained_metadata_store.get(wf_id, {}).get("testAccuracy"),
                "cvAccuracy":           state.trained_metadata_store.get(wf_id, {}).get("cvAccuracy"),
                "overfitWarning":       state.trained_metadata_store.get(wf_id, {}).get("overfitWarning", False),
                "entityMetrics":        state.trained_metadata_store.get(wf_id, {}).get("entityMetrics", {}),
                "totalTrainedEntities": state.trained_metadata_store.get(wf_id, {}).get("totalTrainedEntities", 0),
            }
            for wf_id in state.trained_classifiers
        },
        "totalTrainedWorkflows":         len(state.trained_classifiers),
        "totalTrainedEntityClassifiers": sum(len(e) for e in state.trained_entity_classifiers.values()),
        "error": state.model_load_error,
    }


@router.get("/training-status")
async def training_status(workflowId: Optional[str] = Query(default=None)):
    """
    - ?workflowId=X → status of a specific workflow model
    - No param      → status of ALL trained models
    """
    if workflowId:
        return {
            "workflowId":      workflowId,
            "trained":         workflowId in state.trained_classifiers,
            "modelFile":       str(model_path(workflowId)),
            "modelFileExists": model_path(workflowId).exists(),
            "metadata":        state.trained_metadata_store.get(workflowId),
        }

    all_status = {
        wf_id: {
            "trained":         True,
            "modelFile":       str(model_path(wf_id)),
            "modelFileExists": model_path(wf_id).exists(),
            "metadata":        state.trained_metadata_store.get(wf_id),
        }
        for wf_id in state.trained_classifiers
    }
    return {"totalTrained": len(state.trained_classifiers), "workflows": all_status}


@router.delete("/training-status")
async def delete_trained_model(workflowId: str = Query(...), api_key: str = Depends(get_api_key)):
    """Delete a trained model — reverts that workflow to zero-shot mode."""
    if workflowId not in state.trained_classifiers:
        raise HTTPException(status_code=404, detail=f"No trained model found for workflowId='{workflowId}'")

    state.trained_classifiers.pop(workflowId, None)
    state.trained_label_encoders.pop(workflowId, None)
    state.trained_metadata_store.pop(workflowId, None)

    pkl = model_path(workflowId)
    if pkl.exists():
        pkl.unlink()

    from app.core.model_registry import registry
    registry.remove_model(workflowId)

    return {"success": True, "message": f"Trained model for '{workflowId}' deleted. Reverted to zero-shot mode."}


# ─── Version Management Endpoints ────────────────────────────────────────────────

@router.get("/versions")
async def get_versions(workflowId: str = Query(...), api_key: str = Depends(get_api_key)):
    """
    List all available versions for a workflow.
    Returns list with: version, cvAccuracy, testAccuracy, timestamp, isActive, filename
    """
    try:
        versions = list_versions(workflowId)
        return {
            "success": True,
            "workflowId": workflowId,
            "versions": versions,
            "totalVersions": len(versions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list versions: {str(e)}")


@router.post("/activate-version")
async def activate_version_endpoint(
    workflowId: str = Query(...),
    version: int = Query(..., description="Version number to activate (0 for current)"),
    api_key: str = Depends(get_api_key)
):
    """
    Activate a specific version of the workflow model.
    - version=0 keeps current active
    - version>0 activates that specific backup version
    """
    try:
        activate_version(workflowId, version)

        # Reload into memory
        from app.core.persistence import load_trained_classifier
        load_trained_classifier(workflowId)

        return {
            "success": True,
            "message": f"Activated version {version} for workflow '{workflowId}'",
            "workflowId": workflowId,
            "activatedVersion": version
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate version: {str(e)}")


@router.delete("/version")
async def delete_version_endpoint(
    workflowId: str = Query(...),
    version: int = Query(..., description="Version number to delete (cannot delete active version 0)"),
    api_key: str = Depends(get_api_key)
):
    """Delete a specific backup version (cannot delete active version)."""
    try:
        delete_version(workflowId, version)
        return {
            "success": True,
            "message": f"Deleted version {version} for workflow '{workflowId}'",
            "workflowId": workflowId,
            "deletedVersion": version
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete version: {str(e)}")
