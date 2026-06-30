"""
app/models.py — Pydantic request/response models for the NLU API.
"""

from typing import Optional
from pydantic import BaseModel


class EntityDef(BaseModel):
    name:     str
    label:    Optional[str]       = ""
    type:     Optional[str]       = "regex"   # "regex" | "train"
    regex:    Optional[str]       = ""
    examples: Optional[list[str]] = []
    minLength:       Optional[int] = None
    maxLength:       Optional[int] = None
    validationRegex: Optional[str] = None


class SubIntent(BaseModel):
    """A single subintent definition with training examples."""
    name:             Optional[str]       = ""
    label:            Optional[str]       = ""
    description:      Optional[str]       = ""
    examples:         Optional[list[str]] = []
    # list of strings (legacy) or list of dicts (new format)
    requiredEntities: Optional[list]      = []


class TrainRequest(BaseModel):
    """Full retrain — re-encodes every subintent from scratch."""
    workflowId:         str
    subIntents:         list[SubIntent]
    entities:           Optional[list[EntityDef]] = []
    augmentationFactor: int = 2


class IncrementalTrainRequest(BaseModel):
    """Incremental retrain — only re-encodes changed subintents."""
    workflowId:         str
    subIntents:         list[SubIntent]   # full list, NOT just changed ones
    entities:           Optional[list[EntityDef]] = []
    augmentationFactor: int = 2


class AnalyzeRequest(BaseModel):
    """Full NLU pipeline request (intent + entities)."""
    text:                Optional[str]             = ""
    sessionId:           Optional[str]             = "default"
    languageCode:        Optional[str]             = "en"
    workflowId:          Optional[str]             = "welcome-intent"
    subIntents:          Optional[list[SubIntent]] = []   # zero-shot fallback only
    confidenceThreshold: Optional[float]           = 0.30
    customPatterns:      Optional[dict]            = {}


class IntentRequest(BaseModel):
    """Intent/subintent detection only."""
    text:                Optional[str]             = ""
    workflowId:          Optional[str]             = "welcome-intent"
    subIntents:          Optional[list[SubIntent]] = []
    confidenceThreshold: Optional[float]           = 0.30


class EntityRequest(BaseModel):
    """Entity extraction only."""
    text:           Optional[str]            = ""
    customPatterns: Optional[dict[str, str]] = {}
    entityConfigs:  Optional[list]           = []
