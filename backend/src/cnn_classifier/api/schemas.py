"""Pydantic response models for the serving API (Task 6)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: Literal["ok"] = "ok"


class PredictionItem(BaseModel):
    """One class prediction with its softmax confidence."""

    class_name: str
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    """Ranked predictions for one image, plus serving metadata."""

    predictions: list[PredictionItem]
    latency_ms: float = Field(ge=0.0)
    model: str


class ModelInfoResponse(BaseModel):
    """Metadata about the checkpoint currently served."""

    model: str
    input_size: str
    classes: list[str]
    device: str
    trained_at: datetime | None = Field(default=None)
