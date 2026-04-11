"""
Pydantic schemas for the ML Prediction API.
Auto-generated — edit with care.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Input features for a single prediction."""
    Age: float = Field(..., description="Feature: Age")
    Sex: float = Field(..., description="Feature: Sex")
    ChestPainType: float = Field(..., description="Feature: ChestPainType")
    RestingBP: float = Field(..., description="Feature: RestingBP")
    Cholesterol: float = Field(..., description="Feature: Cholesterol")
    FastingBS: float = Field(..., description="Feature: FastingBS")
    RestingECG: float = Field(..., description="Feature: RestingECG")
    MaxHR: float = Field(..., description="Feature: MaxHR")
    ExerciseAngina: float = Field(..., description="Feature: ExerciseAngina")
    Oldpeak: float = Field(..., description="Feature: Oldpeak")
    ST_Slope: float = Field(..., description="Feature: ST_Slope")

    model_config = {
        "json_schema_extra": {
            "example": {"Age": 0.0, "Sex": 0.0, "ChestPainType": 0.0, "RestingBP": 0.0, "Cholesterol": 0.0}
        }
    }


class BatchPredictionInput(BaseModel):
    """Input features for batch predictions."""
    inputs: List[Dict[str, float]] = Field(
        ..., description="List of feature dicts, one per row."
    )


class ClassProbability(BaseModel):
    label: str
    probability: float


class PredictionOutput(BaseModel):
    """Prediction result for a single input."""
    prediction:      Any
    confidence:      float                     = Field(..., ge=0, le=1)
    interpretation:  str
    probabilities:   Optional[List[ClassProbability]] = None
    model_name:      str
    latency_ms:      Optional[float]           = None


class BatchPredictionOutput(BaseModel):
    """Prediction results for a batch of inputs."""
    predictions:     List[Any]
    confidences:     List[float]
    probabilities:   Optional[List[Optional[List[ClassProbability]]]] = None
    count:           int
    latency_ms:      Optional[float]           = None


class ExplainOutput(BaseModel):
    """Feature-contribution explanation for a single prediction."""
    prediction:       Any
    feature_contributions: Dict[str, float]
    top_positive:     List[str]
    top_negative:     List[str]
    explanation_text: str


class DriftReport(BaseModel):
    """Statistical drift report compared to the training distribution."""
    drifted_features: List[str]
    drift_scores:     Dict[str, float]
    overall_drift:    str      # "none" | "mild" | "severe"
    recommendation:   str


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_class:  Optional[str] = None
    has_proba:    bool          = False
