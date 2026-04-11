"""
Predictor — wraps the loaded model with validated, enriched prediction logic.
Auto-generated — edit with care.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FEATURE_NAMES: List[str] = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']


# ── Confidence interpretation thresholds ──────────────────────────────────────
def _interpret_confidence(confidence: float, prediction: Any) -> str:
    """Return a plain-English interpretation of the prediction confidence."""
    label = str(prediction)
    if confidence >= 0.90:
        return f"High confidence prediction: '{label}' ({confidence:.0%})."
    if confidence >= 0.70:
        return f"Moderate confidence prediction: '{label}' ({confidence:.0%}). Review edge cases."
    if confidence >= 0.50:
        return f"Low confidence prediction: '{label}' ({confidence:.0%}). Treat with caution."
    return (
        f"Very low confidence ({confidence:.0%}). The model is uncertain — "
        "manual review strongly recommended."
    )


def _validate_input(features: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
    """
    Validate a single feature dict and convert to numpy row vector.

    Returns:
        (feature_array, list_of_warnings)
    """
    warnings: List[str] = []
    row: List[float] = []

    for name in FEATURE_NAMES:
        value = features.get(name)
        if value is None:
            warnings.append(f"Feature '{name}' is missing — substituting 0.0.")
            value = 0.0
        if not isinstance(value, (int, float)):
            warnings.append(f"Feature '{name}' has unexpected type {type(value).__name__} — coercing.")
            try:
                value = float(value)
            except (ValueError, TypeError):
                warnings.append(f"Feature '{name}' could not be coerced — substituting 0.0.")
                value = 0.0
        if not np.isfinite(value):
            warnings.append(f"Feature '{name}' is {value} — substituting 0.0.")
            value = 0.0
        row.append(value)

    return np.array(row, dtype=float).reshape(1, -1), warnings


def predict_single(model_data: dict, features: Dict[str, float]) -> dict:
    """
    Run a validated, enriched single prediction.

    Returns a dict compatible with PredictionOutput.
    """
    t0 = time.perf_counter()

    model        = model_data["model"]
    preprocessor = model_data.get("preprocessor")
    feature_arr, input_warnings = _validate_input(features)

    # Preprocess
    if preprocessor is not None:
        import pandas as pd
        df = pd.DataFrame(feature_arr, columns=FEATURE_NAMES)
        feature_arr = preprocessor.transform(df)

    # Predict
    raw_pred = model.predict(feature_arr)
    prediction = raw_pred[0]
    if hasattr(prediction, "item"):
        prediction = prediction.item()

    # Probabilities
    class_probs: Optional[List[dict]] = None
    confidence   = 1.0

    if hasattr(model, "predict_proba"):
        proba_matrix = model.predict_proba(feature_arr)[0]
        classes      = getattr(model, "classes_", list(range(len(proba_matrix))))
        class_probs  = [
            {"label": str(c), "probability": round(float(p), 4)}
            for c, p in sorted(zip(classes, proba_matrix), key=lambda x: -x[1])
        ]
        confidence = float(np.max(proba_matrix))
    elif hasattr(model, "decision_function"):
        scores     = model.decision_function(feature_arr)[0]
        # Platt-scale the raw score as an approximate confidence
        exp_s      = np.exp(np.clip(scores, -500, 500))
        confidence = float(exp_s / (1 + exp_s)) if np.isscalar(scores) else 0.5

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "prediction":     prediction,
        "confidence":     round(confidence, 4),
        "interpretation": _interpret_confidence(confidence, prediction),
        "probabilities":  class_probs,
        "warnings":       input_warnings,
        "latency_ms":     latency_ms,
    }


def predict_batch(model_data: dict, inputs: List[Dict[str, float]]) -> dict:
    """
    Run validated batch predictions, returning per-row confidence scores.
    """
    t0 = time.perf_counter()
    model        = model_data["model"]
    preprocessor = model_data.get("preprocessor")

    rows:     List[List[float]] = []
    all_warnings: List[List[str]] = []

    for item in inputs:
        arr, w = _validate_input(item)
        rows.append(arr[0].tolist())
        all_warnings.append(w)

    feature_matrix = np.array(rows, dtype=float)

    if preprocessor is not None:
        import pandas as pd
        df = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES)
        feature_matrix = preprocessor.transform(df)

    raw_preds = model.predict(feature_matrix)
    predictions = [
        p.item() if hasattr(p, "item") else p for p in raw_preds
    ]

    confidences:  List[float] = []
    class_probs_all: Optional[List] = [] if hasattr(model, "predict_proba") else None

    if hasattr(model, "predict_proba"):
        proba_matrix = model.predict_proba(feature_matrix)
        classes      = getattr(model, "classes_", list(range(proba_matrix.shape[1])))
        for row_proba in proba_matrix:
            confidences.append(float(np.max(row_proba)))
            class_probs_all.append([
                {"label": str(c), "probability": round(float(p), 4)}
                for c, p in sorted(zip(classes, row_proba), key=lambda x: -x[1])
            ])
    else:
        confidences = [1.0] * len(predictions)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "predictions":   predictions,
        "confidences":   [round(c, 4) for c in confidences],
        "probabilities": class_probs_all,
        "count":         len(predictions),
        "warnings":      all_warnings,
        "latency_ms":    latency_ms,
    }


def explain_prediction(model_data: dict, features: Dict[str, float]) -> dict:
    """
    Approximate per-feature contributions using coefficient magnitude
    (linear models) or mean-decrease impurity (tree models).

    For linear models: contribution_i = coef_i × feature_i
    For tree models:   contribution_i = importance_i × feature_i (normalised)
    Falls back to a uniform attribution when neither is available.
    """
    model = model_data["model"]
    arr, _ = _validate_input(features)

    prediction = model.predict(arr)[0]
    if hasattr(prediction, "item"):
        prediction = prediction.item()

    contributions: Dict[str, float] = {}

    if hasattr(model, "coef_"):
        coef = np.array(model.coef_).ravel()
        n    = min(len(FEATURE_NAMES), len(coef), arr.shape[1])
        for i, name in enumerate(FEATURE_NAMES[:n]):
            contributions[name] = round(float(coef[i] * arr[0, i]), 6)

    elif hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        n   = min(len(FEATURE_NAMES), len(imp), arr.shape[1])
        for i, name in enumerate(FEATURE_NAMES[:n]):
            contributions[name] = round(float(imp[i] * arr[0, i]), 6)

    else:
        n = min(len(FEATURE_NAMES), arr.shape[1])
        for i, name in enumerate(FEATURE_NAMES[:n]):
            contributions[name] = round(float(arr[0, i] / n), 6)

    sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_pos = [k for k, v in sorted_contribs if v > 0][:3]
    top_neg = [k for k, v in sorted_contribs if v < 0][:3]

    explanation = "The prediction was primarily driven by: " + (
        ", ".join(top_pos) if top_pos else "no strongly positive features"
    )
    if top_neg:
        explanation += f". Features pulling against this prediction: {', '.join(top_neg)}."

    return {
        "prediction":            prediction,
        "feature_contributions": contributions,
        "top_positive":          top_pos,
        "top_negative":          top_neg,
        "explanation_text":      explanation,
    }


def check_drift(
    model_data: dict,
    features:   Dict[str, float],
    z_threshold: float = 3.0,
) -> dict:
    """
    Compare incoming feature values against the reference training distribution.

    A feature is flagged as drifted when its z-score w.r.t. the training mean
    and std exceeds *z_threshold*.

    Returns a DriftReport-compatible dict.
    """
    reference = model_data.get("reference_stats", {})
    if not reference:
        return {
            "drifted_features": [],
            "drift_scores":     {},
            "overall_drift":    "unknown",
            "recommendation":   "No reference statistics saved. Re-save the model with reference_stats.",
        }

    drift_scores: Dict[str, float] = {}
    drifted: List[str] = []

    for name, value in features.items():
        stats = reference.get(name)
        if stats is None:
            continue
        mean = stats.get("mean", 0.0)
        std  = stats.get("std",  1.0) or 1.0   # guard against zero std
        z    = abs((value - mean) / std)
        drift_scores[name] = round(z, 3)
        if z > z_threshold:
            drifted.append(name)

    n_drifted = len(drifted)
    if n_drifted == 0:
        level = "none"
        rec   = "Input is within the training distribution — prediction should be reliable."
    elif n_drifted <= 2:
        level = "mild"
        rec   = f"{n_drifted} feature(s) outside training distribution: {', '.join(drifted)}. Monitor closely."
    else:
        level = "severe"
        rec   = (
            f"{n_drifted} features are far outside training distribution: {', '.join(drifted)}. "
            "Prediction may be unreliable. Consider retraining."
        )

    return {
        "drifted_features": drifted,
        "drift_scores":     drift_scores,
        "overall_drift":    level,
        "recommendation":   rec,
    }
