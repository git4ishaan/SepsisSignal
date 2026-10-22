"""
Single risk prediction function for SepsisSignal.
Called by the line server, FastAPI, and the dashboard.
Queries SQLite, builds causal features, runs the model, and guarantees immutability.
"""

import json
import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np

from src.features import FEATURE_NAMES, extract_features_point_in_time, load_imputer_values
from src.store import get_observations_up_to, get_prediction, save_prediction

MODEL_NAME = "logistic_regression_baseline"
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model.joblib")
THRESHOLD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "threshold.json")

_CACHED_MODEL = None
_CACHED_THRESHOLD = 0.2500

FEATURE_LABELS = {
    "hr_val": "Heart rate level",
    "hr_missing": "Heart rate missingness",
    "hr_time_since": "Hours since heart rate",
    "hr_delta": "Heart rate change",
    "sbp_val": "Systolic pressure level",
    "sbp_missing": "Blood pressure missingness",
    "sbp_time_since": "Hours since blood pressure",
    "sbp_delta": "Systolic pressure change",
    "resp_val": "Respiratory rate level",
    "resp_missing": "Respiratory rate missingness",
    "resp_time_since": "Hours since respiratory rate",
    "resp_delta": "Respiratory rate change",
    "temp_val": "Temperature level",
    "temp_missing": "Temperature missingness",
    "temp_time_since": "Hours since temperature",
    "temp_delta": "Temperature change",
    "wbc_val": "WBC level",
    "wbc_missing": "WBC missingness",
    "wbc_time_since": "Hours since WBC",
    "wbc_delta": "WBC change",
    "hour": "Stay hour",
    "current_empty_count": "Empty signals",
}


def get_model():
    """Load and cache the trained scikit-learn model pipeline."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if os.path.exists(MODEL_PATH):
            _CACHED_MODEL = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    return _CACHED_MODEL


def get_threshold() -> float:
    """Load and cache the validation-selected threshold."""
    global _CACHED_THRESHOLD
    if os.path.exists(THRESHOLD_PATH):
        try:
            with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _CACHED_THRESHOLD = float(data.get("threshold", 0.2500))
        except Exception:
            pass
    return _CACHED_THRESHOLD


def get_model_contributions(feature_vec: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
    """Return the largest positive feature contributions to the model logit."""
    try:
        model = get_model()
        scaled = model.named_steps["scaler"].transform(feature_vec)[0]
        coefficients = model.named_steps["clf"].coef_[0]
        contributions = [float(value * coefficient) for value, coefficient in zip(scaled, coefficients)]
        ranked = sorted(
            (
                {
                    "feature": name,
                    "label": FEATURE_LABELS.get(name, name),
                    "contribution": round(contribution, 3),
                }
                for name, contribution in zip(FEATURE_NAMES, contributions)
                if contribution > 0
            ),
            key=lambda item: item["contribution"],
            reverse=True,
        )
        return ranked[:top_k]
    except Exception:
        # Attribution is supplemental; it must never block the risk estimate.
        return []


def predict(
    patient_id: str,
    hour: int,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes (or retrieves existing) sepsis risk score for (patient_id, hour).
    
    Guarantees:
    - Immutability: If a prediction already exists for this (patient_id, hour), returns it immediately.
    - Causal isolation: Uses ONLY observations with hour <= requested hour.
    - Consistency: Probability formatted to 4 decimal places.
    - Zero data leak: Model fit on other patients, threshold chosen on validation.
    """
    # 1. Immutability check: return stored prediction if already computed
    kwargs = {"db_path": db_path} if db_path else {}
    existing = get_prediction(patient_id, hour, **kwargs)
    if existing is not None:
        observations = get_observations_up_to(
            patient_id,
            hour,
            received_before=existing["created_at"],
            **kwargs,
        )
        feature_vec, _ = extract_features_point_in_time(observations, target_hour=hour)
        return {
            "patient_id": existing["patient_id"],
            "hour": int(existing["hour"]),
            "probability": round(float(existing["probability"]), 4),
            "alert": int(existing["alert"]),
            "empty_count": int(existing["empty_count"]),
            "model_name": existing["model_name"],
            "threshold": get_threshold(),
            "created_at": existing["created_at"],
            "live_path_label": "Live path: Logistic Regression baseline (research prototype)",
            "contributors": get_model_contributions(feature_vec),
        }

    # 2. Fetch causal observations up to requested hour
    observations = get_observations_up_to(patient_id, hour, **kwargs)

    # 3. Extract causal point-in-time features
    feature_vec, feature_dict = extract_features_point_in_time(observations, target_hour=hour)
    empty_count = int(feature_dict.get("current_empty_count", 0))

    # 4. Predict probability with model
    model = get_model()
    proba = float(model.predict_proba(feature_vec)[0, 1])
    contributors = get_model_contributions(feature_vec)
    proba_rounded = round(proba, 4)

    # 5. Evaluate alert threshold
    threshold = get_threshold()
    alert = 1 if proba_rounded >= threshold else 0

    # 6. Store prediction once (immutable)
    saved = save_prediction(
        patient_id=patient_id,
        hour=hour,
        probability=proba_rounded,
        alert=alert,
        empty_count=empty_count,
        model_name=MODEL_NAME,
        **kwargs,
    )

    return {
        "patient_id": patient_id,
        "hour": hour,
        "probability": proba_rounded,
        "alert": alert,
        "empty_count": empty_count,
        "model_name": MODEL_NAME,
        "threshold": threshold,
        "created_at": saved.get("created_at"),
        "live_path_label": "Live path: Logistic Regression baseline (research prototype)",
        "contributors": contributors,
    }
