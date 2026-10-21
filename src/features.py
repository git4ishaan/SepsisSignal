"""
Single point-in-time causal feature extractor for SepsisSignal.
Used identically by offline training and live serving.
Only observations with hour <= T are ever accessed. No future lookahead.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Canonical signal list from the jury protocol (DATA_CONTRACT.md)
SIGNALS = ["hr", "sbp", "resp", "temp", "wbc"]

# Default training-set median fill values if imputer.json is not yet loaded
DEFAULT_IMPUTER = {
    "hr": 84.0,
    "sbp": 118.0,
    "resp": 18.0,
    "temp": 36.8,
    "wbc": 10.5,
}

SENTINEL_TIME_SINCE = 99.0


def get_feature_names() -> List[str]:
    """
    Returns the deterministic ordered list of feature names.
    For each of the 5 signals:
      - {sig}_val: current measurement or causal forward-fill
      - {sig}_missing: 1 if unmeasured at hour T, 0 if measured
      - {sig}_time_since: hours since last measurement (0 if measured at T, 99.0 if never)
      - {sig}_delta: change from previous measurement (0.0 if <= 1 measurement)
    Plus context:
      - hour: current hour of stay
      - current_empty_count: count of missing signals in this hour (0 to 5)
    Total = 5*4 + 2 = 22 features.
    """
    names = []
    for sig in SIGNALS:
        names.extend([
            f"{sig}_val",
            f"{sig}_missing",
            f"{sig}_time_since",
            f"{sig}_delta",
        ])
    names.extend(["hour", "current_empty_count"])
    return names


FEATURE_NAMES = get_feature_names()


def load_imputer_values(imputer_path: Optional[str] = None) -> Dict[str, float]:
    """Load training-set fill values from imputer.json."""
    if imputer_path is None:
        imputer_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "imputer.json"
        )
    if os.path.exists(imputer_path):
        try:
            with open(imputer_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                return {k.lower(): float(v) for k, v in raw.items()}
        except Exception:
            pass
    return DEFAULT_IMPUTER.copy()


def extract_features_point_in_time(
    observations: List[Dict[str, Any]],
    target_hour: int,
    imputer: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extract causal features for a single patient at target_hour T.
    
    Parameters
    ----------
    observations : List[Dict[str, Any]]
        List of observation dictionaries for this patient.
        May include raw keys 'hr', 'sbp', 'resp', 'temp', 'wbc' and 'hour'.
    target_hour : int
        The hour T to predict for.
    imputer : Optional[Dict[str, float]]
        Baseline training-set fill values.

    Returns
    -------
    feature_vector : np.ndarray (shape: (1, 22), dtype: float32)
    feature_dict : Dict[str, float] mapping feature name to value
    """
    if imputer is None:
        imputer = load_imputer_values()

    # 1. Filter rows causally: only rows with hour <= target_hour
    causal_rows = [
        row for row in observations
        if int(row["hour"]) <= target_hour
    ]
    # Sort strictly by hour ascending
    causal_rows.sort(key=lambda r: int(r["hour"]))

    # Find the observation row for target_hour if present
    current_row = None
    for r in causal_rows:
        if int(r["hour"]) == target_hour:
            current_row = r
            break

    features: Dict[str, float] = {}

    current_empty_count = 0
    for sig in SIGNALS:
        history_vals: List[Tuple[int, float]] = []
        for r in causal_rows:
            v = r.get(sig)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                history_vals.append((int(r["hour"]), float(v)))

        # Was it measured at target_hour?
        measured_now = False
        val_now = None
        if current_row is not None:
            raw_v = current_row.get(sig)
            if raw_v is not None and not (isinstance(raw_v, float) and np.isnan(raw_v)):
                measured_now = True
                val_now = float(raw_v)

        if not measured_now:
            current_empty_count += 1

        # Causal forward-fill
        if measured_now and val_now is not None:
            sig_val = val_now
            time_since = 0.0
        elif len(history_vals) > 0:
            last_hr, last_v = history_vals[-1]
            sig_val = last_v
            time_since = float(target_hour - last_hr)
        else:
            # Never measured up to target_hour: use training-set default
            sig_val = imputer.get(sig, DEFAULT_IMPUTER[sig])
            time_since = SENTINEL_TIME_SINCE

        # Delta from previous measurement
        if len(history_vals) >= 2:
            sig_delta = history_vals[-1][1] - history_vals[-2][1]
        else:
            sig_delta = 0.0

        missing_flag = 0.0 if measured_now else 1.0

        features[f"{sig}_val"] = float(sig_val)
        features[f"{sig}_missing"] = float(missing_flag)
        features[f"{sig}_time_since"] = float(time_since)
        features[f"{sig}_delta"] = float(sig_delta)

    features["hour"] = float(target_hour)
    features["current_empty_count"] = float(current_empty_count)

    # Format into numpy array in deterministic FEATURE_NAMES order
    vector = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
    return vector, features
