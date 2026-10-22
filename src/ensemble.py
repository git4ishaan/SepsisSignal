"""
Soft-Voting Ensemble Module for SepsisSignal (Research Prototype)

Combines predictions from three diverse candidate models:
1. Baseline Logistic Regression (StandardScaler + LogisticRegression) - live path
2. Gradient Boosted Decision Trees (XGBoostClassifier) - offline comparison
3. Recurrent Sequence Model (PyTorch GRU) - offline sequence benchmark

Equal-weight soft voting formulation:
  ensemble_probability = (lr_probability + xgb_probability + gru_probability) / 3.0

Guarantees:
- Strict causal feature extraction (only observations with hour <= target_hour are accessed)
- Explicit input verification and finite probability guarantees in [0.0, 1.0]
- Zero mutation of production live path (src/predict.py remains independent)
- Zero retraining or mutation of frozen model artifacts

Safety Disclaimer:
Research prototype, not for clinical decision-making.
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import torch

from scripts.train_lstm import SepsisRNN
from src.features import FEATURE_NAMES, extract_features_point_in_time, load_imputer_values

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class SepsisEnsemble:
    """
    Manages loading and causal inference across Logistic Regression, XGBoost, and PyTorch GRU.
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.lr_path = os.path.join(self.models_dir, "model.joblib")
        self.xgb_path = os.path.join(self.models_dir, "xgboost_model.joblib")
        self.gru_path = os.path.join(self.models_dir, "lstm_model.pt")
        self.imputer_path = os.path.join(self.models_dir, "imputer.json")

        self.lr_model = None
        self.xgb_model = None
        self.gru_model = None
        self.train_mean = None
        self.train_std = None
        self.imputer = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load all model artifacts and preprocessing parameters with validation."""
        # 1. Logistic Regression
        if not os.path.exists(self.lr_path):
            raise FileNotFoundError(f"Logistic Regression artifact not found at: {self.lr_path}")
        self.lr_model = joblib.load(self.lr_path)

        # 2. XGBoost
        if not os.path.exists(self.xgb_path):
            raise FileNotFoundError(f"XGBoost artifact not found at: {self.xgb_path}")
        self.xgb_model = joblib.load(self.xgb_path)

        # 3. PyTorch GRU
        if not os.path.exists(self.gru_path):
            raise FileNotFoundError(f"PyTorch GRU checkpoint not found at: {self.gru_path}")
        ckpt = torch.load(self.gru_path, weights_only=False)

        required_ckpt_keys = ["state_dict", "train_mean", "train_std", "input_dim", "hidden_dim", "num_layers", "rnn_type"]
        for k in required_ckpt_keys:
            if k not in ckpt:
                raise KeyError(f"GRU checkpoint missing required metadata key: '{k}'")

        self.train_mean = np.array(ckpt["train_mean"], dtype=np.float32)
        self.train_std = np.array(ckpt["train_std"], dtype=np.float32)
        # Avoid division by zero in normalization
        self.train_std[self.train_std < 1e-6] = 1.0

        self.gru_model = SepsisRNN(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            rnn_type=ckpt["rnn_type"],
        )
        self.gru_model.load_state_dict(ckpt["state_dict"])
        self.gru_model.eval()

        # 4. Imputer
        self.imputer = load_imputer_values(self.imputer_path)

    @staticmethod
    def _validate_probability(prob: float, model_name: str) -> float:
        """Verify that output probability is finite and within [0.0, 1.0]."""
        if not isinstance(prob, (float, np.floating)):
            raise TypeError(f"{model_name} returned non-float probability: {type(prob)}")
        if not math.isfinite(prob):
            raise ValueError(f"{model_name} produced non-finite probability: {prob}")
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"{model_name} probability out of bounds [0, 1]: {prob}")
        return float(prob)

    def predict_point_in_time(
        self,
        observations: List[Dict[str, Any]],
        target_hour: int,
        max_gru_window: int = 48,
    ) -> Dict[str, Any]:
        """
        Run causal point-in-time inference for a patient stay at target_hour across all 3 models.
        Strictly accesses observations with hour <= target_hour.
        """
        causal_obs = [r for r in observations if int(r.get("hour", 0)) <= target_hour]
        if not causal_obs:
            raise ValueError(f"No observations available at or before target_hour {target_hour}")

        # 1. Extract 22-dimensional feature vector at target_hour
        vec, feat_dict = extract_features_point_in_time(
            causal_obs, target_hour=target_hour, imputer=self.imputer
        )
        if vec.shape != (1, 22) or not np.all(np.isfinite(vec)):
            raise ValueError(f"Extracted feature vector is malformed or non-finite: shape={vec.shape}")

        # 2. Logistic Regression probability
        p_lr_raw = float(self.lr_model.predict_proba(vec)[0, 1])
        p_lr = self._validate_probability(p_lr_raw, "LogisticRegression")

        # 3. XGBoost probability
        p_xgb_raw = float(self.xgb_model.predict_proba(vec)[0, 1])
        p_xgb = self._validate_probability(p_xgb_raw, "XGBoost")

        # 4. PyTorch GRU probability (causal sequence up to target_hour, capped at max_gru_window)
        # Sort causal observations chronologically
        sorted_causal = sorted(causal_obs, key=lambda r: int(r.get("hour", 0)))
        window = sorted_causal[-max_gru_window:] if len(sorted_causal) > max_gru_window else sorted_causal

        seq_feats = []
        for r in window:
            h = int(r["hour"])
            v_step, _ = extract_features_point_in_time(sorted_causal, target_hour=h, imputer=self.imputer)
            seq_feats.append(v_step.flatten())

        seq_arr = np.array(seq_feats, dtype=np.float32)
        norm_seq = (seq_arr - self.train_mean) / self.train_std

        t_x = torch.from_numpy(norm_seq).unsqueeze(0)  # Shape: (1, seq_len, 22)
        with torch.no_grad():
            logits = self.gru_model(t_x)
            # Logit at the final timestep (target_hour)
            logit_target = float(logits[0, -1].item())
            p_gru_raw = 1.0 / (1.0 + math.exp(-logit_target))

        p_gru = self._validate_probability(p_gru_raw, "PyTorch GRU")

        # 5. Equal-weight soft-voting ensemble
        p_ensemble_raw = (p_lr + p_xgb + p_gru) / 3.0
        p_ensemble = self._validate_probability(p_ensemble_raw, "Ensemble")

        return {
            "target_hour": target_hour,
            "lr_probability": round(p_lr, 4),
            "xgb_probability": round(p_xgb, 4),
            "gru_probability": round(p_gru, 4),
            "ensemble_probability": round(p_ensemble, 4),
            "empty_count": int(feat_dict.get("current_empty_count", 0)),
        }

    def predict_patient_stay_aligned(
        self,
        raw_obs: List[Dict[str, Any]],
        max_window: int = 48,
    ) -> Dict[str, np.ndarray]:
        """
        Fast batch causal evaluation for a patient's stay, computing strictly aligned
        predictions across LR, XGBoost, and PyTorch GRU for every hour.
        """
        if not raw_obs:
            return {
                "hours": np.array([], dtype=np.int32),
                "labels": np.array([], dtype=np.int32),
                "p_lr": np.array([], dtype=np.float32),
                "p_xgb": np.array([], dtype=np.float32),
                "p_gru": np.array([], dtype=np.float32),
                "p_ensemble": np.array([], dtype=np.float32),
            }

        sorted_obs = sorted(raw_obs, key=lambda r: int(r["hour"]))
        n_hours = len(sorted_obs)

        # Precompute point-in-time features for all hours
        feats_list = []
        labels_list = []
        hours_list = []

        for idx in range(n_hours):
            h = int(sorted_obs[idx]["hour"])
            hours_list.append(h)
            lbl = int(sorted_obs[idx].get("label", 0))
            labels_list.append(lbl)

            vec, _ = extract_features_point_in_time(sorted_obs[: idx + 1], target_hour=h, imputer=self.imputer)
            feats_list.append(vec.flatten())

        X_all = np.array(feats_list, dtype=np.float32)

        # Tabular inferences
        p_lr = self.lr_model.predict_proba(X_all)[:, 1].astype(np.float32)
        p_xgb = self.xgb_model.predict_proba(X_all)[:, 1].astype(np.float32)

        # GRU recurrent sequence inference
        # To strictly replicate the causal sequence model:
        # Window of up to max_window hours ending at each index
        p_gru = np.zeros(n_hours, dtype=np.float32)

        # If stay <= max_window, we can evaluate in a single recurrent forward pass
        if n_hours <= max_window:
            norm_X = (X_all - self.train_mean) / self.train_std
            t_x = torch.from_numpy(norm_X).unsqueeze(0)
            with torch.no_grad():
                logits = self.gru_model(t_x)
                p_gru = torch.sigmoid(logits[0]).numpy().astype(np.float32)
        else:
            # For stays > max_window:
            # First max_window steps
            norm_first = (X_all[:max_window] - self.train_mean) / self.train_std
            with torch.no_grad():
                logits_first = self.gru_model(torch.from_numpy(norm_first).unsqueeze(0))
                p_gru[:max_window] = torch.sigmoid(logits_first[0]).numpy()

            # For subsequent steps, run sliding window of size max_window
            for i in range(max_window, n_hours):
                window_feats = X_all[i - max_window + 1 : i + 1]
                norm_w = (window_feats - self.train_mean) / self.train_std
                with torch.no_grad():
                    logits_w = self.gru_model(torch.from_numpy(norm_w).unsqueeze(0))
                    p_gru[i] = float(torch.sigmoid(logits_w[0, -1]).item())

        p_ensemble = ((p_lr + p_xgb + p_gru) / 3.0).astype(np.float32)

        return {
            "hours": np.array(hours_list, dtype=np.int32),
            "labels": np.array(labels_list, dtype=np.int32),
            "p_lr": p_lr,
            "p_xgb": p_xgb,
            "p_gru": p_gru,
            "p_ensemble": p_ensemble,
        }
