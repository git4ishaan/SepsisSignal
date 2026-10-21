"""
SepsisSignal: Offline Sequence Benchmark Training Script (PyTorch LSTM / GRU).

Strictly follows project rules:
1. OFFLINE ONLY: Does NOT modify, overwrite, or touch models/model.joblib.
   Saves sequence model separately as models/lstm_model.pt.
2. COHORT SPLITS & DATA: Reuses exact patient splits from models/splits/*.txt.
3. CAUSAL LOGIC: Zero future lookahead. Causal padding and truncation up to max_window hours.
4. LOSS & THRESHOLD: BCEWithLogitsLoss with positive class weighting.
   Alert threshold chosen strictly on validation split.
5. METRICS & OUTPUTS: Appends test metrics into models/results/metrics.json without overwriting baseline.
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import extract_features_point_in_time, FEATURE_NAMES


class SepsisSequenceDataset(Dataset):
    """
    Dataset of causal hourly sequences per patient stay.
    Each item contains:
      - features: Tensor of shape (seq_len, 22)
      - labels: Tensor of shape (seq_len,)
      - length: int (seq_len)
    """

    def __init__(self, sequences: List[Tuple[np.ndarray, np.ndarray]]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        feats, labels = self.sequences[idx]
        return torch.from_numpy(feats), torch.from_numpy(labels), len(labels)


def pad_collate_fn(batch):
    """Pad sequences to max sequence length in batch."""
    features_list, labels_list, lengths = zip(*batch)
    max_len = max(lengths)
    batch_size = len(batch)
    feat_dim = features_list[0].shape[1]

    padded_features = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    padded_labels = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (f, l, length) in enumerate(zip(features_list, labels_list, lengths)):
        padded_features[i, :length] = f
        padded_labels[i, :length] = l
        mask[i, :length] = True

    return padded_features, padded_labels, mask, torch.tensor(lengths, dtype=torch.long)


class SepsisRNN(nn.Module):
    """Causal Recurrent Sequence Classifier (LSTM or GRU)."""

    def __init__(
        self,
        input_dim: int = 22,
        hidden_dim: int = 48,
        num_layers: int = 1,
        rnn_type: str = "gru",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.rnn = nn.GRU(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_dim)
        returns logits: (batch_size, seq_len)
        """
        h = self.act(self.proj(x))
        out, _ = self.rnn(h)
        logits = self.head(out).squeeze(-1)
        return logits


def parse_patient_psv_fast(file_path: str) -> List[Dict[str, Any]]:
    """Fast line-by-line parser for PhysioNet PSV files."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return []

    header = lines[0].split("|")
    try:
        hr_idx = header.index("HR")
        sbp_idx = header.index("SBP")
        resp_idx = header.index("Resp")
        temp_idx = header.index("Temp")
        wbc_idx = header.index("WBC")
        lbl_idx = header.index("SepsisLabel") if "SepsisLabel" in header else -1
    except ValueError:
        return []

    rows = []
    for h, line in enumerate(lines[1:], start=1):
        parts = line.split("|")

        def _val(idx):
            if idx >= len(parts):
                return None
            val = parts[idx]
            return float(val) if val and val != "NaN" else None

        rows.append({
            "hour": h,
            "hr": _val(hr_idx),
            "sbp": _val(sbp_idx),
            "resp": _val(resp_idx),
            "temp": _val(temp_idx),
            "wbc": _val(wbc_idx),
            "label": int(parts[lbl_idx]) if lbl_idx >= 0 and lbl_idx < len(parts) and parts[lbl_idx] not in ("", "NaN") else 0,
        })
    return rows


def build_patient_sequence(
    raw_obs: List[Dict[str, Any]],
    imputer: Dict[str, float],
    max_window: int = 48,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build causal feature matrix and label vector for a patient stay."""
    if len(raw_obs) > max_window:
        raw_obs = raw_obs[-max_window:]

    feats = []
    labels = []
    for r in raw_obs:
        vec, _ = extract_features_point_in_time(
            raw_obs, target_hour=int(r["hour"]), imputer=imputer
        )
        feats.append(vec.flatten())
        labels.append(r["label"])

    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.float32)


def load_split_ids(output_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """Load exact patient split IDs generated by baseline."""
    train_file = os.path.join(output_dir, "splits", "train_patients.txt")
    val_file = os.path.join(output_dir, "splits", "val_patients.txt")
    test_file = os.path.join(output_dir, "splits", "test_patients.txt")

    for f_path in [train_file, val_file, test_file]:
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"Missing split file: {f_path}. Run train_baseline.py first.")

    with open(train_file, "r", encoding="utf-8") as f:
        train_ids = [line.strip() for line in f if line.strip()]
    with open(val_file, "r", encoding="utf-8") as f:
        val_ids = [line.strip() for line in f if line.strip()]
    with open(test_file, "r", encoding="utf-8") as f:
        test_ids = [line.strip() for line in f if line.strip()]

    return train_ids, val_ids, test_ids


def load_imputer(output_dir: str) -> Dict[str, float]:
    """Load training-set medians from models/imputer.json."""
    imputer_file = os.path.join(output_dir, "imputer.json")
    if not os.path.exists(imputer_file):
        raise FileNotFoundError(f"Missing imputer file: {imputer_file}. Run train_baseline.py first.")
    with open(imputer_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k.lower(): float(v) for k, v in raw.items()}


def extract_cohort_sequences(
    patient_files: Dict[str, str],
    pids: List[str],
    imputer: Dict[str, float],
    max_window: int = 48,
    cohort_name: str = "cohort",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract causal sequence representations for all patients in a cohort."""
    sequences = []
    total = len(pids)
    print(f"[*] Extracting causal sequences for {cohort_name} ({total} patients)...")

    for i, pid in enumerate(pids):
        if (i + 1) % 10000 == 0 or (i + 1) == total:
            print(f"    Processed {i + 1}/{total} patients in {cohort_name}...")

        file_path = patient_files.get(pid)
        if not file_path or not os.path.exists(file_path):
            continue

        raw_obs = parse_patient_psv_fast(file_path)
        if not raw_obs:
            continue

        feats, labels = build_patient_sequence(raw_obs, imputer, max_window=max_window)
        sequences.append((feats, labels))

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Train SepsisSignal Offline Sequence Model (PyTorch LSTM/GRU)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser(r"~\Desktop\sepsisdataset"),
        help="Directory containing PhysioNet .psv files (default: $HOME/Desktop/sepsisdataset)",
    )
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for model artifacts")
    parser.add_argument("--rnn_type", type=str, default="gru", choices=["lstm", "gru"], help="Sequence architecture (lstm or gru)")
    parser.add_argument("--hidden_dim", type=int, default=48, help="Hidden dimension of recurrent layer")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of recurrent layers")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--max_window", type=int, default=48, help="Maximum causal window length (hours)")
    parser.add_argument("--max_train_patients", type=int, default=0, help="Optional subsample of train patients (0 = all)")
    parser.add_argument("--max_val_patients", type=int, default=0, help="Optional subsample of val patients (0 = all)")
    parser.add_argument("--max_test_patients", type=int, default=0, help="Optional subsample of test patients (0 = all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] PyTorch training device: {device}")

    # 1. Discover PSV files
    print(f"[*] Scanning for patient PSV files in {args.data_dir}...")
    psv_files = glob.glob(os.path.join(args.data_dir, "**", "*.psv"), recursive=True)
    if not psv_files:
        print(f"[ERROR] No .psv files found in {args.data_dir}")
        sys.exit(1)

    patient_files = {os.path.splitext(os.path.basename(f))[0]: f for f in psv_files}
    print(f"[*] Found {len(patient_files)} total patient PSV files.")

    # 2. Load exact split IDs
    train_ids, val_ids, test_ids = load_split_ids(args.output_dir)
    print(f"[*] Loaded exact patient splits from {args.output_dir}/splits/:")
    print(f"    Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    if args.max_train_patients > 0:
        train_ids = train_ids[: args.max_train_patients]
        print(f"    [!] Subsampled train patients to: {len(train_ids)}")
    if args.max_val_patients > 0:
        val_ids = val_ids[: args.max_val_patients]
        print(f"    [!] Subsampled val patients to: {len(val_ids)}")
    if args.max_test_patients > 0:
        test_ids = test_ids[: args.max_test_patients]
        print(f"    [!] Subsampled test patients to: {len(test_ids)}")

    # 3. Load baseline imputer
    imputer = load_imputer(args.output_dir)
    print(f"[*] Loaded baseline imputer medians: {imputer}")

    # 4. Extract sequences
    t0 = time.time()
    train_seqs = extract_cohort_sequences(patient_files, train_ids, imputer, max_window=args.max_window, cohort_name="TRAIN")
    val_seqs = extract_cohort_sequences(patient_files, val_ids, imputer, max_window=args.max_window, cohort_name="VALIDATION")
    test_seqs = extract_cohort_sequences(patient_files, test_ids, imputer, max_window=args.max_window, cohort_name="TEST")
    print(f"[*] Feature extraction completed in {time.time() - t0:.1f}s.")

    # 5. Fit Feature Normalizer STRICTLY on TRAIN sequences
    print("[*] Computing feature normalization strictly from TRAIN cohort...")
    all_train_feats = np.concatenate([s[0] for s in train_seqs], axis=0)
    train_mean = np.mean(all_train_feats, axis=0)
    train_std = np.std(all_train_feats, axis=0)
    train_std[train_std < 1e-6] = 1.0  # avoid division by zero

    # Normalize sequences
    def _normalize(seqs):
        normalized = []
        for feats, labels in seqs:
            norm_feats = (feats - train_mean) / train_std
            normalized.append((norm_feats.astype(np.float32), labels))
        return normalized

    train_seqs = _normalize(train_seqs)
    val_seqs = _normalize(val_seqs)
    test_seqs = _normalize(test_seqs)

    # Calculate positive weight for class imbalance
    all_train_labels = np.concatenate([s[1] for s in train_seqs], axis=0)
    num_pos = int(np.sum(all_train_labels))
    num_neg = len(all_train_labels) - num_pos
    pos_weight_val = float(num_neg / max(1, num_pos))
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    print(f"    Train hours: {len(all_train_labels)}, Sepsis hours: {num_pos} ({num_pos/len(all_train_labels)*100:.2f}%)")
    print(f"    BCE pos_weight: {pos_weight_val:.2f}")

    # DataLoaders
    train_loader = DataLoader(
        SepsisSequenceDataset(train_seqs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
    )
    val_loader = DataLoader(
        SepsisSequenceDataset(val_seqs),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )
    test_loader = DataLoader(
        SepsisSequenceDataset(test_seqs),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )

    # 6. Initialize Model
    model = SepsisRNN(
        input_dim=22,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    print(f"\n[*] Training {args.rnn_type.upper()} Model ({args.epochs} epochs)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        t_epoch_start = time.time()
        for x_b, y_b, mask_b, _ in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            mask_b = mask_b.to(device)

            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits[mask_b], y_b[mask_b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * int(mask_b.sum().item())
            total_tokens += int(mask_b.sum().item())

        avg_loss = total_loss / max(1, total_tokens)
        print(f"    Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | Time: {time.time() - t_epoch_start:.1f}s")

    # 7. Threshold Selection on VALIDATION SPLIT ONLY
    print("\n[*] Evaluating on VALIDATION split for optimal decision threshold...")
    model.eval()
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for x_b, y_b, mask_b, _ in val_loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            probs = torch.sigmoid(logits).cpu()

            val_preds_list.append(probs[mask_b].numpy())
            val_targets_list.append(y_b[mask_b].numpy())

    val_probs = np.concatenate(val_preds_list)
    val_targets = np.concatenate(val_targets_list)

    best_thresh = 0.40
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.85, 81):
        preds = (val_probs >= t).astype(int)
        _, rec, f1, _ = precision_recall_fscore_support(
            val_targets, preds, average="binary", zero_division=0
        )
        if rec >= 0.70 and f1 > best_f1:
            best_f1 = f1
            best_thresh = round(float(t), 4)

    if best_f1 < 0:
        for t in np.linspace(0.05, 0.85, 81):
            preds = (val_probs >= t).astype(int)
            _, rec, f1, _ = precision_recall_fscore_support(
                val_targets, preds, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = round(float(t), 4)

    print(f"[*] Selected optimal threshold on validation: {best_thresh} (Validation F1: {best_f1:.4f})")

    # 8. Evaluation on HELD-OUT TEST SPLIT
    print("\n[*] Evaluating on HELD-OUT TEST split...")
    test_preds_list = []
    test_targets_list = []

    with torch.no_grad():
        for x_b, y_b, mask_b, _ in test_loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            probs = torch.sigmoid(logits).cpu()

            test_preds_list.append(probs[mask_b].numpy())
            test_targets_list.append(y_b[mask_b].numpy())

    test_probs = np.concatenate(test_preds_list)
    test_targets = np.concatenate(test_targets_list)
    test_preds = (test_probs >= best_thresh).astype(int)

    auprc = float(average_precision_score(test_targets, test_probs)) if test_targets.sum() > 0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(test_targets, test_preds, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(test_targets, test_preds, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    alert_rate = float(test_preds.mean() * 100)

    print("\n" + "=" * 50)
    print(f"{args.rnn_type.upper()} HELD-OUT TEST SET EVALUATION METRICS:")
    print(f"  AUPRC:       {auprc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  Alert Rate:  {alert_rate:.2f} alerts per 100 patient-hours")
    print("=" * 50 + "\n")

    # 9. Save Artifacts (Rule 1: DO NOT touch models/model.joblib!)
    os.makedirs(args.output_dir, exist_ok=True)
    pt_path = os.path.join(args.output_dir, "lstm_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "rnn_type": args.rnn_type,
        "input_dim": 22,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
        "threshold": best_thresh,
    }, pt_path)
    print(f"[OK] Saved PyTorch sequence model checkpoint to {pt_path} (Live Logistic Regression untouched)")

    thresh_path = os.path.join(args.output_dir, "lstm_threshold.json")
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": best_thresh,
            "chosen_on": "validation",
            "model_type": f"PyTorch {args.rnn_type.upper()}",
            "selection_rule": "Targeted high recall (>=0.70) while maximizing F1 on validation set",
        }, f, indent=2)
    print(f"[OK] Saved sequence model threshold to {thresh_path}")

    # 10. Update models/results/metrics.json under LSTM / GRU key
    metrics_path = os.path.join(args.output_dir, "results", "metrics.json")
    lstm_entry = {
        "name": f"LSTM / GRU (offline, stretch) - {args.rnn_type.upper()}",
        "auprc": round(auprc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "threshold": best_thresh,
        "alert_burden": f"{alert_rate:.1f} alerts per 100 patient-hours",
        "notes": f"Offline sequence model ({args.rnn_type.upper()}) trained with causal sequence padding, max window {args.max_window}h, and BCE positive weighting",
    }

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"models": [], "disclaimer": "Research prototype, not for clinical decision-making. Evaluated on PhysioNet 2019 test split."}

    updated = False
    for i, m in enumerate(data.get("models", [])):
        if "lstm" in m.get("name", "").lower() or "gru" in m.get("name", "").lower():
            data["models"][i] = lstm_entry
            updated = True
            break
    if not updated:
        data["models"].append(lstm_entry)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Updated sequence model metrics in {metrics_path}")

    print("\n[SUCCESS] Offline sequence model benchmark complete!\n")


if __name__ == "__main__":
    main()
