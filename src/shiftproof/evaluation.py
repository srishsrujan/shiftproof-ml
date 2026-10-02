from pathlib import Path
import numpy as np
import pandas as pd
from .config import TARGET
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)


def expected_calibration_error(y_true, probability, bins=10):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probability >= lo) & ((probability < hi) if hi < 1 else (probability <= hi))
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(probability[mask].mean())
        weight = count / len(y_true)
        ece += weight * abs(acc - conf)
        rows.append({"bin_low": lo, "bin_high": hi, "count": count, "avg_confidence": conf, "empirical_rate": acc})
    return float(ece), pd.DataFrame(rows)


def binary_metrics(y_true, probability, threshold=0.5):
    pred = (np.asarray(probability) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
    }


def evaluate_with_abstention(y_true, probability, abstain_threshold):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(probability).astype(float)
    confidence = np.maximum(p, 1 - p)
    auto = confidence >= abstain_threshold
    decisions = np.full(len(p), -1, dtype=int)
    decisions[auto] = (p[auto] >= 0.5).astype(int)

    metrics = binary_metrics(y_true[auto], p[auto], 0.5) if auto.any() else {k: 0.0 for k in ["precision", "recall", "f1", "pr_auc", "brier"]}
    metrics["review_rate"] = float((~auto).mean())
    metrics["auto_decision_rate"] = float(auto.mean())
    metrics["abstain_threshold"] = float(abstain_threshold)
    return metrics, decisions, confidence


def slice_metrics(frame, probability, columns, threshold=0.5):
    output = []
    for col in columns:
        for value, part in frame.groupby(col, dropna=False):
            idx = part.index.to_numpy()
            p = probability[idx]
            y = part[TARGET].to_numpy().astype(int)
            m = binary_metrics(y, p, threshold)
            m.update({"slice": col, "value": str(value), "n": len(part), "placement_ready_rate": float(y.mean())})
            output.append(m)
    return pd.DataFrame(output)


def analyse_failures(frame, probability, threshold=0.5, top_n=50):
    data = frame.copy().reset_index(drop=True)
    data["predicted_probability"] = probability
    data["predicted"] = (probability >= threshold).astype(int)
    data["confidence"] = np.maximum(probability, 1 - probability)
    wrong = data[data["predicted"] != data[TARGET]].copy()
    wrong["error_type"] = np.where(wrong["predicted"] == 1, "false_positive", "false_negative")
    return wrong.sort_values("confidence").head(top_n)


def save_metrics(path: Path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)


def confidence_calibration(y_true, probability, bins=10):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    confidence = np.maximum(probability, 1 - probability)
    correct = ((probability >= 0.5).astype(int) == y_true).astype(int)
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(confidence[mask].mean())
        empirical = float(correct[mask].mean())
        ece += (count / len(y_true)) * abs(empirical - conf)
        rows.append({"bin_low": lo, "bin_high": hi, "count": count, "avg_confidence": conf, "empirical_accuracy": empirical})
    return float(ece), pd.DataFrame(rows)
