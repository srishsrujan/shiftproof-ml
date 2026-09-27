import json
from pathlib import Path
import numpy as np
import pandas as pd


def psi(expected, actual, bins=10):
    expected = pd.Series(expected).dropna().to_numpy(dtype=float)
    actual = pd.Series(actual).dropna().to_numpy(dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    expected_hist, _ = np.histogram(expected, bins=edges)
    actual_hist, _ = np.histogram(actual, bins=edges)
    expected_pct = np.clip(expected_hist / max(expected_hist.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_hist / max(actual_hist.sum(), 1), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def categorical_shift(expected, actual):
    e = pd.Series(expected).astype("string").fillna("<missing>").value_counts(normalize=True)
    a = pd.Series(actual).astype("string").fillna("<missing>").value_counts(normalize=True)
    cats = sorted(set(e.index) | set(a.index))
    value = 0.0
    for cat in cats:
        pe = max(float(e.get(cat, 0.0)), 1e-6)
        pa = max(float(a.get(cat, 0.0)), 1e-6)
        value += (pa - pe) * np.log(pa / pe)
    return float(value)


def drift_report(train_df, new_df, numeric_cols, categorical_cols):
    rows = []
    for col in numeric_cols:
        score = psi(train_df[col], new_df[col])
        rows.append({"feature": col, "kind": "numeric", "score": score, "status": "review" if score > 0.20 else "monitor" if score > 0.10 else "stable"})
    for col in categorical_cols:
        score = categorical_shift(train_df[col], new_df[col])
        rows.append({"feature": col, "kind": "categorical", "score": score, "status": "review" if score > 0.20 else "monitor" if score > 0.10 else "stable"})
    result = pd.DataFrame(rows).sort_values("score", ascending=False)
    return result


def save_drift_json(report: pd.DataFrame, path: Path):
    payload = {
        "summary": {
            "review_features": int((report["status"] == "review").sum()),
            "monitor_features": int((report["status"] == "monitor").sum()),
            "stable_features": int((report["status"] == "stable").sum()),
        },
        "features": report.to_dict(orient="records"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
