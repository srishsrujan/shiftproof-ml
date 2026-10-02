import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "src"))

from shiftproof.config import FEATURES, REPORT_DIR, TARGET
from shiftproof.data_pipeline import time_split
from shiftproof.predictor import Predictor


def run():
    data_path = ROOT / "data" / "raw" / "students.csv"
    df = pd.read_csv(data_path)
    splits = time_split(df)
    predictor = Predictor()
    out_dir = REPORT_DIR / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. Added missing values.
    missing = splits.test.copy()
    rng = np.random.default_rng(7)
    for col in FEATURES:
        if rng.random() < 0.35:
            mask = rng.random(len(missing)) < 0.25
            missing.loc[mask, col] = np.nan
    pred = predictor.predict_frame(missing)
    results.append({"test": "heavy_missingness", "rows": len(pred), "failed": int(pred.isna().any(axis=1).sum())})

    # 2. Shift class balance by oversampling positive records.
    pos = splits.test[splits.test[TARGET] == 1]
    neg = splits.test[splits.test[TARGET] == 0]
    shifted = pd.concat([neg, pos.sample(n=min(len(neg), max(len(pos) * 3, 1)), replace=True, random_state=7)], ignore_index=True)
    pred = predictor.predict_frame(shifted)
    results.append({"test": "shifted_class_balance", "rows": len(pred), "failed": int(pred.isna().any(axis=1).sum())})

    # 3. Drop optional input columns: the cleaning layer inserts them as missing.
    dropped = splits.test.drop(columns=["certifications_count", "backlogs", "aptitude_score"])
    pred = predictor.predict_frame(dropped)
    results.append({"test": "dropped_columns", "rows": len(pred), "failed": int(pred.isna().any(axis=1).sum())})

    pd.DataFrame(results).to_csv(out_dir / "robustness_summary.csv", index=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
