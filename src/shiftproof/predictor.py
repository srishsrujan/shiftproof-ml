from pathlib import Path
import joblib
import pandas as pd

from .config import ARTIFACT_DIR, ID_COL
from .data_pipeline import clean_input


class Predictor:
    def __init__(self, artifact_path: Path | None = None):
        artifact_path = artifact_path or ARTIFACT_DIR / "shiftproof_model.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {artifact_path}. Run scripts/generate_and_train.py first.")
        self.bundle = joblib.load(artifact_path)

    def predict_one(self, payload: dict):
        frame = clean_input(pd.DataFrame([payload]))
        result = self.bundle.predict(frame)
        return result.iloc[0].to_dict()

    def predict_frame(self, frame: pd.DataFrame):
        data = clean_input(frame)
        result = self.bundle.predict(data)
        if ID_COL in frame:
            result.insert(0, ID_COL, frame[ID_COL].to_numpy())
        return result
