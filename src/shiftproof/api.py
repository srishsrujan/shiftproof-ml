from typing import Any
import json
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict

from .config import ARTIFACT_DIR, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .drift import drift_report
from .predictor import Predictor

app = FastAPI(title="Placement Readiness API", version="1.0.0")
predictor = None


class RequestPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    graduation_year: Any = 2026
    branch: str | None = None
    cgpa: Any = None
    aptitude_score: Any = None
    technical_skills_score: Any = None
    communication_score: Any = None
    coding_hours_per_week: Any = None
    projects_completed: Any = None
    internships_completed: Any = None
    certifications_count: Any = None
    backlogs: Any = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = Predictor()
    return predictor


@app.get("/health")
def health():
    try:
        p = get_predictor()
        return {"status": "ok", "model_version": p.bundle.version, "model_name": p.bundle.model_name}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.post("/predict")
def predict(payload: RequestPayload):
    try:
        result = get_predictor().predict_one(payload.model_dump())
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        from io import BytesIO
        frame = pd.read_csv(BytesIO(content))
        result = get_predictor().predict_frame(frame)
        return {"rows": json.loads(result.to_json(orient="records"))}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/drift")
async def drift(file: UploadFile = File(...)):
    try:
        content = await file.read()
        from io import BytesIO
        new_df = pd.read_csv(BytesIO(content))
        train_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "students.csv"
        train_df = pd.read_csv(train_path).sort_values("graduation_year").head(int(len(pd.read_csv(train_path)) * 0.7))
        report = drift_report(train_df, new_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        return {"features": json.loads(report.to_json(orient="records"))}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
