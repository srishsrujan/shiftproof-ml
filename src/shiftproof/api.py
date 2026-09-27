from typing import Any
import json
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict

from .config import ARTIFACT_DIR, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .drift import drift_report
from .predictor import Predictor

app = FastAPI(title="ShiftProof API", version="1.0.0")
predictor = None


class RequestPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    request_type: str | None = None
    priority: str | None = None
    channel: str | None = None
    customer_tier: str | None = None
    region: str | None = None
    agent_experience_months: Any = None
    queue_length: Any = None
    estimated_work_hours: Any = None
    historical_sla_rate: Any = None
    attachments_count: Any = None
    is_holiday: Any = None
    system_load: Any = None
    customer_complexity_score: Any = None
    hour_of_day: Any = None
    weekday: Any = None
    days_since_last_request: Any = None
    created_at: str


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
        train_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "requests.csv"
        train_df = pd.read_csv(train_path).sort_values("created_at").head(int(len(pd.read_csv(train_path)) * 0.7))
        report = drift_report(train_df, new_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        return {"features": json.loads(report.to_json(orient="records"))}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
