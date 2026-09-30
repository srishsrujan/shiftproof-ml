import streamlit as st
st.title("ShiftProof Dashboard")
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from shiftproof.config import ARTIFACT_DIR, REPORT_DIR
from shiftproof.predictor import Predictor

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="ShiftProof", page_icon="🛡️", layout="wide")
st.title("🛡️ ShiftProof")
st.caption("Predict service-deadline misses, quantify confidence, and send uncertain cases to human review.")

st.success("Dashboard loaded successfully!")

try:
    predictor = Predictor()
except Exception as exc:
    st.error(f"Predictor error: {exc}")
    st.stop()

with st.sidebar:
    st.header("Model")
    st.write(f"**Version:** {predictor.bundle.version}")
    st.write(f"**Model:** {predictor.bundle.model_name}")
    st.write(f"**Review threshold:** {predictor.bundle.abstain_threshold:.3f}")

st.subheader("Single request")
cols = st.columns(4)
with cols[0]:
    request_type = st.selectbox("Request type", ["support", "delivery", "maintenance", "billing", "onboarding"])
    priority = st.selectbox("Priority", ["low", "medium", "high", "critical"])
    channel = st.selectbox("Channel", ["email", "phone", "portal", "api"])
    customer_tier = st.selectbox("Customer tier", ["standard", "premium", "enterprise"])
with cols[1]:
    region = st.selectbox("Region", ["north", "south", "east", "west"])
    agent_experience = st.number_input("Agent experience (months)", 0.0, 100.0, 12.0)
    queue_length = st.number_input("Queue length", 0.0, 1000.0, 24.0)
    estimated_work = st.number_input("Estimated work (hours)", 0.1, 24.0, 5.0)
with cols[2]:
    sla = st.slider("Historical SLA rate", 0.3, 0.99, 0.75)
    attachments = st.number_input("Attachments", 0, 15, 1)
    holiday = st.selectbox("Holiday", [0, 1])
    system_load = st.slider("System load", 0.0, 1.5, 0.8)
with cols[3]:
    complexity = st.slider("Customer complexity", 0.0, 1.0, 0.7)
    hour = st.slider("Hour of day", 0, 23, 15)
    weekday = st.slider("Weekday", 0, 6, 2)
    since_last = st.number_input("Days since last request", 0.0, 60.0, 2.0)

created_at = st.text_input("Created at (ISO timestamp)", "2026-06-20T15:10:00")

if st.button("Predict", type="primary"):
    payload = {
        "request_type": request_type,
        "priority": priority,
        "channel": channel,
        "customer_tier": customer_tier,
        "region": region,
        "agent_experience_months": agent_experience,
        "queue_length": queue_length,
        "estimated_work_hours": estimated_work,
        "historical_sla_rate": sla,
        "attachments_count": attachments,
        "is_holiday": holiday,
        "system_load": system_load,
        "customer_complexity_score": complexity,
        "hour_of_day": hour,
        "weekday": weekday,
        "days_since_last_request": since_last,
        "created_at": created_at,
    }
    result = predictor.predict_one(payload)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probability late", f"{result['probability_late']:.1%}")
    c2.metric("Confidence", f"{result['confidence']:.1%}")
    c3.metric("Prediction", result["prediction"])
    c4.metric("Decision", result["decision"])

st.divider()
st.subheader("Batch CSV")
file = st.file_uploader("Upload rows using the same feature names as the training schema", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    result = predictor.predict_frame(df)
    st.dataframe(result, use_container_width=True)
    st.download_button("Download predictions", result.to_csv(index=False), file_name="shiftproof_predictions.csv")

st.divider()
st.subheader("Evidence")
if (REPORT_DIR / "metrics.csv").exists():
    metrics = pd.read_csv(REPORT_DIR / "metrics.csv")
    st.dataframe(metrics, use_container_width=True)
else:
    st.info("Run the training command to generate evidence reports.")

c1, c2 = st.columns(2)
with c1:
    st.write("### Slice metrics")
    path = REPORT_DIR / "slice_metrics.csv"
    if path.exists():
        st.dataframe(pd.read_csv(path), use_container_width=True)
with c2:
    st.write("### Failure analysis")
    path = REPORT_DIR / "failure_analysis.csv"
    if path.exists():
        st.dataframe(pd.read_csv(path).head(20), use_container_width=True)

st.write("### Drift")
drift_path = REPORT_DIR / "drift_report.json"
if drift_path.exists():
    payload = json.loads(drift_path.read_text(encoding="utf-8"))
    st.json(payload)
