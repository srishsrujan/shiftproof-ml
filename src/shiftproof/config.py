from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
TARGET = "late"
TIME_COL = "created_at"
ID_COL = "request_id"

NUMERIC_FEATURES = [
    "agent_experience_months",
    "queue_length",
    "estimated_work_hours",
    "historical_sla_rate",
    "attachments_count",
    "is_holiday",
    "system_load",
    "customer_complexity_score",
    "hour_of_day",
    "weekday",
    "days_since_last_request",
]

CATEGORICAL_FEATURES = [
    "request_type",
    "priority",
    "channel",
    "customer_tier",
    "region",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
