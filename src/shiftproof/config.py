from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
TARGET = "placement_ready"
TIME_COL = "graduation_year"
ID_COL = "student_id"

NUMERIC_FEATURES = [
    "cgpa",
    "aptitude_score",
    "technical_skills_score",
    "communication_score",
    "coding_hours_per_week",
    "projects_completed",
    "internships_completed",
    "certifications_count",
    "backlogs",
]

CATEGORICAL_FEATURES = [
    "branch",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
