from fastapi.testclient import TestClient

from shiftproof.api import app


client = TestClient(app)


def test_health_endpoint_after_training_artifact_exists():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body


def test_predict_accepts_student_profile_inputs():
    response = client.post("/predict", json={
        "graduation_year": 2026,
        "branch": "computer_science",
        "cgpa": "8.1",
        "aptitude_score": 72,
        "technical_skills_score": 78,
        "communication_score": 69,
        "coding_hours_per_week": 12,
        "projects_completed": 3,
        "internships_completed": 1,
        "certifications_count": 2,
        "backlogs": 0,
    })
    assert response.status_code == 200
    assert "readiness_score" in response.json()
    assert response.json()["readiness_category"] in {"Ready", "Developing", "Needs support"}


def test_predict_csv_preserves_student_ids():
    csv_data = (
        "student_id,graduation_year,branch,cgpa,aptitude_score,technical_skills_score,"
        "communication_score,coding_hours_per_week,projects_completed,internships_completed,"
        "certifications_count,backlogs\n"
        "STU-001,2026,computer_science,8.1,72,78,69,12,3,1,2,0\n"
    )
    response = client.post(
        "/predict_csv",
        files={"file": ("students.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["rows"][0]["student_id"] == "STU-001"
