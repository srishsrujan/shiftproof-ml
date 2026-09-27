from fastapi.testclient import TestClient

from src.shiftproof.api import app


client = TestClient(app)


def test_health_endpoint_after_training_artifact_exists():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body


def test_predict_accepts_badly_formatted_numeric_and_category_values():
    response = client.post("/predict", json={
        "request_type": "SUPPORT",
        "priority": "URGENT",
        "channel": "PORTAL",
        "customer_tier": "STANDARD",
        "region": "east",
        "agent_experience_months": "12",
        "queue_length": "31",
        "estimated_work_hours": "4.5",
        "historical_sla_rate": 0.7,
        "attachments_count": 1,
        "is_holiday": 0,
        "system_load": 0.9,
        "customer_complexity_score": 0.8,
        "hour_of_day": 16,
        "weekday": 2,
        "days_since_last_request": 2,
        "created_at": "2026-08-01T16:00:00"
    })
    assert response.status_code == 200
    assert "decision" in response.json()
