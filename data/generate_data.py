from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def generate_requests(n=12000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2026-08-31", periods=n)
    day_index = np.arange(n) / max(n - 1, 1)

    request_type = rng.choice(
        ["support", "delivery", "maintenance", "billing", "onboarding"],
        size=n,
        p=[0.34, 0.23, 0.16, 0.17, 0.10],
    )
    priority = rng.choice(["low", "medium", "high", "critical"], size=n, p=[0.36, 0.40, 0.19, 0.05])
    channel = rng.choice(["email", "phone", "portal", "api"], size=n, p=[0.34, 0.23, 0.31, 0.12])
    customer_tier = rng.choice(["standard", "premium", "enterprise"], size=n, p=[0.61, 0.27, 0.12])
    region = rng.choice(["north", "south", "east", "west"], size=n)

    hour = rng.integers(0, 24, size=n)
    weekday = rng.integers(0, 7, size=n)
    holiday = ((weekday >= 5) | ((hour == 0) & (rng.random(n) < 0.25))).astype(int)

    # Time drift: later data has higher load and queue pressure.
    system_load = np.clip(rng.normal(0.55 + 0.22 * day_index, 0.16, n), 0.03, 1.25)
    queue_length = np.maximum(0, rng.poisson(9 + 9 * day_index + 4 * system_load))
    agent_experience = np.clip(rng.gamma(shape=2.6, scale=9.0, size=n), 1, 72)
    estimated_work = np.clip(rng.lognormal(mean=0.5, sigma=0.65, size=n), 0.25, 16)
    historical_sla = np.clip(rng.normal(0.86 - 0.12 * day_index, 0.10, n), 0.30, 0.99)
    attachments = np.clip(rng.poisson(1.1, size=n), 0, 8)
    complexity = np.clip(rng.beta(2.2, 3.4, size=n), 0, 1)
    days_since_last = np.clip(rng.exponential(5, size=n), 0, 40)

    priority_effect = {"low": -0.55, "medium": 0.0, "high": 0.55, "critical": 0.75}
    type_effect = {"support": 0.12, "delivery": 0.25, "maintenance": 0.35, "billing": -0.08, "onboarding": 0.03}
    channel_effect = {"email": 0.15, "phone": -0.05, "portal": 0.03, "api": -0.10}
    tier_effect = {"standard": 0.14, "premium": -0.02, "enterprise": -0.12}

    logit = (
        -5.15
        + 0.085 * queue_length
        + 0.29 * estimated_work
        + 1.95 * (1 - historical_sla)
        + 1.18 * complexity
        + 0.95 * system_load
        - 0.022 * agent_experience
        + 0.07 * attachments
        + 0.34 * holiday
        + np.vectorize(priority_effect.get)(priority)
        + np.vectorize(type_effect.get)(request_type)
        + np.vectorize(channel_effect.get)(channel)
        + np.vectorize(tier_effect.get)(customer_tier)
        + 0.18 * (day_index > 0.72)
        + 0.15 * (hour >= 18)
        + rng.normal(0, 0.45, n)
    )

    p_late = sigmoid(logit)
    late = rng.binomial(1, p_late)

    df = pd.DataFrame(
        {
            "request_id": [f"REQ-{i:06d}" for i in range(1, n + 1)],
            "created_at": dates,
            "request_type": request_type,
            "priority": priority,
            "channel": channel,
            "customer_tier": customer_tier,
            "region": region,
            "agent_experience_months": agent_experience.round(2),
            "queue_length": queue_length.astype(float),
            "estimated_work_hours": estimated_work.round(2),
            "historical_sla_rate": historical_sla.round(3),
            "attachments_count": attachments.astype(float),
            "is_holiday": holiday.astype(float),
            "system_load": system_load.round(3),
            "customer_complexity_score": complexity.round(3),
            "hour_of_day": hour.astype(float),
            "weekday": weekday.astype(float),
            "days_since_last_request": days_since_last.round(2),
            "late": late,
        }
    )

    # Missing-value noise: mostly 3-7%, with stronger noise in a few fields.
    missing_rates = {
        "agent_experience_months": 0.045,
        "queue_length": 0.025,
        "estimated_work_hours": 0.035,
        "historical_sla_rate": 0.065,
        "customer_complexity_score": 0.05,
        "request_type": 0.03,
        "channel": 0.025,
        "customer_tier": 0.025,
    }
    for col, rate in missing_rates.items():
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    # Formatting problems in a small fraction of rows.
    df["queue_length"] = df["queue_length"].astype(object)
    fmt_mask = rng.random(n) < 0.012
    df.loc[fmt_mask, "queue_length"] = df.loc[fmt_mask, "queue_length"].astype(float).round(0).astype(str)
    fmt_mask2 = rng.random(n) < 0.008
    df.loc[fmt_mask2, "priority"] = df.loc[fmt_mask2, "priority"].astype(str).str.upper()

    return df


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "raw" / "requests.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_requests()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")
    print(df["late"].value_counts(normalize=True).rename("share"))


if __name__ == "__main__":
    main()
