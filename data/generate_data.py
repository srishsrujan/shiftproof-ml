from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def generate_students(n=12000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    graduation_year = rng.choice([2024, 2025, 2026, 2027], size=n, p=[0.18, 0.24, 0.30, 0.28])
    branch = rng.choice(
        ["computer_science", "information_technology", "electronics", "mechanical", "civil"],
        size=n,
        p=[0.34, 0.22, 0.20, 0.14, 0.10],
    )
    cgpa = np.clip(rng.normal(7.1, 1.15, n), 4.0, 10.0)
    aptitude = np.clip(rng.normal(58 + (cgpa - 7) * 2, 15, n), 0, 100)
    technical = np.clip(rng.normal(55 + (cgpa - 7) * 4, 17, n), 0, 100)
    communication = np.clip(rng.normal(62, 14, n), 0, 100)
    coding_hours = np.clip(rng.gamma(2.3, 5.0, n), 0, 40)
    projects = np.clip(rng.poisson(2.1, n), 0, 10)
    internships = np.clip(rng.poisson(0.65, n), 0, 5)
    certifications = np.clip(rng.poisson(2.0, n), 0, 12)
    backlogs = np.clip(rng.poisson(0.55, n), 0, 8)

    branch_effect = {
        "computer_science": 0.20,
        "information_technology": 0.16,
        "electronics": 0.04,
        "mechanical": -0.08,
        "civil": -0.12,
    }
    logit = (
        -2.65
        + 0.48 * (cgpa - 6.5)
        + 0.018 * (aptitude - 55)
        + 0.023 * (technical - 55)
        + 0.013 * (communication - 55)
        + 0.055 * (coding_hours - 8)
        + 0.22 * projects
        + 0.42 * internships
        + 0.07 * certifications
        - 0.32 * backlogs
        + np.vectorize(branch_effect.get)(branch)
        + rng.normal(0, 0.85, n)
    )
    placement_ready = rng.binomial(1, sigmoid(logit))

    df = pd.DataFrame(
        {
            "student_id": [f"STU-{i:06d}" for i in range(1, n + 1)],
            "graduation_year": graduation_year,
            "branch": branch,
            "cgpa": cgpa.round(2),
            "aptitude_score": aptitude.round(1),
            "technical_skills_score": technical.round(1),
            "communication_score": communication.round(1),
            "coding_hours_per_week": coding_hours.round(1),
            "projects_completed": projects,
            "internships_completed": internships,
            "certifications_count": certifications,
            "backlogs": backlogs,
            "placement_ready": placement_ready,
        }
    )

    missing_rates = {
        "cgpa": 0.025,
        "aptitude_score": 0.04,
        "technical_skills_score": 0.04,
        "communication_score": 0.03,
        "coding_hours_per_week": 0.035,
        "projects_completed": 0.025,
        "internships_completed": 0.025,
        "certifications_count": 0.03,
        "branch": 0.02,
    }
    for column, rate in missing_rates.items():
        df.loc[rng.random(n) < rate, column] = np.nan

    return df


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "raw" / "students.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_students()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")
    print(df["placement_ready"].value_counts(normalize=True).rename("share"))


if __name__ == "__main__":
    main()
