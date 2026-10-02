# Placement predictor demo

## 1. Problem

“This app estimates placement readiness from a student's academic record, skills, coding practice, projects, and internships. The output is a planning signal, not a hiring decision or guarantee.”

## 2. Data and leakage control

Show `data/raw/students.csv`. Explain that it is synthetic, includes realistic missing values, and uses `placement_ready` as its explicit binary target. `graduation_year` defines the chronological 70/15/15 split and is excluded from model features.

## 3. Models and evaluation

Show `reports/metrics.csv`. Explain the custom NumPy logistic-regression baseline and the Random Forest and HistGradientBoosting comparisons on the same held-out cohort.

## 4. Individual and what-if prediction

Enter a profile in Streamlit. Review its readiness score, category, profile strengths and focus areas. Change coding practice, project, and internship values in the what-if controls and compare the updated score.

## 5. Cohorts and batch predictions

Filter cohort analytics by branch and graduation year, then upload `sample_input.csv` in the batch tab and download its predictions.

## 6. API and caveat

Open `/docs` in FastAPI and call `/predict` with a student profile. Close by noting that real use requires representative, consented student data and careful validation for fairness and calibration.
