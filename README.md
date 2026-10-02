# Placement Readiness Predictor

A tabular machine-learning app that estimates a student's placement readiness from academic results, aptitude, technical and communication skills, coding practice, projects, internships, certifications, and backlogs. The dashboard supports individual profiles, live what-if scoring, cohort filters and charts, and CSV batch predictions.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/generate_and_train.py
streamlit run app.py
```

Optional API:

```powershell
uvicorn shiftproof.api:app --app-dir src --reload
```

Run tests with `python -m pytest -q`.

## Data and target

`data/generate_data.py` creates a reproducible synthetic student dataset at `data/raw/students.csv`. The binary target `placement_ready` means the generated profile is ready according to the synthetic outcome process. It is a demonstration label, not a real hiring outcome or guarantee.

The features are `cgpa`, `aptitude_score`, `technical_skills_score`, `communication_score`, `coding_hours_per_week`, `projects_completed`, `internships_completed`, `certifications_count`, `backlogs`, and `branch`. `graduation_year` is used only to create chronological train, validation, and test cohorts; it is not a model feature. `student_id` is an identifier and is never used for training.

The preprocessing pipeline imputes missing numeric and categorical values, scales numeric fields, and one-hot encodes branch. A custom NumPy logistic-regression baseline is compared with Random Forest and HistGradientBoosting on the same held-out cohort. Validation data selects and calibrates the final model; the test cohort remains held out for reporting.

## Dashboard

- Individual readiness score and category with profile strengths and focus areas.
- What-if controls for coding practice, projects, and internships.
- Cohort filters and readiness-rate chart by branch, plus model comparison metrics.
- CSV upload and downloadable predictions. `sample_input.csv` is a ready-to-use example.

Scores and profile signals support planning only. Synthetic training data cannot establish real-world placement probability; train and validate on representative, consented student data before using this for decisions.

## Outputs

Training writes the fitted model and summary under `artifacts/`, plus model comparison, branch/cohort slices, calibration, error analysis, and drift reports under `reports/`.
