# ShiftProof — SLA Miss Prediction with Calibrated Uncertainty

A complete machine-learning product for predicting whether an operational request will miss its service deadline. This implementation follows **Project 01 / MACHINE LEARNING — SHIFT­PROOF** from the AI Club NIT Rourkela AI Track Induction 2026 handbook, while intentionally ignoring the handbook's 30-day sequencing and building the full system directly.

Source brief: the project requires a repeatable preprocessing pipeline, time-based split, a self-built baseline, stronger library models, missing-value and imbalance handling, calibrated confidence, abstention for uncertain cases, an API, dashboard, drift checks, slice-wise evaluation, robustness tests, and failure analysis. fileciteturn0file0L11-L35

## What is included

- Synthetic but realistic noisy operational dataset generator with missing values, categorical fields, class imbalance and time drift.
- Time-based train/validation/test split.
- Custom NumPy logistic-regression baseline with an explicit gradient-descent training loop.
- Random Forest and HistGradientBoosting stronger models.
- Class-imbalance handling with class weights where supported and threshold tuning.
- Custom Platt-style probability calibration.
- Custom abstention threshold selection: low-confidence cases become `uncertain - send for review`.
- Slice evaluation by request type, priority, customer tier, region and time period.
- Robustness tests for dropped columns, missing values and changed class balance.
- Population Stability Index (PSI) drift detector comparing training data with a new batch.
- FastAPI endpoints for single and CSV predictions plus drift checks.
- Streamlit dashboard with single-request prediction, batch upload, model metrics, failure analysis and drift view.
- Unit and API tests.
- Generated demo artifacts, including metrics, calibration data, failure examples and drift report.
- `AI_USAGE.md` describing how AI assistance was used and manually verified.

## Project architecture

```text
                  ┌──────────────────────┐
                  │ Synthetic/CSV source │
                  └──────────┬───────────┘
                             │
                             v
                 ┌────────────────────────┐
                 │ Validation + Cleaning  │
                 │ coercion / categories  │
                 └──────────┬─────────────┘
                            │
                            v
          ┌─────────────────┴──────────────────┐
          │                                    │
          v                                    v
  Time split by date                     Drift monitor
 train / validation / test              PSI + category shift
          │
          v
   ┌───────────────────────┐
   │ Shared preprocessing  │
   │ median + mode + OHE   │
   └──────────┬────────────┘
              │
       ┌──────┴────────┐
       v               v
  Custom baseline   Library models
  NumPy Logistic    RandomForest / HGB
       │               │
       └──────┬────────┘
              v
      Frozen test set
              │
              v
      Calibration layer
        Platt scaling
              │
              v
      Abstention policy
     confident / review
              │
       ┌──────┴──────┐
       v             v
     FastAPI      Streamlit
       │             │
       └──────┬──────┘
              v
       Evidence + audit
```

## Quickstart

### 1. Create an environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows CMD
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the data and train everything

```bash
python scripts/generate_and_train.py
```

This creates:

- `data/raw/requests.csv`
- `artifacts/shiftproof_model.joblib`
- `artifacts/model_summary.json`
- `reports/metrics.csv`
- `reports/slice_metrics.csv`
- `reports/calibration.csv`
- `reports/confidence_calibration.csv`
- `reports/failure_analysis.csv`
- `reports/drift_report.json`
- robustness outputs under `reports/robustness/`

### 4. Run the API

```bash
uvicorn src.shiftproof.api:app --reload
```

Open: `http://127.0.0.1:8000/docs`

### 5. Run the dashboard

In a second terminal:

```bash
streamlit run src/shiftproof/dashboard.py
```

## API examples

### Single prediction

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"request_type\":\"support\",\"priority\":\"high\",\"channel\":\"portal\",\"customer_tier\":\"premium\",\"region\":\"east\",\"agent_experience_months\":12,\"queue_length\":28,\"estimated_work_hours\":5.5,\"historical_sla_rate\":0.71,\"attachments_count\":2,\"is_holiday\":0,\"system_load\":0.82,\"customer_complexity_score\":0.72,\"hour_of_day\":15,\"weekday\":2,\"days_since_last_request\":1,\"created_at\":\"2026-06-20T15:10:00\"}"
```

The response includes:

- prediction (`late` or `on_time`)
- probability of a miss
- confidence
- review decision
- model version
- threshold used

## Design decisions

### Target
`late = 1` means the simulated request exceeds its service deadline. The generator creates a realistic positive class by combining workload, queue length, complexity, historical SLA performance and time-related effects.

### Time split
Rows are sorted by `created_at`. The oldest 70% form train, the next 15% validation, and the newest 15% test. This prevents future information from entering training.

### Baseline
`NumpyLogisticRegression` is implemented from scratch. It uses batch gradient descent, L2 regularisation and weighted binary cross-entropy. This satisfies the project's requirement that the baseline algorithm/training loop is our own work.

### Stronger models
- Random Forest
- HistGradientBoosting

All models consume the same preprocessed data and the exact same frozen test split.

### Calibration
The final model's raw probabilities are transformed by a custom Platt-style calibration layer:

`calibrated_p = sigmoid(a * logit(raw_p) + b)`

The calibration layer is fit only on validation predictions. Test data is untouched during calibration fitting.

### Abstention
The calibrated probability is converted to confidence with:

`confidence = max(p_late, 1 - p_late)`

A validation-set search chooses a confidence threshold that maximises F1 on non-abstained cases while keeping the review rate between 5% and 30%. Cases below that threshold become `uncertain - send for review`. A separate confidence-calibration table measures whether an 80% confidence bucket is correct about 80% of the time.

### Drift
The project uses Population Stability Index for numeric and categorical features. A score around 0.1–0.2 is treated as moderate movement and >0.2 as a practical review signal. The thresholds are operational heuristics, not universal laws.

## Evidence generated

The handbook asks for precision, recall, F1, PR-AUC and calibration rather than accuracy alone; separate slices and time periods; robustness tests; at least 20 analysed mistakes; and identical test data for baseline/final comparisons. fileciteturn0file0L46-L56

The training script generates those evidence artifacts automatically. The handbook also calls for a working repository, README, architecture/design decisions, automated tests, failure log, demo and `AI_USAGE.md`. fileciteturn0file0L57-L65

## Important note

The dataset is synthetic because the handbook does not provide a specific operational dataset. The system is therefore a fully runnable reference implementation rather than a claim about a real company's SLA process. Replace `data/raw/requests.csv` with a real, legally usable dataset while preserving the schema and retraining pipeline.

## Main commands

```bash
python scripts/generate_and_train.py
pytest -q
uvicorn src.shiftproof.api:app --reload
streamlit run src/shiftproof/dashboard.py
```
