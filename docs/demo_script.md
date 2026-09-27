# 5–8 minute demo script

## 1. Problem (45–60 seconds)

“ShiftProof predicts whether an operational request is likely to miss its service deadline. The key requirement is not just a class label: the system should know when it may be wrong and route uncertain cases to review.”

## 2. Data + leakage control (60 seconds)

Show `data/raw/requests.csv`. Explain that the dataset intentionally contains missing values, categorical fields, a changing class rate and time drift. Show the 70/15/15 chronological split.

## 3. Models (60–90 seconds)

Show `reports/metrics.csv`. Explain the custom NumPy logistic baseline and the two library models. Point out that every model is evaluated on the same frozen test set.

## 4. Calibration + abstention (60–90 seconds)

Show `reports/confidence_calibration.csv`. Focus on the 0.80–0.85 confidence bucket and its empirical accuracy. Then enter an uncertain-looking case in Streamlit and show the `uncertain - send for review` decision.

## 5. Robustness + drift (60–90 seconds)

Show `reports/robustness/robustness_summary.csv`: heavy missingness, shifted class balance and dropped columns all complete without prediction failures. Show `reports/drift_report.json` and explain the PSI review flags.

## 6. API + closing (45–60 seconds)

Open `/docs` in FastAPI, call `/predict`, and point out probability, confidence, decision and model version. Close by explaining that the next production step would be retraining on a real, governed operational dataset.
