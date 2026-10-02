from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from .calibration import PlattCalibrator, choose_abstention_threshold
from .config import ARTIFACT_DIR, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, REPORT_DIR, TARGET, TIME_COL
from .data_pipeline import build_preprocessor, prepare_xy, time_split
from .drift import drift_report, save_drift_json
from .evaluation import analyse_failures, binary_metrics, confidence_calibration, evaluate_with_abstention, expected_calibration_error, slice_metrics
from .models import NumpyLogisticRegression, build_stronger_models


class FinalModelBundle:
    def __init__(self, preprocessor, model, calibrator, abstain_threshold, model_name, version="placement-v1"):
        self.preprocessor = preprocessor
        self.model = model
        self.calibrator = calibrator
        self.abstain_threshold = float(abstain_threshold)
        self.model_name = model_name
        self.version = version

    def predict(self, frame):
        x = frame[FEATURES].copy()
        xt = self.preprocessor.transform(x)
        raw_p = self.model.predict_proba(xt)[:, 1]
        p = self.calibrator.transform(raw_p)
        confidence = np.maximum(p, 1 - p)
        prediction = np.where(p >= 0.5, "placement_ready", "not_ready")
        category = np.select([p >= 0.70, p >= 0.45], ["Ready", "Developing"], default="Needs support")
        return pd.DataFrame(
            {
            "readiness_score": p,
                "confidence": confidence,
                "prediction": prediction,
            "readiness_category": category,
                "model_version": self.version,
                "model_name": self.model_name,
            }
        )


def _class_weights(y):
    pos = max(int(y.sum()), 1)
    neg = max(int((1 - y).sum()), 1)
    return {0: 1.0, 1: neg / pos}


def train_all(csv_path: Path):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    splits = time_split(df)

    x_train, y_train = prepare_xy(splits.train)
    x_val, y_val = prepare_xy(splits.validation)
    x_test, y_test = prepare_xy(splits.test)

    pre = build_preprocessor()
    X_train = pre.fit_transform(x_train)
    X_val = pre.transform(x_val)
    X_test = pre.transform(x_test)

    metrics_rows = []
    fitted_models = {}

    baseline = NumpyLogisticRegression(
        learning_rate=0.06,
        epochs=1000,
        l2=0.02,
        class_weight=_class_weights(y_train),
    )
    baseline.fit(X_train, y_train)
    for name, model, matrix in [("baseline_logistic", baseline, X_test)]:
        p = model.predict_proba(matrix)[:, 1]
        m = binary_metrics(y_test, p)
        ece, _ = expected_calibration_error(y_test, p)
        m.update({"model": name, "ece": ece})
        metrics_rows.append(m)
        fitted_models[name] = model

    for name, model in build_stronger_models().items():
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        m = binary_metrics(y_test, p)
        ece, _ = expected_calibration_error(y_test, p)
        m.update({"model": name, "ece": ece})
        metrics_rows.append(m)
        fitted_models[name] = model

    metrics = pd.DataFrame(metrics_rows).sort_values("pr_auc", ascending=False)
    metrics.to_csv(REPORT_DIR / "metrics.csv", index=False)

    # Use the validation set to pick the final model, then freeze it for test evaluation.
    val_scores = {}
    for name, model in fitted_models.items():
        val_scores[name] = model.predict_proba(X_val)[:, 1]
    best_name = max(val_scores, key=lambda name: binary_metrics(y_val, val_scores[name])["pr_auc"])
    best_model = fitted_models[best_name]

    raw_val_p = val_scores[best_name]
    calibrator = PlattCalibrator().fit(raw_val_p, y_val)
    calibrated_val_p = calibrator.transform(raw_val_p)
    abstain_threshold = choose_abstention_threshold(y_val, calibrated_val_p, max_review_rate=0.30)

    raw_test_p = best_model.predict_proba(X_test)[:, 1]
    calibrated_test_p = calibrator.transform(raw_test_p)
    final_metrics, _, confidence = evaluate_with_abstention(y_test, calibrated_test_p, abstain_threshold)
    final_metrics.update({"model": f"final:{best_name}", "calibrated_ece": expected_calibration_error(y_test, calibrated_test_p)[0]})

    _, calibration_bins = expected_calibration_error(y_test, calibrated_test_p)
    calibration_bins.to_csv(REPORT_DIR / "calibration.csv", index=False)
    confidence_ece, confidence_bins = confidence_calibration(y_test, calibrated_test_p)
    confidence_bins.to_csv(REPORT_DIR / "confidence_calibration.csv", index=False)
    final_metrics["confidence_ece"] = confidence_ece
    pd.concat([metrics, pd.DataFrame([final_metrics])], ignore_index=True).to_csv(REPORT_DIR / "metrics.csv", index=False)

    test_frame = splits.test.reset_index(drop=True).copy()
    test_frame["cohort"] = test_frame[TIME_COL].astype(int).astype(str)
    slice_df = slice_metrics(test_frame, calibrated_test_p, ["branch", "cohort"])
    slice_df.to_csv(REPORT_DIR / "slice_metrics.csv", index=False)

    failures = analyse_failures(test_frame, calibrated_test_p)
    if len(failures) < 20:
        # Keep the file explicit even when the random synthetic test is unusually clean.
        failures = analyse_failures(test_frame, calibrated_test_p, top_n=min(50, len(test_frame)))
    failures.to_csv(REPORT_DIR / "failure_analysis.csv", index=False)

    drift = drift_report(splits.train, splits.test, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    save_drift_json(drift, REPORT_DIR / "drift_report.json")

    bundle = FinalModelBundle(pre, best_model, calibrator, abstain_threshold, best_name)
    joblib.dump(bundle, ARTIFACT_DIR / "shiftproof_model.joblib")

    summary = {
        "model_name": best_name,
        "abstention_threshold": abstain_threshold,
        "train_rows": len(splits.train),
        "validation_rows": len(splits.validation),
        "test_rows": len(splits.test),
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
        "calibrator_a": calibrator.a,
        "calibrator_b": calibrator.b,
        "test_calibrated_metrics": final_metrics,
        "features": FEATURES,
    }
    (ARTIFACT_DIR / "model_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    return summary
