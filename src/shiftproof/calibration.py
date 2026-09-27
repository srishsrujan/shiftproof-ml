import numpy as np


class PlattCalibrator:
    """Calibrate probabilities using sigmoid(a*logit(p)+b), fit with gradient descent."""

    def __init__(self, learning_rate=0.03, epochs=2500, l2=0.001):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.a = 1.0
        self.b = 0.0

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))

    @staticmethod
    def _logit(p):
        eps = 1e-7
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def fit(self, raw_probability, y):
        x = self._logit(np.asarray(raw_probability, dtype=float))
        y = np.asarray(y, dtype=float)
        a, b = 1.0, 0.0

        for _ in range(self.epochs):
            p = self._sigmoid(a * x + b)
            error = p - y
            grad_a = np.mean(error * x) + self.l2 * a
            grad_b = np.mean(error)
            a -= self.learning_rate * grad_a
            b -= self.learning_rate * grad_b

        self.a, self.b = float(a), float(b)
        return self

    def transform(self, raw_probability):
        x = self._logit(np.asarray(raw_probability, dtype=float))
        return self._sigmoid(self.a * x + self.b)

    def predict_proba(self, raw_probability):
        p = self.transform(raw_probability)
        return np.column_stack([1 - p, p])


def confidence_from_probability(p_late):
    p_late = np.asarray(p_late, dtype=float)
    return np.maximum(p_late, 1 - p_late)


def choose_abstention_threshold(y_true, calibrated_probability, max_review_rate=0.30, min_review_rate=0.05):
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(calibrated_probability, dtype=float)
    confidence = confidence_from_probability(p)
    candidates = np.unique(np.round(np.linspace(0.51, 0.99, 160), 4))

    best = None
    for threshold in candidates:
        keep = confidence >= threshold
        review_rate = 1 - keep.mean()
        if keep.sum() < 10 or review_rate > max_review_rate or review_rate < min_review_rate:
            continue
        pred = (p[keep] >= 0.5).astype(int)
        yt = y_true[keep]
        tp = int(np.sum((pred == 1) & (yt == 1)))
        fp = int(np.sum((pred == 1) & (yt == 0)))
        fn = int(np.sum((pred == 0) & (yt == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        score = (f1, -review_rate)
        if best is None or score > best[0]:
            best = (score, float(threshold), float(f1), float(review_rate))

    if best is None:
        # Fallback that still enforces an explicit review policy.
        fallback = float(np.quantile(confidence, 0.90))
        return max(0.51, min(0.90, fallback))
    return best[1]
