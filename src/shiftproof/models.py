from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


@dataclass
class TrainingHistory:
    losses: list


class NumpyLogisticRegression:
    """A compact logistic regression implementation with explicit gradient descent."""

    def __init__(self, learning_rate=0.05, epochs=900, l2=0.01, class_weight=None, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.class_weight = class_weight
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0
        self.history = TrainingHistory(losses=[])

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def _sample_weights(self, y):
        if not self.class_weight:
            return np.ones_like(y, dtype=float)
        w = np.ones_like(y, dtype=float)
        for cls, value in self.class_weight.items():
            w[y == int(cls)] = float(value)
        return w

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.weights = np.zeros(X.shape[1], dtype=float)
        self.bias = 0.0
        sw = self._sample_weights(y)
        sw = sw / sw.mean()

        for epoch in range(self.epochs):
            p = self._sigmoid(X @ self.weights + self.bias)
            error = p - y
            grad_w = (X.T @ (sw * error)) / len(y) + self.l2 * self.weights
            grad_b = float(np.mean(sw * error))

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            if epoch % 25 == 0 or epoch == self.epochs - 1:
                eps = 1e-9
                loss = -np.average(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps), weights=sw)
                loss += 0.5 * self.l2 * float(np.dot(self.weights, self.weights))
                self.history.losses.append(float(loss))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = self._sigmoid(X @ self.weights + self.bias)
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)[:, 1]
        return (p >= threshold).astype(int)


def build_stronger_models(random_state=42):
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=12,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.055,
            max_leaf_nodes=31,
            l2_regularization=0.2,
            random_state=random_state,
        ),
    }
