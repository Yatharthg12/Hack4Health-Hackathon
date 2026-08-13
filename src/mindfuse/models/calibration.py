"""Decision-preserving scalar probability temperature calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar


class TemperatureScaledClassifier:
    """Apply a validation-fitted scalar temperature to any probability estimator."""

    def __init__(self, estimator: Any, temperature: float) -> None:
        self.estimator = estimator
        self.temperature = float(temperature)
        self.classes_ = estimator.classes_

    def predict_proba(self, features: Any) -> np.ndarray:
        probabilities = np.clip(self.estimator.predict_proba(features), 1e-12, 1.0)
        scaled_logits = np.log(probabilities) / self.temperature
        scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
        exponentials = np.exp(scaled_logits)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict(self, features: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(features).argmax(axis=1)]

    def fit(self, features: Any, labels: np.ndarray) -> "TemperatureScaledClassifier":
        self.estimator.fit(features, labels)
        self.classes_ = self.estimator.classes_
        return self


def fit_probability_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Minimize validation negative log-likelihood over one positive temperature."""

    clipped = np.clip(probabilities, 1e-12, 1.0)
    log_probabilities = np.log(clipped)

    def objective(log_temperature: float) -> float:
        scaled = log_probabilities / np.exp(log_temperature)
        scaled -= scaled.max(axis=1, keepdims=True)
        normalized = np.exp(scaled)
        normalized /= normalized.sum(axis=1, keepdims=True)
        return float(-np.log(np.clip(normalized[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())

    result = minimize_scalar(objective, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(result.x))

