"""Small logistic model and decision policy for streaming endpoint detection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import FEATURE_NAMES, extract_features


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 0.08, epochs: int = 1800, lr: float = 0.25) -> tuple[list[float], float]:
    """Batch-gradient logistic regression with standardized inputs and L2 penalty."""
    means = X.mean(axis=0)
    scales = X.std(axis=0)
    scales[scales < 1e-8] = 1.0
    Z = (X - means) / scales
    Z = np.column_stack((np.ones(len(Z)), Z))
    w = np.zeros(Z.shape[1], dtype=np.float64)
    for _ in range(epochs):
        p = sigmoid(Z @ w)
        grad = (Z.T @ (p - y)) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    coef = w[1:] / scales
    intercept = float(w[0] - np.dot(w[1:], means / scales))
    return coef.tolist(), intercept


def predict_probability(features: Sequence[float], coef: Sequence[float], intercept: float) -> float:
    z = float(intercept + np.dot(np.asarray(features, dtype=np.float64), np.asarray(coef, dtype=np.float64)))
    return float(sigmoid(z))


def choose_threshold(probabilities: Sequence[float], labels: Sequence[int], gates: Sequence[bool] | None = None) -> float:
    """Choose threshold minimizing 4*false-positive + false-negative cost after fixed gates."""
    gate = np.ones(len(probabilities), dtype=bool) if gates is None else np.asarray(gates, dtype=bool)
    y = np.asarray(labels, dtype=bool)
    best = (float("inf"), 0.5)
    for threshold in np.arange(0.20, 0.901, 0.01):
        pred = (np.asarray(probabilities) >= threshold) & gate
        fp = int(np.sum(pred & ~y))
        fn = int(np.sum(~pred & y))
        score = 4 * fp + fn
        if score < best[0]:
            best = (score, float(threshold))
    return best[1]


class TurnDetector:
    def __init__(self, model_path: str | Path):
        data = json.loads(Path(model_path).read_text(encoding="utf-8"))
        if data.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Model feature schema does not match this code version")
        self.coef = data["coef"]
        self.intercept = data["intercept"]
        self.threshold = float(data["threshold"])
        self.model_version = data.get("model_version", "unknown")

    def predict(self, text: str, pause_ms: float, stability_ms: float, language: str = "en") -> dict:
        features = extract_features(text, pause_ms, stability_ms, language)
        p_end = predict_probability(features, self.coef, self.intercept)
        reasons: list[str] = []
        if pause_ms < 180:
            action = "wait"
            reasons.append("pause_below_minimum")
        elif stability_ms < 100:
            action = "wait"
            reasons.append("transcript_not_stable")
        else:
            action = "respond" if p_end >= self.threshold else "wait"
            reasons.append("model_confident_endpoint" if action == "respond" else "model_below_threshold")
        return {
            "action": action,
            "p_end": round(p_end, 6),
            "threshold": self.threshold,
            "model_version": self.model_version,
            "reason_codes": reasons,
        }
