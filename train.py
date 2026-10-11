from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from turn_detector.dataset import build_rows, write_csv
from turn_detector.features import FEATURE_NAMES, extract_features
from turn_detector.model import choose_threshold, fit_logistic, predict_probability


def metrics(y: list[int], pred: list[bool]) -> dict:
    prediction = np.asarray(pred, dtype=bool)
    truth = np.asarray(y, dtype=bool)
    tp = int(np.sum(prediction & truth)); fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth)); tn = int(np.sum(~prediction & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": len(y), "precision": precision, "recall": recall, "f1": f1,
            "false_respond_rate": fp / (fp + tn) if fp + tn else 0.0,
            "false_wait_rate": fn / (fn + tp) if fn + tp else 0.0,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def main() -> None:
    rows = build_rows()
    write_csv(ROOT / "data" / "synthetic_turns.csv", rows)
    # Group by scenario template to avoid the same exact prompt template crossing splits.
    train, valid, test = [], [], []
    for row in rows:
        fold = int(row["scenario_id"]) % 5
        (test if fold == 4 else valid if fold == 3 else train).append(row)

    def matrix(part):
        X = np.asarray([extract_features(r["text"], r["pause_ms"], r["stability_ms"], r["language"]) for r in part])
        y = np.asarray([int(r["label_respond"]) for r in part], dtype=np.int64)
        gates = np.asarray([float(r["pause_ms"]) >= 180 and float(r["stability_ms"]) >= 100 for r in part])
        return X, y, gates

    X_train, y_train, _ = matrix(train)
    X_valid, y_valid, gate_valid = matrix(valid)
    X_test, y_test, gate_test = matrix(test)
    coef, intercept = fit_logistic(X_train, y_train.astype(np.float64))
    p_valid = [predict_probability(x, coef, intercept) for x in X_valid]
    threshold = choose_threshold(p_valid, y_valid, gate_valid)
    p_test = [predict_probability(x, coef, intercept) for x in X_test]
    raw_pred = [p >= threshold for p in p_test]
    policy_pred = [bool(p >= threshold and gate) for p, gate in zip(p_test, gate_test)]
    model = {"model_version": "synthetic-logreg-v1", "feature_names": list(FEATURE_NAMES),
             "coef": coef, "intercept": intercept, "threshold": threshold,
             "training_note": "Synthetic prototype only; do not use as a production quality claim."}
    model_path = ROOT / "artifacts" / "model.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    result = {"dataset": "deterministic synthetic templates; not production audio/call data",
              "split_method": "scenario/template grouped; scenario_id mod 5 (train 0-2, validation 3, test 4)",
              "counts": {"train": len(train), "validation": len(valid), "test": len(test)},
              "positive_rate": float(np.mean(y_test)), "selected_threshold": threshold,
              "validation_cost": "4*false_respond + 1*false_wait; pause/stability gates applied",
              "test_policy_with_gates": metrics(y_test.tolist(), policy_pred),
              "test_raw_model_at_same_threshold": metrics(y_test.tolist(), raw_pred)}
    (ROOT / "artifacts" / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
