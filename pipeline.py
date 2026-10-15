"""Dependency regression-risk baseline on synthetic update metadata."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["major_update", "breaking_api", "files_changed", "lines_changed", "transitive_updates",
            "days_since_release", "test_coverage", "historical_failure_rate"]


def _dataset(seed=SEED, rows=1600):
    rng = np.random.default_rng(seed)
    major = rng.binomial(1, 0.22, rows)
    X = pd.DataFrame({
        "major_update": major,
        "breaking_api": np.where(major == 1, rng.binomial(1, 0.65, rows), rng.binomial(1, 0.06, rows)),
        "files_changed": rng.integers(1, 121, rows),
        "lines_changed": np.clip(rng.lognormal(4.1, 1.0, rows).astype(int), 1, 5000),
        "transitive_updates": rng.integers(0, 46, rows),
        "days_since_release": rng.integers(1, 731, rows),
        "test_coverage": rng.uniform(0.35, 0.99, rows),
        "historical_failure_rate": rng.beta(1.4, 8.0, rows),
    })
    score = (0.85 * X.major_update + 0.85 * X.breaking_api + 0.012 * X.files_changed
             + 0.00045 * X.lines_changed + 0.022 * X.transitive_updates
             + 0.0011 * X.days_since_release - 1.3 * X.test_coverage
             + 2.0 * X.historical_failure_rate + rng.normal(0, 0.28, rows))
    y = (score > 1.55).astype(int)
    return X, y


def _model(seed=SEED):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))


def sample_input():
    return {"major_update": 1, "breaking_api": 1, "files_changed": 42, "lines_changed": 1280,
            "transitive_updates": 18, "days_since_release": 260, "test_coverage": 0.58,
            "historical_failure_rate": 0.12}


def train_model(seed=SEED):
    X, y = _dataset(seed)
    return _model(seed).fit(X, y)


def predict(model, payload):
    row = pd.DataFrame([payload], columns=FEATURES)
    probability = float(model.predict_proba(row)[0, 1])
    return {"regression_risk": bool(probability >= 0.5), "risk_probability": round(probability, 4)}


def evaluate(seed=SEED):
    X, y = _dataset(seed)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    model = _model(seed).fit(x_train, y_train)
    pred = model.predict(x_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])), 4),
        "test_rows": int(len(y_test)), "positive_test_rows": int(y_test.sum()),
    }
