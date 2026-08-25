"""Evaluation metrics designed for imbalanced probabilistic classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def choose_macro_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a stable threshold, preferring values closest to 0.5 on ties."""
    candidates = np.linspace(0.20, 0.80, 121)
    scored = [
        (f1_score(y_true, probabilities >= threshold, average="macro", zero_division=0), threshold)
        for threshold in candidates
    ]
    best_score = max(score for score, _ in scored)
    tied = [threshold for score, threshold in scored if np.isclose(score, best_score)]
    return float(min(tied, key=lambda value: abs(value - 0.5)))


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    boundaries = np.linspace(0, 1, bins + 1)
    indexes = np.clip(np.digitize(probabilities, boundaries) - 1, 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = indexes == index
        if not mask.any():
            continue
        error += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(error)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int | list[list[int]]]:
    predictions = (probabilities >= threshold).astype(int)
    attendance_precision, attendance_recall, attendance_f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    no_show_precision, no_show_recall, no_show_f1, _ = precision_recall_fscore_support(
        1 - y_true, 1 - predictions, average="binary", zero_division=0
    )
    cutoff = max(int(np.ceil(len(probabilities) * 0.20)), 1)
    highest_risk = np.argsort(1 - probabilities)[-cutoff:]
    actual_no_shows = 1 - y_true
    top_precision = float(actual_no_shows[highest_risk].mean())
    base_no_show_rate = float(actual_no_shows.mean())
    top_recall = float(actual_no_shows[highest_risk].sum() / max(actual_no_shows.sum(), 1))

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "attendance_precision": float(attendance_precision),
        "attendance_recall": float(attendance_recall),
        "attendance_f1": float(attendance_f1),
        "no_show_precision": float(no_show_precision),
        "no_show_recall": float(no_show_recall),
        "no_show_f1": float(no_show_f1),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "no_show_pr_auc": float(average_precision_score(1 - y_true, 1 - probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, np.clip(probabilities, 1e-7, 1 - 1e-7))),
        "ece": expected_calibration_error(y_true, probabilities),
        "top_20_no_show_precision": top_precision,
        "top_20_no_show_recall": top_recall,
        "top_20_no_show_lift": float(top_precision / base_no_show_rate) if base_no_show_rate else 0.0,
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def summarize_fold_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = {"confusion_matrix", "seed", "fold", "candidate", "feature_mode"}
    numeric_keys = [
        key
        for key, value in folds[0].items()
        if key not in excluded and isinstance(value, (int, float, np.floating, np.integer))
    ]
    summary: dict[str, Any] = {"folds": len(folds)}
    for key in numeric_keys:
        values = np.asarray([float(fold[key]) for fold in folds], dtype=float)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return summary


def calibration_points(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 8
) -> list[dict[str, float]]:
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=bins, strategy="quantile")
    return [
        {"mean_predicted_probability": float(pred), "observed_attendance_rate": float(obs)}
        for pred, obs in zip(predicted, observed, strict=True)
    ]

