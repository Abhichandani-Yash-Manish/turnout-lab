"""Leakage-safe model comparison, calibration, and final artifact training."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from turnout_lab.config import (
    CATEGORICAL_COLUMNS,
    MODEL_VERSION,
    OUTER_SEEDS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from turnout_lab.features import ENGINEERED_NUMERIC_COLUMNS, ModelFeatureTransformer
from turnout_lab.metrics import (
    calibration_points,
    choose_macro_f1_threshold,
    classification_metrics,
    summarize_fold_metrics,
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    feature_mode: str
    estimator: BaseEstimator
    parameter_grid: dict[str, list[Any]]

    @property
    def label(self) -> str:
        return f"{self.name} · {self.feature_mode}"


def candidate_specs(quick: bool = False) -> list[CandidateSpec]:
    trees = 120 if quick else 300
    return [
        CandidateSpec("dummy", "raw", DummyClassifier(strategy="prior"), {}),
        CandidateSpec(
            "logistic_regression",
            "raw",
            LogisticRegression(max_iter=2500, solver="liblinear", random_state=42),
            {"model__C": [0.1, 1.0], "model__class_weight": [None, "balanced"]},
        ),
        CandidateSpec(
            "logistic_regression",
            "engineered",
            LogisticRegression(max_iter=2500, solver="liblinear", random_state=42),
            {"model__C": [0.1, 1.0], "model__class_weight": [None, "balanced"]},
        ),
        CandidateSpec(
            "random_forest",
            "raw",
            RandomForestClassifier(
                n_estimators=trees,
                class_weight="balanced_subsample",
                max_features="sqrt",
                random_state=42,
                n_jobs=1,
            ),
            {"model__max_depth": [6, None], "model__min_samples_leaf": [3, 8]},
        ),
        CandidateSpec(
            "random_forest",
            "engineered",
            RandomForestClassifier(
                n_estimators=trees,
                class_weight="balanced_subsample",
                max_features="sqrt",
                random_state=42,
                n_jobs=1,
            ),
            {"model__max_depth": [6, None], "model__min_samples_leaf": [3, 8]},
        ),
        CandidateSpec(
            "gradient_boosting",
            "raw",
            GradientBoostingClassifier(random_state=42, min_samples_leaf=8),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [1, 2],
                "model__learning_rate": [0.03],
            },
        ),
        CandidateSpec(
            "gradient_boosting",
            "engineered",
            GradientBoostingClassifier(random_state=42, min_samples_leaf=8),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [1, 2],
                "model__learning_rate": [0.03],
            },
        ),
    ]


def build_pipeline(spec: CandidateSpec) -> Pipeline:
    numeric_columns = (
        [column for column in RAW_FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]
        if spec.feature_mode == "raw"
        else ENGINEERED_NUMERIC_COLUMNS
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline(
        [
            ("features", ModelFeatureTransformer(spec.feature_mode)),
            ("preprocess", preprocessor),
            ("model", clone(spec.estimator)),
        ]
    )


def group_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indexes = np.arange(len(y))
    return list(splitter.split(indexes, y, groups))


def tune_candidate(
    spec: CandidateSpec,
    X: pd.DataFrame,  # noqa: N803
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> Pipeline:
    pipeline = build_pipeline(spec)
    if not spec.parameter_grid:
        return pipeline
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=spec.parameter_grid,
        scoring="roc_auc",
        cv=splits,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    return search.best_estimator_


def evaluate_candidates(
    development: pd.DataFrame,
    quick: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], CandidateSpec]:
    X = development[RAW_FEATURE_COLUMNS].copy()
    y = development[TARGET_COLUMN].to_numpy(dtype=int)
    groups = development["_group"].to_numpy()
    outer_seeds = OUTER_SEEDS[:1] if quick else OUTER_SEEDS
    outer_folds = 3 if quick else 5
    inner_folds = 3 if quick else 4
    all_fold_metrics: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for spec in candidate_specs(quick):
        candidate_folds: list[dict[str, Any]] = []
        for seed in outer_seeds:
            for fold, (train_indexes, validation_indexes) in enumerate(
                group_splits(y, groups, outer_folds, seed), start=1
            ):
                X_train, X_validation = X.iloc[train_indexes], X.iloc[validation_indexes]
                y_train, y_validation = y[train_indexes], y[validation_indexes]
                groups_train = groups[train_indexes]
                inner_splits = group_splits(y_train, groups_train, inner_folds, seed + fold)
                best = tune_candidate(spec, X_train, y_train, inner_splits)
                inner_probabilities = cross_val_predict(
                    clone(best),
                    X_train,
                    y_train,
                    cv=inner_splits,
                    method="predict_proba",
                    n_jobs=-1,
                )[:, 1]
                threshold = choose_macro_f1_threshold(y_train, inner_probabilities)
                best.fit(X_train, y_train)
                probabilities = best.predict_proba(X_validation)[:, 1]
                fold_metrics = classification_metrics(y_validation, probabilities, threshold)
                fold_metrics.update(
                    {
                        "seed": seed,
                        "fold": fold,
                        "candidate": spec.name,
                        "feature_mode": spec.feature_mode,
                    }
                )
                candidate_folds.append(fold_metrics)
                all_fold_metrics.append(fold_metrics)

        summary = summarize_fold_metrics(candidate_folds)
        summary.update({"candidate": spec.name, "feature_mode": spec.feature_mode, "label": spec.label})
        summaries.append(summary)

    champion = select_champion(summaries, candidate_specs(quick))
    return summaries, all_fold_metrics, champion


def select_champion(summaries: list[dict[str, Any]], specs: list[CandidateSpec]) -> CandidateSpec:
    by_label = {summary["label"]: summary for summary in summaries}
    dummy_brier = by_label["dummy · raw"]["brier"]["mean"]
    non_dummy = [summary for summary in summaries if summary["candidate"] != "dummy"]
    best_macro = max(summary["macro_f1"]["mean"] for summary in non_dummy)
    eligible = [
        summary
        for summary in non_dummy
        if summary["roc_auc"]["mean"] > 0.5
        and summary["brier"]["mean"] <= dummy_brier
        and summary["macro_f1"]["mean"] >= best_macro - 0.03
    ]
    if not eligible:
        eligible = sorted(non_dummy, key=lambda item: item["brier"]["mean"])

    highest_auc = max(summary["roc_auc"]["mean"] for summary in eligible)
    close = [summary for summary in eligible if highest_auc - summary["roc_auc"]["mean"] <= 0.01]
    complexity = {"logistic_regression": 0, "random_forest": 1, "gradient_boosting": 2}
    winner = min(
        close,
        key=lambda item: (
            item["brier"]["mean"],
            complexity.get(item["candidate"], 99),
            0 if item["feature_mode"] == "raw" else 1,
        ),
    )
    return next(
        spec
        for spec in specs
        if spec.name == winner["candidate"] and spec.feature_mode == winner["feature_mode"]
    )


def _calibrated_oof(
    estimator: Pipeline,
    X: pd.DataFrame,  # noqa: N803
    y: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> np.ndarray:
    probabilities = np.full(len(y), np.nan, dtype=float)
    for fold, (train_indexes, validation_indexes) in enumerate(splits, start=1):
        nested_splits = group_splits(
            y[train_indexes], groups[train_indexes], min(3, len(np.unique(groups[train_indexes]))), seed + fold
        )
        calibrator = CalibratedClassifierCV(
            estimator=clone(estimator), method="sigmoid", cv=nested_splits, ensemble=True
        )
        calibrator.fit(X.iloc[train_indexes], y[train_indexes])
        probabilities[validation_indexes] = calibrator.predict_proba(X.iloc[validation_indexes])[:, 1]
    if np.isnan(probabilities).any():
        raise RuntimeError("Calibrated out-of-fold predictions are incomplete.")
    return probabilities


def evaluate_calibrated_champion(
    development: pd.DataFrame,
    champion: CandidateSpec,
    quick: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    X = development[RAW_FEATURE_COLUMNS].copy()
    y = development[TARGET_COLUMN].to_numpy(dtype=int)
    groups = development["_group"].to_numpy()
    outer_seeds = OUTER_SEEDS[:1] if quick else OUTER_SEEDS
    outer_folds = 3 if quick else 5
    inner_folds = 3 if quick else 4
    fold_metrics: list[dict[str, Any]] = []
    all_probabilities: list[float] = []
    all_targets: list[int] = []
    importances: list[np.ndarray] = []

    for seed in outer_seeds:
        for fold, (train_indexes, validation_indexes) in enumerate(
            group_splits(y, groups, outer_folds, seed), start=1
        ):
            X_train, X_validation = X.iloc[train_indexes], X.iloc[validation_indexes]
            y_train, y_validation = y[train_indexes], y[validation_indexes]
            groups_train = groups[train_indexes]
            inner_splits = group_splits(y_train, groups_train, inner_folds, seed + fold)
            best = tune_candidate(champion, X_train, y_train, inner_splits)
            inner_probabilities = _calibrated_oof(
                best, X_train, y_train, groups_train, inner_splits, seed + fold * 10
            )
            threshold = choose_macro_f1_threshold(y_train, inner_probabilities)
            calibrator = CalibratedClassifierCV(
                estimator=clone(best), method="sigmoid", cv=inner_splits, ensemble=True
            )
            calibrator.fit(X_train, y_train)
            probabilities = calibrator.predict_proba(X_validation)[:, 1]
            metrics = classification_metrics(y_validation, probabilities, threshold)
            metrics.update({"seed": seed, "fold": fold})
            fold_metrics.append(metrics)
            all_probabilities.extend(probabilities.tolist())
            all_targets.extend(y_validation.tolist())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = permutation_importance(
                    calibrator,
                    X_validation,
                    y_validation,
                    scoring="roc_auc",
                    n_repeats=3 if quick else 7,
                    random_state=seed + fold,
                    n_jobs=-1,
                )
            importances.append(result.importances_mean)

    probabilities_array = np.asarray(all_probabilities)
    targets_array = np.asarray(all_targets)
    aggregate_threshold = float(np.mean([fold["threshold"] for fold in fold_metrics]))
    aggregate = classification_metrics(targets_array, probabilities_array, aggregate_threshold)
    mean_importance = np.mean(np.vstack(importances), axis=0)
    std_importance = np.std(np.vstack(importances), axis=0)
    feature_importance = sorted(
        [
            {"feature": feature, "importance_mean": float(mean), "importance_std": float(std)}
            for feature, mean, std in zip(RAW_FEATURE_COLUMNS, mean_importance, std_importance, strict=True)
        ],
        key=lambda item: item["importance_mean"],
        reverse=True,
    )
    return (
        {
            "candidate": champion.name,
            "feature_mode": champion.feature_mode,
            "summary": summarize_fold_metrics(fold_metrics),
            "aggregate_repeated_oof": aggregate,
            "fold_metrics": fold_metrics,
            "calibration_points": calibration_points(targets_array, probabilities_array),
            "feature_importance": feature_importance,
        },
        targets_array,
        probabilities_array,
    )


def train_final_bundle(
    development: pd.DataFrame,
    champion: CandidateSpec,
    feature_contract: dict[str, Any],
    quick: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    X = development[RAW_FEATURE_COLUMNS].copy()
    y = development[TARGET_COLUMN].to_numpy(dtype=int)
    groups = development["_group"].to_numpy()
    folds = 3 if quick else 5
    splits = group_splits(y, groups, folds, 2026)
    best = tune_candidate(champion, X, y, splits)
    oof_probabilities = _calibrated_oof(best, X, y, groups, splits, 3030)
    threshold = choose_macro_f1_threshold(y, oof_probabilities)
    no_show_probabilities = 1 - oof_probabilities
    risk_median = float(np.quantile(no_show_probabilities, 0.50))
    risk_high = float(np.quantile(no_show_probabilities, 0.80))
    final_model = CalibratedClassifierCV(
        estimator=clone(best), method="sigmoid", cv=splits, ensemble=True
    )
    final_model.fit(X, y)
    bundle = {
        "model": final_model,
        "model_version": MODEL_VERSION,
        "candidate": champion.name,
        "feature_mode": champion.feature_mode,
        "decision_threshold": float(threshold),
        "risk_thresholds": {"medium": risk_median, "high": risk_high},
        "feature_contract": feature_contract,
        "reference_profile": feature_contract["reference_profile"],
        "training_rows": int(len(development)),
    }
    return bundle, oof_probabilities


def refit_for_deployment(
    bundle: dict[str, Any],
    full_labelled: pd.DataFrame,
    quick: bool = False,
) -> dict[str, Any]:
    """Refit the already-selected pipeline on every labelled row.

    Model choice, hyperparameters, decision threshold, and risk bands are all
    fixed beforehand on the leakage-safe cohort, so nothing here is tuned or
    measured against data the evaluation never saw. This only widens the
    training set for final scoring, which is the usual refit-before-predict
    step; the reported metrics deliberately stay the leakage-safe ones.
    """
    X = full_labelled[RAW_FEATURE_COLUMNS].copy()
    y = full_labelled[TARGET_COLUMN].to_numpy(dtype=int)
    groups = full_labelled["_group"].to_numpy()
    splits = group_splits(y, groups, 3 if quick else 5, 2026)
    estimator = clone(bundle["model"].estimator)
    deployed = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=splits, ensemble=True)
    deployed.fit(X, y)
    return {
        **bundle,
        "model": deployed,
        "training_rows": int(len(full_labelled)),
        "evaluation_rows": int(bundle["training_rows"]),
        "refit_on_all_labelled_rows": True,
    }


def write_training_artifacts(
    model_path: Path,
    metrics_path: Path,
    bundle: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
