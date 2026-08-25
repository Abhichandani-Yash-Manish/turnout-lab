"""Deterministic feature normalization and engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from turnout_lab.config import CATEGORICAL_COLUMNS, RAW_FEATURE_COLUMNS

ENGINEERED_NUMERIC_COLUMNS = [
    "registration_days_model",
    "previous_events_registered",
    "previous_events_attended",
    "distance_log",
    "previous_no_shows",
    "previous_attendance_rate",
    "has_history",
    "is_weekend",
    "event_hour",
    "history_inconsistent",
    "late_registration",
    "distance_outlier",
]


def normalize_category(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "n/a", "na"}:
        return np.nan
    if text in {"true", "1", "y"}:
        return "yes"
    if text in {"false", "0", "n"}:
        return "no"
    return text


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in RAW_FEATURE_COLUMNS:
        if column not in normalized:
            normalized[column] = np.nan
    for column in CATEGORICAL_COLUMNS:
        normalized[column] = normalized[column].map(normalize_category)
    for column in set(RAW_FEATURE_COLUMNS) - set(CATEGORICAL_COLUMNS):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def build_model_frame(frame: pd.DataFrame, feature_mode: str = "engineered") -> pd.DataFrame:
    clean = normalize_features(frame)
    if feature_mode == "raw":
        return clean[RAW_FEATURE_COLUMNS].copy()
    if feature_mode != "engineered":
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    output = clean[CATEGORICAL_COLUMNS].copy()
    registered = clean["previous_events_registered"]
    attended = clean["previous_events_attended"]
    history_inconsistent = registered.notna() & attended.notna() & (attended > registered)
    valid_history = registered.notna() & attended.notna() & (registered > 0) & ~history_inconsistent

    output["registration_days_model"] = clean["registration_days_before"].clip(lower=0)
    output["previous_events_registered"] = registered
    output["previous_events_attended"] = attended
    output["distance_log"] = np.log1p(clean["travel_distance_km"].clip(lower=0))
    output["previous_no_shows"] = (registered - attended).clip(lower=0)
    output["previous_attendance_rate"] = np.where(valid_history, attended / registered, np.nan)
    output["has_history"] = (registered.fillna(0) > 0).astype(int)
    output["is_weekend"] = clean["event_day"].isin(["saturday", "sunday"]).astype(int)
    output["event_hour"] = pd.to_numeric(
        clean["event_time"].astype("string").str.extract(r"(\d{1,2})")[0], errors="coerce"
    )
    output["history_inconsistent"] = history_inconsistent.astype(int)
    output["late_registration"] = (clean["registration_days_before"] < 0).fillna(False).astype(int)
    output["distance_outlier"] = (clean["travel_distance_km"] > 30).fillna(False).astype(int)
    return output[[*CATEGORICAL_COLUMNS, *ENGINEERED_NUMERIC_COLUMNS]]


class ModelFeatureTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible wrapper around the shared feature logic."""

    def __init__(self, feature_mode: str = "engineered") -> None:
        self.feature_mode = feature_mode

    def fit(self, X: pd.DataFrame, y: object = None) -> ModelFeatureTransformer:  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return build_model_frame(X, self.feature_mode)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        if self.feature_mode == "raw":
            return np.asarray(RAW_FEATURE_COLUMNS, dtype=object)
        return np.asarray([*CATEGORICAL_COLUMNS, *ENGINEERED_NUMERIC_COLUMNS], dtype=object)
