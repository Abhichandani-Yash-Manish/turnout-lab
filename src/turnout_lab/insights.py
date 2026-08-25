"""Descriptive, non-causal attendance patterns with bootstrap uncertainty."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from turnout_lab.config import TARGET_COLUMN


def bootstrap_rate_interval(
    values: np.ndarray, seed: int = 42, samples: int = 2000
) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        value = float(values[0])
        return (value, value)
    generator = np.random.default_rng(seed)
    indexes = generator.integers(0, len(values), size=(samples, len(values)))
    rates = values[indexes].mean(axis=1)
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def summarize_groups(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (value, group) in enumerate(frame.groupby(column, observed=True, dropna=False)):
        label = "missing" if pd.isna(value) else str(value)
        outcomes = group[TARGET_COLUMN].to_numpy(dtype=int)
        low, high = bootstrap_rate_interval(outcomes, seed=42 + index)
        rows.append(
            {
                "segment": label,
                "n": int(len(group)),
                "attendance_rate": float(outcomes.mean()),
                "ci_95_low": low,
                "ci_95_high": high,
            }
        )
    return sorted(rows, key=lambda item: item["attendance_rate"], reverse=True)


def build_insights(development: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    frame = development.copy()
    distance = pd.to_numeric(frame["travel_distance_km"], errors="coerce")
    frame["distance_band"] = pd.cut(
        distance,
        bins=[-np.inf, 3, 10, 30, np.inf],
        labels=["0–3 km", "3–10 km", "10–30 km", "30+ km"],
    )
    lead = pd.to_numeric(frame["registration_days_before"], errors="coerce")
    frame["registration_band"] = pd.cut(
        lead,
        bins=[-np.inf, 0, 3, 7, 14, np.inf],
        labels=["late / same day", "1–3 days", "4–7 days", "8–14 days", "15+ days"],
    )
    return {
        "club_membership": summarize_groups(frame, "club_member"),
        "event_type": summarize_groups(frame, "event_type"),
        "distance_band": summarize_groups(frame, "distance_band"),
        "registration_band": summarize_groups(frame, "registration_band"),
    }

