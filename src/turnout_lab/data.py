"""Data lineage, leakage quarantine, profiling, and feature contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from turnout_lab.config import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    RAW_FEATURE_COLUMNS,
    REQUIRED_TEST_COLUMNS,
    REQUIRED_TRAIN_COLUMNS,
    TARGET_COLUMN,
)
from turnout_lab.features import normalize_features


class DataContractError(ValueError):
    """Raised when an input table does not satisfy the feature contract."""


@dataclass(frozen=True)
class PreparedData:
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame
    development: pd.DataFrame
    full_labelled: pd.DataFrame
    test: pd.DataFrame
    quarantine_index: pd.DataFrame
    quality_report: dict[str, Any]
    feature_contract: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataContractError(f"{name} is missing required columns: {', '.join(missing)}")


def feature_fingerprint(frame: pd.DataFrame) -> pd.Series:
    normalized = normalize_features(frame[RAW_FEATURE_COLUMNS])
    stable = normalized.copy()
    for column in stable.columns:
        stable[column] = stable[column].map(
            lambda value: "<NA>" if pd.isna(value) else format(value, ".12g") if isinstance(value, float) else str(value)
        )
    joined = stable.astype(str).agg("|".join, axis=1)
    return joined.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def _category_variants(series: pd.Series) -> dict[str, list[str]]:
    variants: dict[str, set[str]] = {}
    for value in series.dropna().astype(str):
        key = value.strip().lower()
        variants.setdefault(key, set()).add(value)
    return {key: sorted(values) for key, values in variants.items() if len(values) > 1}


def _connected_groups(frame: pd.DataFrame) -> pd.Series:
    parent = list(range(len(frame)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for keys in [frame[ID_COLUMN].astype("string").fillna("<NA>"), feature_fingerprint(frame)]:
        first_seen: dict[str, int] = {}
        for index, key in enumerate(keys.astype(str)):
            if key in first_seen:
                union(index, first_seen[key])
            else:
                first_seen[key] = index
    roots = [find(index) for index in range(len(frame))]
    compact = {root: group_id for group_id, root in enumerate(dict.fromkeys(roots))}
    return pd.Series([f"group-{compact[root]:04d}" for root in roots], index=frame.index)


def prepare_datasets(train_path: Path, test_path: Path) -> PreparedData:
    raw_train = pd.read_csv(train_path)
    raw_test = pd.read_csv(test_path)
    validate_columns(raw_train, REQUIRED_TRAIN_COLUMNS, "Training data")
    validate_columns(raw_test, REQUIRED_TEST_COLUMNS, "Test data")

    train = raw_train.copy()
    test = raw_test.copy()
    train["_fingerprint"] = feature_fingerprint(train)
    test["_fingerprint"] = feature_fingerprint(test)

    test_ids = set(test[ID_COLUMN].astype(str))
    test_fingerprints = set(test["_fingerprint"].astype(str))
    id_overlap = train[ID_COLUMN].astype(str).isin(test_ids)
    feature_overlap = train["_fingerprint"].astype(str).isin(test_fingerprints)
    quarantine_mask = id_overlap | feature_overlap

    quarantine_index = train.loc[quarantine_mask, [ID_COLUMN, "_fingerprint"]].copy()
    quarantine_index["matched_by_id"] = id_overlap[quarantine_mask].to_numpy()
    quarantine_index["matched_by_features"] = feature_overlap[quarantine_mask].to_numpy()

    development = train.loc[~quarantine_mask].copy()
    before_missing_target = len(development)
    development = development.loc[development[TARGET_COLUMN].notna()].copy()
    missing_target_removed = before_missing_target - len(development)
    before_duplicates = len(development)
    development = development.drop_duplicates(subset=REQUIRED_TRAIN_COLUMNS).reset_index(drop=True)
    development_duplicates_removed = before_duplicates - len(development)

    development[TARGET_COLUMN] = pd.to_numeric(development[TARGET_COLUMN], errors="raise").astype(int)
    normalized = normalize_features(development[RAW_FEATURE_COLUMNS])
    development.loc[:, RAW_FEATURE_COLUMNS] = normalized
    development["_group"] = _connected_groups(development).to_numpy()

    # Same cleaning as the development cohort, minus the leakage quarantine. Used only to
    # refit the already-selected pipeline for final scoring, never to evaluate it.
    full_labelled = train.loc[train[TARGET_COLUMN].notna()].copy()
    full_labelled = full_labelled.drop_duplicates(subset=REQUIRED_TRAIN_COLUMNS).reset_index(drop=True)
    full_labelled[TARGET_COLUMN] = pd.to_numeric(full_labelled[TARGET_COLUMN], errors="raise").astype(int)
    full_labelled.loc[:, RAW_FEATURE_COLUMNS] = normalize_features(full_labelled[RAW_FEATURE_COLUMNS])
    full_labelled["_group"] = _connected_groups(full_labelled).to_numpy()

    normalized_test = normalize_features(test[RAW_FEATURE_COLUMNS])
    test.loc[:, RAW_FEATURE_COLUMNS] = normalized_test

    full_history_invalid = (
        pd.to_numeric(raw_train["previous_events_attended"], errors="coerce")
        > pd.to_numeric(raw_train["previous_events_registered"], errors="coerce")
    ).fillna(False)
    negative_lead = (pd.to_numeric(raw_train["registration_days_before"], errors="coerce") < 0).fillna(False)
    distance = pd.to_numeric(raw_train["travel_distance_km"], errors="coerce")

    exact_test_matches = 0
    for _, row in test.iterrows():
        match = train[ID_COLUMN].astype(str).eq(str(row[ID_COLUMN])) & train["_fingerprint"].eq(row["_fingerprint"])
        exact_test_matches += int(match.any())

    quality_report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_grain": "one student registration for one event",
        "raw": {
            "train_rows": int(len(raw_train)),
            "test_rows": int(len(raw_test)),
            "train_columns": int(len(raw_train.columns)),
            "test_columns": int(len(raw_test.columns)),
            "train_missing_by_column": {key: int(value) for key, value in raw_train.isna().sum().items()},
            "test_missing_by_column": {key: int(value) for key, value in raw_test.isna().sum().items()},
            "exact_duplicate_train_rows": int(raw_train.duplicated().sum()),
            "duplicate_train_student_ids": int(raw_train[ID_COLUMN].duplicated().sum()),
            "missing_targets": int(raw_train[TARGET_COLUMN].isna().sum()),
            "category_variants": {
                column: _category_variants(raw_train[column]) for column in CATEGORICAL_COLUMNS
            },
            "history_inconsistencies": int(full_history_invalid.sum()),
            "negative_registration_days": int(negative_lead.sum()),
            "distance_max_km": None if distance.dropna().empty else float(distance.max()),
            "distance_p99_km": None if distance.dropna().empty else float(distance.quantile(0.99)),
        },
        "overlap": {
            "test_rows": int(len(test)),
            "test_ids_present_in_train": int(test[ID_COLUMN].astype(str).isin(set(train[ID_COLUMN].astype(str))).sum()),
            "test_feature_rows_present_in_train": int(test["_fingerprint"].isin(set(train["_fingerprint"])).sum()),
            "exact_id_and_feature_matches": int(exact_test_matches),
            "quarantined_training_rows": int(quarantine_mask.sum()),
            "severity": "critical",
            "impact": "The official test set is not independent and cannot estimate generalization.",
            "remediation": "Quarantine matching training rows before target analysis and use grouped cross-validation on the remaining cohort.",
        },
        "development": {
            "rows": int(len(development)),
            "unique_student_ids": int(development[ID_COLUMN].nunique()),
            "groups": int(development["_group"].nunique()),
            "missing_targets_removed": int(missing_target_removed),
            "exact_duplicates_removed": int(development_duplicates_removed),
            "attendance_count": int(development[TARGET_COLUMN].sum()),
            "no_show_count": int((1 - development[TARGET_COLUMN]).sum()),
            "attendance_rate": float(development[TARGET_COLUMN].mean()),
        },
        "deployment_refit": {
            "rows": int(len(full_labelled)),
            "rationale": "Performance is estimated on the leakage-safe cohort; the selected pipeline is refit on every labelled row before final scoring.",
        },
    }

    feature_contract = make_feature_contract(development)
    return PreparedData(
        raw_train=raw_train,
        raw_test=raw_test,
        development=development,
        full_labelled=full_labelled,
        test=test,
        quarantine_index=quarantine_index,
        quality_report=quality_report,
        feature_contract=feature_contract,
    )


def make_feature_contract(development: pd.DataFrame) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "required_columns": RAW_FEATURE_COLUMNS,
        "categorical": {},
        "numeric": {},
        "reference_profile": {},
    }
    for column in CATEGORICAL_COLUMNS:
        series = development[column].dropna().astype(str)
        values = sorted(series.unique().tolist())
        contract["categorical"][column] = {"allowed_values": values}
        contract["reference_profile"][column] = None if series.empty else str(series.mode().iloc[0])
    for column in NUMERIC_COLUMNS:
        series = pd.to_numeric(development[column], errors="coerce").dropna()
        contract["numeric"][column] = {
            "min": None if series.empty else float(series.min()),
            "max": None if series.empty else float(series.max()),
            "p01": None if series.empty else float(series.quantile(0.01)),
            "p99": None if series.empty else float(series.quantile(0.99)),
        }
        contract["reference_profile"][column] = None if series.empty else float(series.median())
    return contract


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
