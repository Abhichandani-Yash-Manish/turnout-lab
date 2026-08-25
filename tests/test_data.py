from __future__ import annotations

import hashlib
import json

import numpy as np
import tomllib

import scripts.leakage_demo as leakage_demo
from turnout_lab.config import PROVENANCE_PATH, RAW_FEATURE_COLUMNS, TEST_PATH, TRAIN_PATH
from turnout_lab.data import feature_fingerprint, prepare_datasets
from turnout_lab.modeling import group_splits


def test_official_overlap_is_quarantined_before_development() -> None:
    prepared = prepare_datasets(TRAIN_PATH, TEST_PATH)

    assert len(prepared.raw_train) == 508
    assert len(prepared.raw_test) == 100
    assert len(prepared.development) == 397
    assert prepared.quality_report["overlap"]["exact_id_and_feature_matches"] == 100
    assert prepared.quality_report["overlap"]["quarantined_training_rows"] == 101

    development_ids = set(prepared.development["student_id"].astype(str))
    test_ids = set(prepared.test["student_id"].astype(str))
    assert development_ids.isdisjoint(test_ids)

    development_fingerprints = set(feature_fingerprint(prepared.development))
    test_fingerprints = set(feature_fingerprint(prepared.test))
    assert development_fingerprints.isdisjoint(test_fingerprints)


def test_group_splits_do_not_share_identity_or_fingerprint() -> None:
    prepared = prepare_datasets(TRAIN_PATH, TEST_PATH)
    development = prepared.development
    y = development["attended"].to_numpy(dtype=int)
    groups = development["_group"].to_numpy()

    for train_indexes, validation_indexes in group_splits(y, groups, 5, 11):
        assert set(groups[train_indexes]).isdisjoint(set(groups[validation_indexes]))


def test_student_id_is_not_a_model_feature() -> None:
    assert "student_id" not in RAW_FEATURE_COLUMNS
    assert RAW_FEATURE_COLUMNS == [
        "event_type",
        "registration_days_before",
        "previous_events_registered",
        "previous_events_attended",
        "club_member",
        "event_day",
        "event_time",
        "travel_distance_km",
    ]


def test_feature_contract_has_finite_numeric_ranges() -> None:
    prepared = prepare_datasets(TRAIN_PATH, TEST_PATH)
    for bounds in prepared.feature_contract["numeric"].values():
        assert bounds["min"] <= bounds["max"]
        assert np.isfinite([bounds["min"], bounds["max"], bounds["p01"], bounds["p99"]]).all()


def test_raw_snapshots_match_committed_provenance() -> None:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

    for path in (TRAIN_PATH, TEST_PATH):
        expected = provenance["files"][path.name]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]

def test_leakage_unaware_pipeline_memorizes_the_official_test_split() -> None:
    """The overlap is severe enough that a naive model recites the test labels."""
    result = leakage_demo.run()

    assert result["recovered"] >= 99
    assert result["accuracy"] >= 0.95
    assert result["macro_f1"] >= 0.95

def test_requirements_txt_matches_pyproject_runtime_dependencies() -> None:
    """Two dependency sources must never drift apart."""
    project_root = TRAIN_PATH.parents[2]
    declared = tomllib.loads((project_root / "pyproject.toml").read_text())["project"]["dependencies"]
    pinned = [
        line.strip()
        for line in (project_root / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert pinned == declared
