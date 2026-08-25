from __future__ import annotations

import numpy as np

from turnout_lab.config import RAW_FEATURE_COLUMNS, TEST_PATH, TRAIN_PATH
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

