from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from turnout_lab.config import METRICS_PATH, MODEL_PATH, TEST_PATH
from turnout_lab.prediction import AttendancePredictor
from turnout_lab.schemas import AttendanceInput


@pytest.fixture(scope="module")
def predictor() -> AttendancePredictor:
    return AttendancePredictor.from_path(MODEL_PATH)


def test_invalid_history_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AttendanceInput(
            event_type="workshop",
            registration_days_before=3,
            previous_events_registered=1,
            previous_events_attended=2,
            club_member="yes",
            event_day="monday",
            event_time="18:00",
            travel_distance_km=3,
        )


def test_probability_contract_and_complement(predictor: AttendancePredictor) -> None:
    reference = predictor.bundle["reference_profile"]
    result = predictor.predict(
        AttendanceInput(
            event_type=reference["event_type"],
            registration_days_before=reference["registration_days_before"],
            previous_events_registered=int(reference["previous_events_registered"]),
            previous_events_attended=int(reference["previous_events_attended"]),
            club_member=reference["club_member"],
            event_day=reference["event_day"],
            event_time=reference["event_time"],
            travel_distance_km=reference["travel_distance_km"],
        )
    )
    assert 0 <= result.attendance_probability <= 1
    assert result.attendance_probability + result.no_show_probability == pytest.approx(1)
    assert result.reliability.value == "high"


def test_unseen_category_requires_review(predictor: AttendancePredictor) -> None:
    reference = predictor.bundle["reference_profile"]
    result = predictor.predict(
        AttendanceInput(
            event_type="unseen-format",
            registration_days_before=reference["registration_days_before"],
            previous_events_registered=int(reference["previous_events_registered"]),
            previous_events_attended=int(reference["previous_events_attended"]),
            club_member=reference["club_member"],
            event_day=reference["event_day"],
            event_time=reference["event_time"],
            travel_distance_km=reference["travel_distance_km"],
        )
    )
    assert result.status.value == "review_required"
    assert result.reliability.value == "low"


def test_official_test_scores_exactly_100_rows(predictor: AttendancePredictor) -> None:
    official = pd.read_csv(TEST_PATH)
    outputs = predictor.score_dataframe(official)
    assert len(outputs) == 100
    assert outputs["student_id"].nunique() == 100
    assert outputs["status"].eq("scored").all()
    assert outputs["error"].eq("").all()
    assert outputs["attendance_probability"].between(0, 1).all()


def test_committed_metrics_are_from_full_protocol() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics["generated_from_quick_run"] is False
    assert metrics["evaluation_protocol"]["outer_seeds"] == [11, 22, 33, 44, 55]
    assert metrics["calibrated_champion"]["summary"]["folds"] == 25
    assert metrics["calibrated_champion"]["summary"]["roc_auc"]["mean"] > 0.60
    assert metrics["calibrated_champion"]["summary"]["brier_skill"]["mean"] > 0

