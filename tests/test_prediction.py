from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest
from pydantic import ValidationError

from turnout_lab.config import METRICS_PATH, MODEL_PATH, PREDICTIONS_PATH, TEST_PATH
from turnout_lab.prediction import AttendancePredictor, summarize_batch
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


def test_committed_official_predictions_preserve_order_and_probability_contract() -> None:
    official = pd.read_csv(TEST_PATH)
    committed = pd.read_csv(PREDICTIONS_PATH)

    assert committed["input_row"].tolist() == list(range(100))
    assert committed["student_id"].tolist() == official["student_id"].tolist()
    assert committed["student_id"].nunique() == 100
    assert committed["status"].eq("scored").all()
    assert committed["attendance_probability"].between(0, 1).all()
    assert committed["no_show_probability"].between(0, 1).all()
    assert (
        committed["attendance_probability"] + committed["no_show_probability"]
    ).to_numpy() == pytest.approx(1)


def test_batch_summary_reconciles_and_excludes_rejected_rows(
    predictor: AttendancePredictor,
) -> None:
    official = pd.read_csv(TEST_PATH).head(2)
    invalid = official.iloc[[0]].copy()
    invalid["previous_events_registered"] = 1
    invalid["previous_events_attended"] = 2
    outputs = predictor.score_dataframe(pd.concat([official, invalid], ignore_index=True))

    summary = summarize_batch(outputs, predictor.bundle["model_version"])
    assert summary.total_rows == 3
    assert summary.valid_rows == 2
    assert summary.scored_rows == 2
    assert summary.rejected_rows == 1
    assert summary.expected_attendees + summary.expected_no_shows == pytest.approx(2)
    assert summary.high_risk_count <= summary.valid_rows


def test_committed_metrics_are_from_full_protocol() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics["generated_from_quick_run"] is False
    assert metrics["evaluation_protocol"]["outer_seeds"] == [11, 22, 33, 44, 55]
    assert metrics["calibrated_champion"]["summary"]["folds"] == 25
    assert metrics["calibrated_champion"]["summary"]["roc_auc"]["mean"] > 0.60
    assert metrics["calibrated_champion"]["summary"]["brier_skill"]["mean"] > 0
    diagnostics = metrics["decision_diagnostics"]
    assert diagnostics["class_order"] == ["no_show", "attended"]
    assert diagnostics["outer_seeds"] == [11, 22, 33, 44, 55]
    assert diagnostics["repeated_oof_predictions"] == 397 * 5
    assert all(
        sum(row) == pytest.approx(1)
        for row in diagnostics["normalized_confusion_matrix"]
    )

def test_shipped_model_is_refit_on_all_labelled_rows_but_scored_on_the_safe_cohort() -> None:
    """Widening the training set must not quietly widen what the metrics claim."""
    bundle = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    assert bundle["refit_on_all_labelled_rows"] is True
    assert bundle["training_rows"] > bundle["evaluation_rows"]
    # Reported performance still describes the leakage-safe cohort only.
    assert bundle["evaluation_rows"] == metrics["dataset"]["rows"]
    assert metrics["champion"]["deployment_refit_rows"] == bundle["training_rows"]
