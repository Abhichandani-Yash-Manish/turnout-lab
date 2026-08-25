from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

import turnout_lab.database as database
from turnout_lab.database import initialize, log_batch, log_prediction, operations_summary
from turnout_lab.schemas import PredictionResult


def prediction_result() -> PredictionResult:
    return PredictionResult(
        status="scored",
        attendance_probability=0.7,
        no_show_probability=0.3,
        predicted_attendance=True,
        decision_threshold=0.59,
        no_show_risk_band="low",
        reliability="high",
        reason_codes=[],
        warnings=[],
        model_version="test-model",
    )


def test_anonymous_operation_totals_reconcile(tmp_path) -> None:
    database_path = tmp_path / "operations.sqlite3"
    assert log_prediction(database_path, prediction_result())
    assert log_batch(
        database_path,
        "test-model",
        {"row_count": 3, "scored_count": 2, "rejected_count": 1, "review_count": 0},
    )

    summary = operations_summary(database_path)
    assert summary["available"] is True
    assert summary["predictions"]["total"] == 1
    assert summary["batches"]["total"] == 1
    assert summary["batches"]["rows"] == 3

    with sqlite3.connect(database_path) as connection:
        prediction_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(prediction_logs)").fetchall()
        }
    assert "student_id" not in prediction_columns
    assert "raw_features" not in prediction_columns


def test_concurrent_initialization_and_reads_are_serialized(tmp_path) -> None:
    database_path = tmp_path / "concurrent.sqlite3"

    with ThreadPoolExecutor(max_workers=8) as executor:
        summaries = list(executor.map(lambda _: operations_summary(database_path), range(24)))

    assert all(summary["available"] for summary in summaries)
    assert all(summary["predictions"]["total"] == 0 for summary in summaries)


def test_locked_analytics_degrade_without_breaking_predictions(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "locked.sqlite3"
    initialize(database_path)
    monkeypatch.setattr(database, "SQLITE_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(database, "SQLITE_BUSY_TIMEOUT_MS", 1)
    monkeypatch.setattr(database, "LOCK_RETRY_DELAYS", (0.0,))

    with sqlite3.connect(database_path, isolation_level=None) as blocker:
        blocker.execute("BEGIN EXCLUSIVE")
        with pytest.warns(RuntimeWarning, match="analytics read skipped"):
            summary = operations_summary(database_path)
        with pytest.warns(RuntimeWarning, match="analytics write skipped"):
            logged = log_prediction(database_path, prediction_result())
        blocker.execute("ROLLBACK")

    assert summary["available"] is False
    assert summary["predictions"]["total"] == 0
    assert logged is False


def test_connections_do_not_change_journal_mode_per_request(tmp_path) -> None:
    database_path = tmp_path / "journal.sqlite3"
    initialize(database_path)

    with sqlite3.connect(database_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert journal_mode == "delete"
