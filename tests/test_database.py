from __future__ import annotations

import sqlite3

from turnout_lab.database import log_batch, log_prediction, operations_summary
from turnout_lab.schemas import PredictionResult


def test_anonymous_operation_totals_reconcile(tmp_path) -> None:
    database_path = tmp_path / "operations.sqlite3"
    result = PredictionResult(
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
    log_prediction(database_path, result)
    log_batch(
        database_path,
        "test-model",
        {"row_count": 3, "scored_count": 2, "rejected_count": 1, "review_count": 0},
    )

    summary = operations_summary(database_path)
    assert summary["predictions"]["total"] == 1
    assert summary["batches"]["total"] == 1
    assert summary["batches"]["rows"] == 3

    with sqlite3.connect(database_path) as connection:
        prediction_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(prediction_logs)").fetchall()
        }
    assert "student_id" not in prediction_columns
    assert "raw_features" not in prediction_columns

