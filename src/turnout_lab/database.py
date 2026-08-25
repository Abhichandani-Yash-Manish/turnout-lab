"""Anonymous SQLite auditing for local dashboard operations."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from turnout_lab.schemas import BatchPredictionSummary, PredictionResult


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialize(path: Path) -> None:
    with connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                model_version TEXT NOT NULL,
                attendance_probability REAL NOT NULL,
                no_show_risk_band TEXT NOT NULL,
                reliability TEXT NOT NULL,
                warning_codes TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS batch_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_version TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                scored_count INTEGER NOT NULL,
                rejected_count INTEGER NOT NULL,
                review_count INTEGER NOT NULL
            );
            """
        )


def log_prediction(path: Path, result: PredictionResult, source: str = "single") -> None:
    initialize(path)
    with connect(path) as connection:
        connection.execute(
            """
            INSERT INTO prediction_logs (
                created_at, source, model_version, attendance_probability,
                no_show_risk_band, reliability, warning_codes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                source,
                result.model_version,
                result.attendance_probability,
                result.no_show_risk_band.value,
                result.reliability.value,
                json.dumps(result.warnings),
            ),
        )


def log_batch(
    path: Path,
    model_version: str,
    summary: BatchPredictionSummary | dict[str, int],
) -> None:
    if isinstance(summary, BatchPredictionSummary):
        row_count = summary.total_rows
        scored_count = summary.valid_rows
        rejected_count = summary.rejected_rows
        review_count = summary.review_required_rows
    else:
        row_count = summary["row_count"]
        scored_count = summary["scored_count"]
        rejected_count = summary["rejected_count"]
        review_count = summary["review_count"]
    initialize(path)
    with connect(path) as connection:
        connection.execute(
            """
            INSERT INTO batch_runs (
                created_at, model_version, row_count, scored_count, rejected_count, review_count
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                model_version,
                row_count,
                scored_count,
                rejected_count,
                review_count,
            ),
        )


def log_batch_predictions(path: Path, outputs: pd.DataFrame) -> None:
    """Persist anonymous batch scores without student IDs or raw features."""
    initialize(path)
    valid = outputs.loc[outputs["status"].isin(["scored", "review_required"])].copy()
    if valid.empty:
        return
    created_at = datetime.now(timezone.utc).isoformat()
    records = [
        (
            created_at,
            "batch",
            str(row.model_version),
            float(row.attendance_probability),
            str(row.no_show_risk_band),
            str(row.reliability),
            json.dumps([] if pd.isna(row.warnings) or not row.warnings else str(row.warnings).split("; ")),
        )
        for row in valid.itertuples(index=False)
    ]
    with connect(path) as connection:
        connection.executemany(
            """
            INSERT INTO prediction_logs (
                created_at, source, model_version, attendance_probability,
                no_show_risk_band, reliability, warning_codes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )


def operations_summary(path: Path) -> dict[str, Any]:
    initialize(path)
    with connect(path) as connection:
        predictions = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   AVG(attendance_probability) AS average_probability,
                   SUM(CASE WHEN no_show_risk_band = 'high' THEN 1 ELSE 0 END) AS high_risk,
                   SUM(CASE WHEN reliability = 'low' THEN 1 ELSE 0 END) AS review_required
            FROM prediction_logs
            """
        ).fetchone()
        batches = connection.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(row_count), 0) AS rows FROM batch_runs"
        ).fetchone()
        risk_rows = connection.execute(
            "SELECT no_show_risk_band, COUNT(*) AS count FROM prediction_logs GROUP BY no_show_risk_band"
        ).fetchall()
    return {
        "predictions": dict(predictions),
        "batches": dict(batches),
        "risk_distribution": [dict(row) for row in risk_rows],
    }
