"""Anonymous SQLite auditing for local dashboard operations."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from turnout_lab.schemas import BatchPredictionSummary, PredictionResult

SQLITE_TIMEOUT_SECONDS = 2.0
SQLITE_BUSY_TIMEOUT_MS = 2_000
LOCK_RETRY_DELAYS = (0.05, 0.15, 0.30)
_DATABASE_LOCK = threading.RLock()
_T = TypeVar("_T")


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=SQLITE_TIMEOUT_SECONDS)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _is_lock_error(error: sqlite3.OperationalError) -> bool:
    message = str(error).lower()
    return "locked" in message or "busy" in message


def _retry_locked(operation: Callable[[], _T]) -> _T:
    """Retry only transient SQLite lock failures; propagate every other error."""
    for attempt in range(len(LOCK_RETRY_DELAYS) + 1):
        try:
            return operation()
        except sqlite3.OperationalError as error:
            if not _is_lock_error(error) or attempt == len(LOCK_RETRY_DELAYS):
                raise
            time.sleep(LOCK_RETRY_DELAYS[attempt])
    raise RuntimeError("unreachable SQLite retry state")


def _run_serialized(operation: Callable[[], _T]) -> _T:
    """Serialize Streamlit threads before relying on SQLite's process lock."""
    with _DATABASE_LOCK:
        return _retry_locked(operation)


def _initialize_unlocked(path: Path) -> None:
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


def initialize(path: Path) -> None:
    _run_serialized(lambda: _initialize_unlocked(path))


def _optional_write(operation: Callable[[], None]) -> bool:
    """Keep optional analytics failures from interrupting model predictions."""
    try:
        _run_serialized(operation)
    except (sqlite3.Error, OSError) as error:
        warnings.warn(f"Anonymous analytics write skipped: {error}", RuntimeWarning, stacklevel=2)
        return False
    return True


def log_prediction(path: Path, result: PredictionResult, source: str = "single") -> bool:
    def operation() -> None:
        _initialize_unlocked(path)
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

    return _optional_write(operation)


def log_batch(
    path: Path,
    model_version: str,
    summary: BatchPredictionSummary | dict[str, int],
) -> bool:
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
    def operation() -> None:
        _initialize_unlocked(path)
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

    return _optional_write(operation)


def log_batch_predictions(path: Path, outputs: pd.DataFrame) -> bool:
    """Persist anonymous batch scores without student IDs or raw features."""
    valid = outputs.loc[outputs["status"].isin(["scored", "review_required"])].copy()
    if valid.empty:
        return True
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
    def operation() -> None:
        _initialize_unlocked(path)
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

    return _optional_write(operation)


def _unavailable_summary(error: Exception) -> dict[str, Any]:
    return {
        "available": False,
        "warning": f"Anonymous analytics are temporarily unavailable ({type(error).__name__}).",
        "predictions": {
            "total": 0,
            "average_probability": None,
            "high_risk": 0,
            "review_required": 0,
        },
        "batches": {"total": 0, "rows": 0},
        "risk_distribution": [],
    }


def operations_summary(path: Path) -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        _initialize_unlocked(path)
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
            "available": True,
            "warning": None,
            "predictions": dict(predictions),
            "batches": dict(batches),
            "risk_distribution": [dict(row) for row in risk_rows],
        }

    try:
        return _run_serialized(operation)
    except (sqlite3.Error, OSError) as error:
        warnings.warn(f"Anonymous analytics read skipped: {error}", RuntimeWarning, stacklevel=2)
        return _unavailable_summary(error)
