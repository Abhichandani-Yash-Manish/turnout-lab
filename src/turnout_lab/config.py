"""Project-wide paths and immutable feature definitions."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNTIME_DIR = PROJECT_ROOT / "runtime"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
PROVENANCE_PATH = RAW_DATA_DIR / "provenance.json"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
QUALITY_REPORT_PATH = ARTIFACTS_DIR / "data_quality_report.json"
FEATURE_CONTRACT_PATH = ARTIFACTS_DIR / "feature_contract.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
SAMPLE_REGISTRATIONS_PATH = DATA_DIR / "samples" / "demo_registrations.csv"
DATABASE_PATH = RUNTIME_DIR / "turnout_lab.sqlite3"

MODEL_VERSION = "turnout-lab-0.1.0"
TARGET_COLUMN = "attended"
ID_COLUMN = "student_id"

RAW_FEATURE_COLUMNS = [
    "event_type",
    "registration_days_before",
    "previous_events_registered",
    "previous_events_attended",
    "club_member",
    "event_day",
    "event_time",
    "travel_distance_km",
]
REQUIRED_TRAIN_COLUMNS = [ID_COLUMN, *RAW_FEATURE_COLUMNS, TARGET_COLUMN]
REQUIRED_TEST_COLUMNS = [ID_COLUMN, *RAW_FEATURE_COLUMNS]
CATEGORICAL_COLUMNS = ["event_type", "club_member", "event_day", "event_time"]
NUMERIC_COLUMNS = [column for column in RAW_FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]

OUTER_SEEDS = [11, 22, 33, 44, 55]

