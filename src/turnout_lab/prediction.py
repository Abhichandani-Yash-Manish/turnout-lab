"""Prediction service shared by CLI, tests, and Streamlit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError

from turnout_lab.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, RAW_FEATURE_COLUMNS
from turnout_lab.features import normalize_category
from turnout_lab.schemas import (
    AttendanceInput,
    BatchPredictionSummary,
    PredictionResult,
    PredictionStatus,
    Reliability,
    RiskBand,
)

FEATURE_LABELS = {
    "event_type": "event format",
    "registration_days_before": "registration lead time",
    "previous_events_registered": "registration history",
    "previous_events_attended": "attendance history",
    "club_member": "club membership",
    "event_day": "event day",
    "event_time": "event time",
    "travel_distance_km": "travel distance",
}


class AttendancePredictor:
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle
        self.model = bundle["model"]
        self.contract = bundle["feature_contract"]

    @classmethod
    def from_path(cls, path: Path) -> AttendancePredictor:
        return cls(joblib.load(path))

    def _reliability(self, features: dict[str, Any]) -> tuple[PredictionStatus, Reliability, list[str]]:
        warnings: list[str] = []
        missing_count = sum(value is None or (isinstance(value, float) and np.isnan(value)) for value in features.values())
        hard_review = False
        mild_issue = False

        for column in CATEGORICAL_COLUMNS:
            value = normalize_category(features.get(column))
            if pd.isna(value):
                continue
            allowed = self.contract["categorical"][column]["allowed_values"]
            if str(value) not in allowed:
                warnings.append(f"Unseen {FEATURE_LABELS[column]} category: {value}.")
                hard_review = True

        for column in NUMERIC_COLUMNS:
            value = features.get(column)
            if value is None or pd.isna(value):
                continue
            bounds = self.contract["numeric"][column]
            numeric = float(value)
            if bounds["min"] is not None and (numeric < bounds["min"] or numeric > bounds["max"]):
                warnings.append(f"{FEATURE_LABELS[column].capitalize()} is outside the observed training range.")
                hard_review = True
            elif bounds["p01"] is not None and (numeric < bounds["p01"] or numeric > bounds["p99"]):
                warnings.append(f"{FEATURE_LABELS[column].capitalize()} is unusual in the development cohort.")
                mild_issue = True

        if missing_count >= 3:
            warnings.append("Three or more model inputs are missing.")
            hard_review = True
        elif missing_count:
            warnings.append(f"{missing_count} model input(s) were imputed.")
            mild_issue = True

        if hard_review:
            return PredictionStatus.REVIEW_REQUIRED, Reliability.LOW, warnings
        if mild_issue:
            return PredictionStatus.SCORED, Reliability.MEDIUM, warnings
        return PredictionStatus.SCORED, Reliability.HIGH, warnings

    def _reason_codes(self, frame: pd.DataFrame, probability: float) -> list[str]:
        reference = self.bundle["reference_profile"]
        deltas: list[tuple[float, str]] = []
        for column in RAW_FEATURE_COLUMNS:
            if frame.iloc[0][column] is None or pd.isna(frame.iloc[0][column]):
                continue
            counterfactual = frame.copy()
            counterfactual.loc[counterfactual.index[0], column] = reference[column]
            reference_probability = float(self.model.predict_proba(counterfactual)[:, 1][0])
            delta = probability - reference_probability
            if abs(delta) < 0.01:
                continue
            direction = "raised" if delta > 0 else "lowered"
            message = (
                f"{FEATURE_LABELS[column].capitalize()} {direction} the estimate by "
                f"{abs(delta) * 100:.1f} points versus the reference profile."
            )
            deltas.append((abs(delta), message))
        return [message for _, message in sorted(deltas, reverse=True)[:3]]

    def _build_result(
        self,
        attendance_input: AttendanceInput,
        frame: pd.DataFrame,
        probability: float,
        include_reasons: bool,
    ) -> PredictionResult:
        features = attendance_input.feature_dict()
        no_show = 1 - probability
        threshold = float(self.bundle["decision_threshold"])
        risk_thresholds = self.bundle["risk_thresholds"]
        if no_show >= risk_thresholds["high"]:
            risk_band = RiskBand.HIGH
        elif no_show >= risk_thresholds["medium"]:
            risk_band = RiskBand.MEDIUM
        else:
            risk_band = RiskBand.LOW
        status, reliability, warnings = self._reliability(features)
        return PredictionResult(
            status=status,
            attendance_probability=probability,
            no_show_probability=no_show,
            predicted_attendance=probability >= threshold,
            decision_threshold=threshold,
            no_show_risk_band=risk_band,
            reliability=reliability,
            reason_codes=self._reason_codes(frame, probability) if include_reasons else [],
            warnings=warnings,
            model_version=self.bundle["model_version"],
        )

    def predict(self, attendance_input: AttendanceInput) -> PredictionResult:
        features = attendance_input.feature_dict()
        frame = pd.DataFrame([features], columns=RAW_FEATURE_COLUMNS)
        probability = float(self.model.predict_proba(frame)[:, 1][0])
        return self._build_result(attendance_input, frame, probability, include_reasons=True)

    def score_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing_columns = [column for column in RAW_FEATURE_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        outputs: list[dict[str, Any] | None] = [None] * len(frame)
        valid_inputs: list[AttendanceInput] = []
        valid_positions: list[int] = []
        for input_row, (_, row) in enumerate(frame.iterrows()):
            raw = {column: row.get(column) for column in RAW_FEATURE_COLUMNS}
            raw["student_id"] = None if "student_id" not in frame else str(row.get("student_id"))
            raw = {key: (None if pd.isna(value) else value) for key, value in raw.items()}
            try:
                validated = AttendanceInput.model_validate(raw)
                valid_inputs.append(validated)
                valid_positions.append(input_row)
            except ValidationError as error:
                outputs[input_row] = {
                    "input_row": input_row,
                    "student_id": raw.get("student_id"),
                    "status": PredictionStatus.REJECTED.value,
                    "attendance_probability": np.nan,
                    "no_show_probability": np.nan,
                    "predicted_attendance": None,
                    "decision_threshold": self.bundle["decision_threshold"],
                    "no_show_risk_band": None,
                    "reliability": Reliability.LOW.value,
                    "reason_codes": "",
                    "warnings": "",
                    "model_version": self.bundle["model_version"],
                    "error": error.errors()[0]["msg"],
                }

        if valid_inputs:
            valid_frame = pd.DataFrame(
                [attendance_input.feature_dict() for attendance_input in valid_inputs],
                columns=RAW_FEATURE_COLUMNS,
            )
            probabilities = self.model.predict_proba(valid_frame)[:, 1]
            for local_index, (position, validated, probability) in enumerate(
                zip(valid_positions, valid_inputs, probabilities, strict=True)
            ):
                one_row = valid_frame.iloc[[local_index]].copy()
                result = self._build_result(
                    validated, one_row, float(probability), include_reasons=False
                )
                outputs[position] = {
                    "input_row": position,
                    "student_id": validated.student_id,
                    **result.model_dump(mode="json"),
                    "reason_codes": "",
                    "warnings": "; ".join(result.warnings),
                    "error": "",
                }

        if any(output is None for output in outputs):
            raise RuntimeError("Batch output assembly is incomplete.")
        return pd.DataFrame(outputs).sort_values("input_row").reset_index(drop=True)


def summarize_batch(outputs: pd.DataFrame, model_version: str) -> BatchPredictionSummary:
    """Summarize valid batch probabilities without retaining identities or raw inputs."""
    required = {
        "status",
        "attendance_probability",
        "no_show_probability",
        "no_show_risk_band",
    }
    missing = sorted(required - set(outputs.columns))
    if missing:
        raise ValueError(f"Cannot summarize batch; missing columns: {', '.join(missing)}")

    valid_mask = outputs["status"].isin(
        [PredictionStatus.SCORED.value, PredictionStatus.REVIEW_REQUIRED.value]
    )
    valid = outputs.loc[valid_mask]
    scored_rows = int(outputs["status"].eq(PredictionStatus.SCORED.value).sum())
    review_rows = int(
        outputs["status"].eq(PredictionStatus.REVIEW_REQUIRED.value).sum()
    )
    rejected_rows = int(outputs["status"].eq(PredictionStatus.REJECTED.value).sum())
    return BatchPredictionSummary(
        total_rows=int(len(outputs)),
        valid_rows=int(valid_mask.sum()),
        scored_rows=scored_rows,
        rejected_rows=rejected_rows,
        review_required_rows=review_rows,
        expected_attendees=float(valid["attendance_probability"].sum()),
        expected_no_shows=float(valid["no_show_probability"].sum()),
        high_risk_count=int(valid["no_show_risk_band"].eq(RiskBand.HIGH.value).sum()),
        model_version=model_version,
    )


def serialize_prediction(result: PredictionResult) -> str:
    return json.dumps(result.model_dump(mode="json"), sort_keys=True)
