"""Validated public input and prediction contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictionStatus(str, Enum):
    SCORED = "scored"
    REVIEW_REQUIRED = "review_required"
    REJECTED = "rejected"


class Reliability(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AttendanceInput(BaseModel):
    """One event registration at prediction time."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    student_id: str | None = None
    event_type: str | None = None
    registration_days_before: float | None = None
    previous_events_registered: int | None = Field(default=None, ge=0)
    previous_events_attended: int | None = Field(default=None, ge=0)
    club_member: str | bool | None = None
    event_day: str | None = None
    event_time: str | None = None
    travel_distance_km: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_history(self) -> AttendanceInput:
        if (
            self.previous_events_registered is not None
            and self.previous_events_attended is not None
            and self.previous_events_attended > self.previous_events_registered
        ):
            raise ValueError("Previous events attended cannot exceed previous events registered.")
        return self

    def feature_dict(self) -> dict[str, Any]:
        payload = self.model_dump()
        payload.pop("student_id", None)
        return payload


class PredictionResult(BaseModel):
    """Auditable scoring response returned by every prediction surface."""

    status: PredictionStatus
    attendance_probability: float = Field(ge=0, le=1)
    no_show_probability: float = Field(ge=0, le=1)
    predicted_attendance: bool
    decision_threshold: float = Field(ge=0, le=1)
    no_show_risk_band: RiskBand
    reliability: Reliability
    reason_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model_version: str
