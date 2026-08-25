from __future__ import annotations

import numpy as np
import pandas as pd

from turnout_lab.features import build_model_frame, normalize_category


def test_category_normalization_is_case_and_whitespace_safe() -> None:
    assert normalize_category(" Workshop ") == "workshop"
    assert normalize_category("YES") == "yes"
    assert normalize_category(True) == "yes"
    assert pd.isna(normalize_category("n/a"))


def test_engineering_preserves_anomalies_as_flags() -> None:
    frame = pd.DataFrame(
        [
            {
                "event_type": "Workshop",
                "registration_days_before": -2,
                "previous_events_registered": 2,
                "previous_events_attended": 3,
                "club_member": "YES",
                "event_day": "Saturday",
                "event_time": "18:00",
                "travel_distance_km": 120,
            }
        ]
    )
    engineered = build_model_frame(frame, "engineered")

    assert engineered.loc[0, "history_inconsistent"] == 1
    assert engineered.loc[0, "late_registration"] == 1
    assert engineered.loc[0, "distance_outlier"] == 1
    assert engineered.loc[0, "registration_days_model"] == 0
    assert pd.isna(engineered.loc[0, "previous_attendance_rate"])
    assert np.isclose(engineered.loc[0, "distance_log"], np.log1p(120))

