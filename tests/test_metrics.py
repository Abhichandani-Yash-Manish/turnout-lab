from __future__ import annotations

import numpy as np
import pytest

from turnout_lab.metrics import decision_diagnostics


def test_decision_diagnostics_are_normalized_and_monotonic() -> None:
    targets = np.asarray([0, 0, 0, 1, 1, 1, 0, 1], dtype=int)
    probabilities = np.asarray([0.1, 0.35, 0.55, 0.45, 0.62, 0.9, 0.7, 0.8])
    diagnostics = decision_diagnostics(
        targets,
        probabilities,
        selected_threshold=0.59,
        development_rows=8,
        outer_seeds=[11],
    )

    confusion = np.asarray(diagnostics["normalized_confusion_matrix"])
    assert diagnostics["class_order"] == ["no_show", "attended"]
    assert confusion.shape == (2, 2)
    assert confusion.sum(axis=1) == pytest.approx([1.0, 1.0])

    curve = diagnostics["threshold_curve"]
    assert any(point["threshold"] == pytest.approx(0.59) for point in curve)
    assert all(
        0 <= value <= 1
        for point in curve
        for key, value in point.items()
        if key != "threshold"
    )
    attendance_recall = [point["attendance_recall"] for point in curve]
    assert all(
        current >= following
        for current, following in zip(attendance_recall, attendance_recall[1:], strict=False)
    )
