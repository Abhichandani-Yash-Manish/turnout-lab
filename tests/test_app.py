import re

from streamlit.testing.v1 import AppTest


def test_streamlit_dashboard_starts_without_exceptions() -> None:
    app = AppTest.from_file("app.py").run(timeout=45)
    assert not app.exception
    assert app.title or app.markdown
    assert len(app.tabs) == 5
    assert any("Decision diagnostics" in block.value for block in app.markdown)


def test_official_batch_flow_shows_reconciled_expected_totals() -> None:
    app = AppTest.from_file("app.py").run(timeout=45)
    next(button for button in app.button if button.label == "Use official 100-row test snapshot").click()
    app.run(timeout=45)
    next(button for button in app.button if button.label == "Score loaded rows").click()
    app.run(timeout=45)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Valid registrations"] == "100"
    assert float(metrics["Expected attendees"]) + float(metrics["Expected no-shows"]) == 100
    assert metrics["Rejected"] == "0"
    assert len(app.dataframe) == 1


def test_single_prediction_reports_a_percentage_likelihood() -> None:
    app = AppTest.from_file("app.py").run(timeout=45)
    next(button for button in app.button if button.label == "Estimate turnout").click()
    app.run(timeout=45)

    assert not app.exception
    ticket = next(block.value for block in app.markdown if '<div class="ticket">' in block.value)
    assert "likely to attend" in ticket
    percentage = re.search(r'<div class="probability">(\d+)%</div>', ticket)
    assert percentage is not None
    assert 0 <= int(percentage.group(1)) <= 100
