from streamlit.testing.v1 import AppTest


def test_streamlit_dashboard_starts_without_exceptions() -> None:
    app = AppTest.from_file("app.py").run(timeout=45)
    assert not app.exception
    assert app.title or app.markdown
    assert len(app.tabs) == 5

