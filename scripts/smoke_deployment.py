"""Verify the deployable runtime without development or notebook dependencies."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"
HEALTH_URL = "http://127.0.0.1:8765/_stcore/health"
COLD_START_TIMEOUT_SECONDS = 90
SERVER_TIMEOUT_SECONDS = 60


def verify_app_test() -> None:
    """Execute the Streamlit script and fail on any rendered exception."""
    app = AppTest.from_file(str(APP_PATH)).run(timeout=COLD_START_TIMEOUT_SECONDS)
    if app.exception:
        messages = "; ".join(str(exception.value) for exception in app.exception)
        raise RuntimeError(f"Streamlit AppTest reported an exception: {messages}")
    if len(app.tabs) != 5:
        raise RuntimeError(f"Expected five dashboard tabs, found {len(app.tabs)}")


def verify_server_health() -> None:
    """Launch the production-style server and wait for its health endpoint."""
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.headless=true",
        "--server.port=8765",
        "--browser.gatherUsageStats=false",
    ]

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + SERVER_TIMEOUT_SECONDS
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
                        if response.status == 200 and response.read().strip() == b"ok":
                            return
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.5)

            server_log.seek(0)
            output = server_log.read()
            raise RuntimeError(f"Streamlit health check failed. Server output:\n{output}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def main() -> None:
    verify_app_test()
    verify_server_health()
    print("Deployment smoke check passed: AppTest clean and health endpoint OK.")


if __name__ == "__main__":
    main()
