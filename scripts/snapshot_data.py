"""Download immutable challenge data snapshots and record their provenance."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

SOURCES = {
    "train.csv": {
        "document_url": "https://docs.google.com/spreadsheets/d/1-kJpEan0ubFgaT0XIqC6GZdevqOnx7qmdsP1bBuEH5g/edit",
        "export_url": "https://docs.google.com/spreadsheets/d/1-kJpEan0ubFgaT0XIqC6GZdevqOnx7qmdsP1bBuEH5g/export?format=csv",
    },
    "test.csv": {
        "document_url": "https://docs.google.com/spreadsheets/d/1eKM8R6Ew2woVv6cm9AZFbnbYrDTbrlIBmkLuotl8iKQ/edit",
        "export_url": "https://docs.google.com/spreadsheets/d/1eKM8R6Ew2woVv6cm9AZFbnbYrDTbrlIBmkLuotl8iKQ/export?format=csv",
    },
}


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Turnout-Lab/0.1 reproducibility snapshot"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed official URLs
        return response.read()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    provenance: dict[str, object] = {
        "retrieved_at": retrieved_at,
        "task_document": "https://docs.google.com/document/d/1J5j4YsN6r2fVGQ8Y-3snOFqtqmKH9paFI_6s8k2siIM/edit",
        "files": {},
    }

    for filename, source in SOURCES.items():
        payload = download(source["export_url"])
        path = RAW_DIR / filename
        path.write_bytes(payload)
        decoded = payload.decode("utf-8-sig").splitlines()
        rows = list(csv.reader(decoded))
        provenance["files"][filename] = {
            "document_url": source["document_url"],
            "export_url": source["export_url"],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows_excluding_header": max(len(rows) - 1, 0),
            "columns": rows[0] if rows else [],
        }

    (RAW_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
