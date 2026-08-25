# Data snapshots

`train.csv` and `test.csv` are immutable snapshots of the official challenge sheets. Their source links, retrieval time, row counts, schemas, and SHA-256 hashes are recorded in `provenance.json`.

The raw files are never edited in place. The modeling pipeline quarantines official-test overlaps before reading targets for model development, then performs every remaining transformation inside the validation workflow.

Refresh the snapshot only when the organizers deliberately update the source sheets:

```bash
python scripts/snapshot_data.py
```

Refreshing changes the recorded hashes and requires rerunning the full audit, evaluation, tests, and prediction export.
