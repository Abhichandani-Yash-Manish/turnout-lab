# Data snapshots

`train.csv` and `test.csv` are immutable snapshots of the official challenge sheets. Their source links, retrieval time, row counts, schemas, and SHA-256 hashes are recorded in `provenance.json`.

The raw files are never edited in place. The modeling pipeline quarantines official-test overlaps before reading targets for model development, then performs every remaining transformation inside the validation workflow.

Refresh the snapshot only when the organizers deliberately update the source sheets:

```bash
python scripts/snapshot_data.py
```

Refreshing changes the recorded hashes and requires rerunning the full audit, evaluation, tests, and prediction export.

## `samples/demo_registrations.csv`

Ten fabricated registrations used to exercise the batch scorer. No real student
data. Each row targets a specific behaviour:

| Row | Demonstrates |
|---|---|
| DEMO-001, DEMO-002 | Confident attendance scores, high reliability |
| DEMO-003, DEMO-004 | High no-show risk — the reminder targets |
| DEMO-005 | One missing field imputed, reliability drops to medium |
| DEMO-006 | Three or more missing fields, flagged for review |
| DEMO-007 | Category never seen in training (`brunch`), flagged for review |
| DEMO-008 | Travel distance far outside the usual range |
| DEMO-009 | Rejected: attended more events than registered |
| DEMO-010 | Rejected: negative travel distance |

`tests/test_prediction.py` asserts this file still produces six scored, two
review-required, and two rejected rows.
