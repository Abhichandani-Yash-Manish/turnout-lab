# Data-quality and leakage report

Generated from the versioned challenge snapshots retrieved on 25 August 2026.

## Decision gate

**Status: suitable for exploratory model development only after leakage quarantine.** The official test set is unsuitable for independent accuracy measurement because all 100 test identities and normalized feature rows are present in training.

| Severity | Finding | Evidence | Resolution |
|---|---|---:|---|
| Critical | Official train/test overlap | 100/100 test IDs; 100/100 feature fingerprints | Quarantine 101 matching train rows and their labels before modeling |
| High | Missing labels | 5 raw rows | Remove the 4 remaining after quarantine |
| Medium | Exact duplicates | 7 raw rows | Remove the 6 remaining after quarantine |
| Medium | Missing predictors | 8–15 train rows depending on column | Fold-local median/mode imputation with indicators |
| Medium | Impossible histories | 3 rows | Preserve raw fields; invalidate derived rate and flag |
| Medium | Negative lead times | 2 rows | Preserve raw field; clip only a derived modeling value and flag |
| Low | Category casing variants | multiple values | Normalize whitespace and casing in preprocessing |
| Low | Long-distance value | maximum 120 km; raw p99 25.735 km | Retain as plausible, transform robustly if engineered, and flag |

## Cohort accounting

```text
508 raw training rows
- 101 official-test-linked rows quarantined
-   4 remaining missing-target rows
-   6 remaining exact duplicates
= 397 leakage-safe development rows
```

The cohort has 396 connected groups, 252 attended registrations (63.5%), and 145 no-shows (36.5%).

## Controls

- Raw snapshots are never edited in place.
- Source URLs, retrieval timestamp, schemas, row counts, and SHA-256 hashes are stored in `data/raw/provenance.json`.
- Quarantine is calculated from student identity **or** normalized feature fingerprint before targets are selected.
- Linked identity/fingerprint rows are kept in one connected validation group.
- Preprocessing is fitted inside training folds only.
- The official test rows are used solely to generate the required 100-row prediction artifact.

## Residual limitations

The collection procedure is undocumented. There are no event identifiers or timestamps, so same-event correlation, temporal drift, and event-level generalization cannot be measured. There are no demographic fields, so subgroup fairness cannot be evaluated. Data fitness is therefore conditional: the project demonstrates an honest modeling workflow, not production readiness.

Machine-readable evidence is in `artifacts/data_quality_report.json`; executable checks are in `notebooks/01_data_audit_and_model_selection.ipynb` and `tests/test_data.py`.
