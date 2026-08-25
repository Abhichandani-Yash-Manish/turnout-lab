# Turnout Lab

Leakage-aware event attendance forecasting with calibrated probabilities, no-show prioritization, reproducible evaluation, and an interactive decision dashboard.

![Turnout Lab prediction dashboard](docs/app-prediction.png)

## Why this project exists

Event organizers often need a better estimate of turnout than a raw registration count. Turnout Lab converts registration-time information into an attendance probability and a capacity-based no-show risk band. The result is designed for supportive actions—such as reminders and capacity planning—not for rejecting registrations or penalizing students.

The main technical finding was not a model result. It was a data-integrity problem: **all 100 official test rows also appear in the training data by both student ID and normalized feature fingerprint**. Those matching training rows and their targets are quarantined before any *evaluation*, so every metric reported here comes from grouped out-of-fold scoring on the remaining development cohort. The shipped model is then refit on all labelled rows for prediction only — see [Two questions, two models](#two-questions-two-models).

## What it includes

- A reproducible data audit with source snapshots, URLs, timestamps, row counts, and SHA-256 hashes.
- Exact leakage quarantine before evaluation, with a documented refit-before-predict step.
- Fold-local imputation, categorical encoding, feature engineering, calibration, tuning, and threshold selection.
- Comparison of a prevalence baseline, logistic regression, random forest, and gradient boosting on raw and engineered features.
- Calibrated attendance and no-show probabilities, reliability warnings, and capacity-based risk bands.
- Single and batch scoring, non-causal scenario comparison, a model card, and operational analytics.
- Anonymous SQLite logs that never store student IDs or raw registration features.
- An executed audit notebook and automated data, model, prediction, database, and Streamlit tests.

## Official task compliance

| Task requirement | Turnout Lab evidence |
|---|---|
| Clean the dataset and handle missing or inconsistent values | Versioned audit, documented treatments, reusable preprocessing pipeline, executed notebook |
| Perform preprocessing and convert categorical data into a suitable format | Fold-local imputation and one-hot encoding with unknown-category handling |
| Train a classification model to predict whether a student will attend | Prevalence baseline, logistic regression, random forest, and gradient boosting across raw/engineered representations |
| Evaluate using Precision, Recall, and F1-score | Both classes plus macro-F1, balanced accuracy, ranking, calibration, and stability metrics |
| Use the trained model to predict attendance for new registrations | Single form, CSV batch scorer, and CLI |
| Output `Student A → 87% likely to attend` | Sigmoid-calibrated percentage on the Predict view and in every batch result row |
| *(Optional)* Identify 2–3 useful insights | Three sample-sized findings with bootstrap 95% intervals |

### A note on the "technical or non-technical" field

The task description lists this as a dataset column, but **neither supplied sheet contains it**. The closest field is `event_type`, whose five values are `Workshop`, `Talk`, `Hackathon`, `Competition`, and `Social`. Any technical/non-technical split would be an assumption layered on top of those categories, so Turnout Lab models `event_type` directly and reports this absence rather than inventing an unsupported mapping. The distinction is still available to organizers through the per-category attendance table in the Data & operations view.

## Architecture

```mermaid
flowchart LR
    A[Official train + test snapshots] --> B[Schema and quality audit]
    B --> C{ID or feature match<br/>with official test?}
    C -->|Yes| Q[Quarantine from evaluation]
    C -->|No| D[397-row development cohort]
    D --> E[Connected identity/fingerprint groups]
    E --> F[Nested StratifiedGroupKFold]
    F --> G[Fold-local preprocessing, tuning, calibration, threshold]
    G --> H[Frozen model, threshold, risk bands]
    Q -.rejoins for prediction only.-> R[Refit on 496 labelled rows]
    H --> R
    R --> I[CLI and Streamlit predictor]
    I --> J[(Anonymous SQLite operations log)]
    I --> K[100-row official prediction export]
```

The same serialized pipeline powers evaluation, CLI prediction, and Streamlit. `student_id` is retained only for row mapping and is never passed to the model.

## Data-quality audit

| Finding | Evidence | Treatment |
|---|---:|---|
| Official split overlap | 100/100 test IDs and feature rows occur in training | Quarantine 101 matching training rows before target use |
| Missing training targets | 5 raw; 4 outside quarantine | Remove from the development cohort |
| Exact duplicate rows | 7 raw; 6 outside quarantine | Deduplicate development rows |
| Impossible attendance history | 3 rows | Preserve source, invalidate derived rate, emit anomaly flag |
| Negative registration lead time | 2 rows | Preserve source, clip only derived field, emit anomaly flag |
| Distance outlier | maximum 120 km | Preserve, robustly transform when engineered, flag as unusual |
| Category casing variants | `YES`, `Yes`, `yes`, etc. | Normalize whitespace and casing inside the pipeline |

### Why the reported numbers look low

A leakage-unaware submission would train on all 507 rows and score against the official test split. Reproduce what that reports:

```bash
uv run python scripts/leakage_demo.py
```

| Pipeline | Evaluated on | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Leakage-unaware random forest | Official test split | **1.000** | **1.000** |
| Turnout Lab | Grouped out-of-fold, leakage-safe cohort | 0.621 | 0.584 |

The perfect score is the symptom, not the achievement — 99 of the 100 test labels are directly recoverable from the training file, so the model is reciting rows it was shown. **The lower number is the honest one.** `tests/test_data.py` asserts this memorization still reproduces, so the claim cannot quietly rot.

### Two questions, two models

"How good is this model?" and "what are your predictions for these 100 rows?" are different questions, and conflating them is how leakage does its damage. Turnout Lab answers them separately:

| | Trained on | Used for |
|---|---:|---|
| **Evaluation** | 397-row leakage-safe cohort | Every metric reported anywhere in this repository |
| **Shipped model** | 496 labelled rows | Scoring new registrations and the 100 official predictions |

Model family, hyperparameters, the 0.59 decision threshold, and the risk-band cutoffs are all selected and frozen on the leakage-safe cohort *before* the refit. The refit only widens the training set — the ordinary refit-before-predict step — and it never touches a reported number. `tests/test_prediction.py` asserts the shipped model is the wider one and that the metrics still describe the narrower cohort, so the two can never silently merge.

Quarantining is what makes the *measurement* honest. It was never a reason to throw away information at *prediction* time.

The final leakage-safe cohort contains **397 rows**, **396 connected groups**, 252 attendances, and 145 no-shows. Full evidence is available in [the data-quality report](docs/data_quality_report.md), [the machine-readable audit](artifacts/data_quality_report.json), and [the executed notebook](notebooks/01_data_audit_and_model_selection.ipynb).

## Modeling and validation

Four model families were evaluated across raw and engineered feature sets. Candidate selection used five-fold `StratifiedGroupKFold` over five fixed seeds. Inner group-safe folds handled hyperparameter selection and macro-F1 threshold selection. The champion was then evaluated with sigmoid calibration fitted only on training-side group splits.

Rows sharing a student ID or normalized feature fingerprint are assigned to the same connected group. This prevents duplicate or linked registrations from crossing validation boundaries. Every reported score below is generated from the 25 outer validation folds; the official test labels are never used.

### Candidate comparison

| Candidate | ROC-AUC | Macro-F1 | Brier ↓ |
|---|---:|---:|---:|
| Dummy · raw | 0.500 | 0.375 | 0.233 |
| Logistic regression · raw | 0.587 | 0.557 | 0.232 |
| Logistic regression · engineered | 0.580 | 0.558 | 0.246 |
| **Random forest · raw** | **0.638** | **0.590** | 0.227 |
| Random forest · engineered | 0.628 | 0.577 | 0.231 |
| Gradient boosting · raw | 0.607 | 0.559 | **0.227** |
| Gradient boosting · engineered | 0.601 | 0.568 | 0.227 |

The locked selection rule chose the raw-feature random forest. Its final calibrated repeated-CV results are:

| Metric | Mean | Fold SD |
|---|---:|---:|
| ROC-AUC | **0.635** | 0.050 |
| PR-AUC (attendance) | 0.748 | 0.054 |
| Macro-F1 | 0.584 | 0.050 |
| Balanced accuracy | 0.591 | 0.046 |
| Accuracy | 0.621 | 0.040 |
| Attendance precision / recall | 0.705 / 0.706 | 0.047 / 0.096 |
| No-show precision / recall / F1 | 0.485 / 0.476 / 0.468 | 0.102 / 0.135 / 0.096 |
| Brier score | **0.221** | 0.012 |
| Brier skill vs. prevalence baseline | **+5.1%** | — |
| Log loss | 0.634 | 0.027 |
| Top-20% no-show precision | 0.558 | 0.121 |
| Top-20% no-show recall | 0.306 | 0.051 |
| Top-20% no-show lift | **1.52×** | 0.25 |

This is modest predictive signal, not a high-certainty classifier. That limitation is deliberately visible in the product and [model card](docs/model_card.md).

## Three club-planning insights

These are descriptive associations from the leakage-safe cohort, not causal effects.

1. **Workshops recorded 75.9% attendance** (`n=112`, bootstrap 95% interval 67.9%–83.0%) versus **53.8% for social events** (`n=65`, 41.5%–64.6%). Organizers can study which workshop characteristics transfer to other formats.
2. **Club members recorded 68.9% attendance** (`n=251`, 62.9%–74.5%) versus **53.2% for non-members** (`n=141`, 45.4%–61.7%). Clearer onboarding and reminders for non-members are reasonable experiments.
3. **Registrations made 8–14 days early recorded 73.4% attendance** (`n=188`, 66.5%–79.8%) versus **48.6% for 1–3 days early** (`n=72`, 37.5%–59.7%). Clubs can prospectively test earlier promotion windows.

Tiny groups are not converted into recommendations even when their observed percentages look extreme. The full segment table and uncertainty are generated in the dashboard and metrics artifact.

## Product views

1. **Predict** — score one registration with probability, decision threshold, risk band, reliability, warnings, and one-field-at-a-time scenario deltas.
2. **Batch score** — validate a CSV, preserve row order, isolate rejected rows, and download predictions. Upload [`data/samples/demo_registrations.csv`](data/samples/demo_registrations.csv) to see all four outcomes at once: confident scores, high no-show risk, imputed and out-of-range inputs flagged for review, and two rows rejected for impossible history and a negative distance.
3. **Scenario lab** — compare organizer-controlled inputs while clearly labelling estimates as non-causal.
4. **Model card** — inspect generated evaluation, calibration, candidate comparison, feature importance, and limitations.
5. **Data & operations** — inspect audit findings and reconcile anonymous prediction/batch logs.

Risk bands come from leakage-safe out-of-fold no-show probabilities: high is the top 20%, medium is the 50th–80th percentile, and low is below the median. Probability and reliability are separate: missing, unseen, inconsistent, or out-of-distribution inputs lower reliability even when a probability can be computed.

## Run locally

Python 3.12 is the verified environment; package metadata supports Python 3.10–3.13.

### With `uv` (recommended)

```bash
git clone https://github.com/Abhichandani-Yash-Manish/turnout-lab.git
cd turnout-lab
uv sync --locked --extra dev --extra notebook
uv run streamlit run app.py
```

### With `venv` and `pip`

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,notebook]"
streamlit run app.py
```

No API key, hosted model, or external runtime service is required.

## Deploy on Streamlit Community Cloud

The repository is deployment-ready for Streamlit Community Cloud, and deliberately carries two dependency sources so it installs correctly whichever one the platform detects:

- `uv.lock` — the authoritative lockfile. `uv sync` also installs this repository as a package.
- `requirements.txt` — the same pinned runtime set, for platforms that only detect pip.

Notebook and development tools are optional extras, so a hosted install pulls runtime packages only. `app.py` adds `src/` to the import path at startup, so it runs whether or not the platform installed the repository as a package. Both paths are verified: a clean `uv sync` clone and a `pip install -r requirements.txt` environment with no package install both render all five views.

Use these locked settings at [share.streamlit.io](https://share.streamlit.io):

| Setting | Value |
|---|---|
| Repository | `Abhichandani-Yash-Manish/turnout-lab` |
| Branch | `main` |
| Entry point | `app.py` |
| Python | `3.12` |
| Secrets | None |

The repository may stay private while testing if Streamlit is authorized to access private repositories. Before evaluator submission, make both the GitHub repository and the Streamlit app evaluator-accessible and verify both links in a signed-out browser.

Run the same runtime-only deployment check locally or in CI:

```bash
uv sync --locked
uv run python scripts/smoke_deployment.py
```

The Community Cloud filesystem is suitable for this demonstration, but runtime SQLite logs are not promised to survive app restarts. They remain intentionally anonymous and are not part of model inference. See [the deployment and release runbook](docs/deployment.md) for the exact procedure and rollback checks.

## Reproduce the audit and model

```bash
# Verify data quality and rebuild the audit artifacts
uv run turnout-lab audit

# Run the full nested evaluation, train the champion, and export 100 predictions
uv run turnout-lab train

# Rebuild and execute the reader-facing notebook
uv run python scripts/build_notebook.py
uv run jupyter nbconvert --execute --to notebook --inplace \
  notebooks/01_data_audit_and_model_selection.ipynb

# Score another contract-compatible CSV
uv run turnout-lab predict --input registrations.csv --output predictions.csv
```

The full training run is intentionally slower because it evaluates seven candidate/feature combinations over repeated nested grouped folds.

## Test and verify

```bash
uv run ruff check app.py src scripts tests
uv run pytest
```

Tests enforce the critical promises: exact overlap detection, zero test-linked rows in development, no group overlap between folds, absence of student ID from model features, bounded/complementary probabilities, 100 ordered official predictions, anomaly handling, anonymous database persistence, and Streamlit startup/workflow smoke checks.

## Repository map

```text
app.py                         Streamlit decision dashboard
src/turnout_lab/               Reusable audit, features, modeling, prediction, DB code
data/raw/                      Versioned source snapshots and provenance
artifacts/                     Model, metrics, feature contract, audit, predictions
notebooks/                     Executed inspection notebook
tests/                         Deterministic automated checks
docs/                          Model card, data-quality report, walkthrough, deployment runbook
scripts/                       Snapshot and notebook builders
runtime/                       Ignored local SQLite state
```

## Intended use and limitations

Turnout Lab is a planning and supportive-outreach prototype. It must not be used to deny access, rank students, impose penalties, or infer motivation. The small cohort, modest discrimination, unknown collection process, missing event identifiers/dates, absent demographic variables, and compromised official split limit generalization and fairness claims. Scenario changes show model associations, not intervention effects.

The source sheets are a challenge snapshot retrieved on 25 August 2026. Their URLs and exact hashes are recorded in [`data/raw/provenance.json`](data/raw/provenance.json).

## Privacy

The app stores only timestamp, source, model version, probability, risk band, reliability, and warning codes. It does not persist student IDs or raw registration values. No LLM or external AI API is used at runtime; predictions are deterministic classical-ML outputs from the saved pipeline.

## Demo

**Demo video:** add the evaluator-accessible Google Drive link before submission.

For the full decision chain and short/long verbal explanations, see the [project walkthrough](docs/project_walkthrough.md).

---

<sub>Built by Yash Abhichandani. AI coding assistance was used during development and documentation; the data-leakage finding, modeling protocol, metric interpretation, and final product behavior were authored and verified by me.</sub>
