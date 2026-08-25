# Turnout Lab demo script (7–9 minutes)

Use fake identities only. Reset runtime logs before the final recording if you want a clean dashboard.

## 0:00–0:40 — Problem and thesis

“A registration count is not a turnout estimate. Turnout Lab predicts calibrated attendance probability and prioritizes no-show risk, but keeps uncertainty visible and limits the model to supportive planning.”

Show the five app views and state that no LLM or API is used at runtime.

## 0:40–1:45 — The data-integrity discovery

Open **Data & operations** and the audit notebook.

- Training: 508 rows; test: 100.
- Every official test identity and feature fingerprint occurs in training.
- Explain why using those labels would leak the answer.
- Show the 101-row quarantine and 397-row development cohort.
- Mention missing targets, duplicates, inconsistent history, negative lead time, and the retained distance outlier.

Key line: “The most important model decision happened before model training.”

## 1:45–3:20 — Validation, ablation, and decision diagnostics

Open **Model card**.

- Explain connected identity/fingerprint groups.
- Explain five outer folds × five seeds and inner-only tuning, calibration, and threshold selection.
- Compare the baseline, logistic regression, random forest, and gradient boosting.
- Contrast raw random forest ROC-AUC/macro-F1 (`0.638/0.590`) with engineered (`0.628/0.577`): engineered features were tested and rejected on evidence.
- State the calibrated champion results: ROC-AUC 0.635, macro-F1 0.584, Brier 0.221, and 1.52× top-20% no-show lift.
- Show the threshold curve and explain why moving the threshold trades attendance recall against no-show recall.
- Show the normalized confusion matrix and 25-fold stability view. Each development row is evaluated once per seed.
- Call the signal modest; do not oversell it.

## 3:20–4:20 — Single prediction

Open **Predict** and submit a realistic registration.

- Point out probability versus the learned decision threshold.
- Explain low/medium/high no-show risk bands.
- Show reliability separately from probability.
- Read one associated-factor delta and state that it is non-causal.
- Enter an unusual value to demonstrate a review warning if time permits.

## 4:20–5:20 — Batch scoring and expected turnout

Open **Batch score**, load the official test snapshot, and score it.

- Show exactly 100 rows in original order.
- Show scored/review/rejected counts and risk distribution.
- Show expected attendees and expected no-shows as sums of valid probabilities; emphasize that these are planning expectations, not guaranteed counts.
- Download the prediction CSV.
- Mention that invalid rows are isolated rather than crashing the batch.

## 5:20–6:00 — Scenario lab

Open **Scenario lab**, change an organizer-controlled field, and compare.

- Explain that the display asks “what does this fitted model associate with this scenario?”
- Explicitly state that it does not estimate causal intervention effects.

## 6:00–6:55 — Three insights and anonymous operations

Open **Data & operations**.

- Workshops: 75.9% attendance (`n=112`) versus social events: 53.8% (`n=65`).
- Club members: 68.9% (`n=251`) versus non-members: 53.2% (`n=141`).
- Registrations 8–14 days early: 73.4% (`n=188`) versus 1–3 days: 48.6% (`n=72`).
- Point to the bootstrap intervals and state that these are descriptive associations, not causal recommendations.
- Show prediction and batch totals reconciling to SQLite.
- Show risk/reliability charts.
- Explain that student IDs and raw features are not persisted.

## 6:55–8:15 — Code, compliance, and verification

Show:

- `src/turnout_lab/data.py` for quarantine and grouping.
- `src/turnout_lab/modeling.py` for fold-local evaluation/calibration.
- `src/turnout_lab/prediction.py` for the shared prediction contract.
- `tests/` and the passing test command.
- `artifacts/test_predictions.csv` and the executed notebook.
- The README task-compliance matrix and the three explanation guides.

Close with the limitations: small cohort, no event IDs/dates, compromised official split, modest signal, and no demographic fairness analysis.

Final line: “The strongest result is not a perfect score; it is an evaluation process that refuses leakage and makes uncertainty inspectable.”

## Before recording

- Run `uv run pytest` and `uv run ruff check app.py src scripts tests`.
- Start the app with `uv run streamlit run app.py`.
- Use only fake registration details.
- Record at a readable zoom and avoid exposing local usernames unnecessarily.
- Upload to Google Drive, enable link access, and test the link signed out.
- Test the GitHub repository link signed out or add evaluator collaborators.
