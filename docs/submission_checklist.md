# Turnout Lab — submission checklist

This is the final operational checklist. Do not submit until every evaluator-access item is checked.

## Official Task 1 requirements

- [x] Missing and inconsistent values are audited and handled.
- [x] Categorical fields are converted through fold-local one-hot encoding.
- [x] Classification models are trained and compared.
- [x] Precision, Recall, and F1-score are reported for both classes.
- [x] New registrations can be scored through form, CSV, and CLI.
- [x] Output includes percentage likelihood of attendance.
- [x] Three useful club insights include sample size and uncertainty.
- [x] Official test output contains exactly 100 ordered unique rows.
- [x] The absent technical/non-technical field is documented rather than invented.

## Submission requirements

- [x] Complete code is pushed to GitHub.
- [x] README explains the approach, setup, evaluation, and limitations.
- [x] Streamlit app demonstrates single and batch prediction.
- [x] Repository is organized and runnable from pinned dependencies.
- [x] Model artifact and generated evaluation outputs are committed.
- [x] Demo script covers approach, model, results, application, and limitations.
- [ ] Final demo video is recorded using fake identities only.
- [ ] Video is uploaded to Google Drive.
- [ ] Google Drive sharing is set to “Anyone with the link can view.”
- [ ] Final Drive link replaces the README placeholder.
- [ ] Repository is made public or evaluators are added as collaborators.

## Required artifacts

- [x] `artifacts/model.joblib`
- [x] `artifacts/metrics.json`
- [x] `artifacts/test_predictions.csv`
- [x] `artifacts/data_quality_report.json`
- [x] `artifacts/feature_contract.json`
- [x] Executed audit/model-selection notebook
- [x] Model card
- [x] Data-quality report
- [x] Project walkthrough
- [x] Viva questions
- [x] Demo script

## Local verification

Run from the repository root:

```bash
uv sync --locked
uv run python scripts/smoke_deployment.py
uv sync --locked --extra dev --extra notebook
uv run ruff check app.py src scripts tests
uv run pytest
uv run python scripts/build_notebook.py
uv run jupyter nbconvert --execute --to notebook --inplace \
  notebooks/01_data_audit_and_model_selection.ipynb
uv run streamlit run app.py
```

Confirm manually:

- [ ] Predict shows probability, class, risk, reliability, factors, and warnings.
- [ ] Official Batch Score produces 100 rows and aggregate expected turnout.
- [ ] Invalid batch rows are isolated and excluded from expected totals.
- [ ] Scenario Lab is clearly labelled non-causal.
- [ ] Model Card renders comparison, threshold, confusion, stability, calibration, and importance.
- [ ] Data & Operations shows audit evidence, insights, and anonymous log totals.
- [ ] No browser or server-console errors occur.
- [ ] Streamlit deployment uses `main`, `app.py`, Python 3.12, and no secrets.
- [ ] Hosted Predict, Batch score, Scenario lab, Model card, and Data & operations journeys pass.
- [ ] Deployed app is public or evaluator viewer access is confirmed.

## Demo sequence

- [ ] Problem and operational use.
- [ ] 100/100 official overlap and 101-row quarantine.
- [ ] Cleaning and fold-local preprocessing.
- [ ] Connected grouped nested validation.
- [ ] Candidate comparison and raw-versus-engineered ablation.
- [ ] Final metrics, calibration, threshold trade-off, confusion, and stability.
- [ ] Single prediction and reliability warning.
- [ ] Batch expected turnout and CSV download.
- [ ] Three descriptive insights.
- [ ] Privacy, limitations, and future work.
- [ ] Total duration remains 7–9 minutes.

## Release and access

1. Keep the repository private while recording.
2. Replace the README video placeholder with the final Drive URL.
3. Commit and push the link update.
4. Make the repository public, or add every evaluator as a collaborator.
5. Open an incognito/signed-out browser.
6. Verify the GitHub repository opens and the README screenshot renders.
7. Verify the Drive video plays without requesting access.
8. Copy the exact final links into the submission form.
9. Reopen both submitted links from the form confirmation if available.

The detailed deployment, hosted smoke, access, and rollback procedure is in [`docs/deployment.md`](deployment.md).

## Final teach-back gate

Without reading, be able to explain:

- why the official test set cannot measure accuracy;
- how the 397-row cohort was formed;
- why grouped nested validation was used;
- why raw random forest won;
- what ROC-AUC, macro-F1, Brier score, and the 0.59 threshold mean;
- why probabilities, risk bands, and reliability differ;
- the three club insights and why they are non-causal;
- the model’s intended use, privacy boundary, and largest limitation.
