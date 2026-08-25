# Deployment and Release Runbook

This runbook makes the hosted demo reproducible without changing the modeling protocol. The deployment target is Streamlit Community Cloud because Turnout Lab is a self-contained Streamlit application with no API keys, external database, or hosted-model dependency.

## Why the existing repository is the release repository

`Abhichandani-Yash-Manish/turnout-lab` was created specifically for Task 1 and is already separate from the earlier ClubAtlas work. Reusing it preserves one auditable Git history, one CI history, and one evaluator link. A duplicate repository would add release drift without improving isolation.

## Deployment contract

| Item | Locked value |
|---|---|
| GitHub repository | `Abhichandani-Yash-Manish/turnout-lab` |
| Release branch | `main` |
| App entry point | `app.py` |
| Python version | `3.12` |
| Dependency source | root `uv.lock` |
| OS packages | none |
| Secrets | none |
| Runtime model | `artifacts/model.joblib` |

Streamlit Community Cloud searches the app directory and repository root for dependency files. Because `uv.lock` has the highest recognized priority and is committed at the root, it is the only deployment dependency source. The default dependency group contains only runtime packages; the `dev` and `notebook` extras are excluded from deployment.

## Pre-deployment gate

Run from the repository root:

```bash
git status --short
uv sync --locked
uv run python scripts/smoke_deployment.py
uv sync --locked --extra dev --extra notebook
uv run ruff check app.py src scripts tests
uv run pytest
uv run jupyter nbconvert --execute --to notebook --inplace \
  notebooks/01_data_audit_and_model_selection.ipynb
```

Pass conditions:

- The worktree is clean at the release commit.
- Runtime-only AppTest has no exception and exposes all five tabs.
- The Streamlit health endpoint returns `ok`.
- A first-run dependency/model cold start completes within the 90-second smoke-test budget.
- Lint and all automated tests pass.
- The notebook executes without an error.
- `artifacts/test_predictions.csv` still contains exactly 100 ordered, unique IDs.
- The latest GitHub Actions run is green for both `test` and `deployment-smoke`.

## Deploy from a private repository

1. Sign in at [share.streamlit.io](https://share.streamlit.io) using the GitHub account that administers the repository.
2. Connect GitHub and authorize private-repository access if the repository is not listed.
3. Select **Create app** and then **Yup, I have an app**.
4. Enter the repository, branch, entry point, and Python version from the deployment contract above.
5. Leave secrets empty.
6. Choose an available URL such as `turnout-lab` or `turnout-lab-gdg`.
7. Deploy and inspect the build log until the app is healthy.

The private repository can be used during rehearsal. A private-source app is private initially; its viewing policy can be changed from Streamlit app settings. Community Cloud requires repository-admin permission to deploy.

## Hosted smoke journey

Run these checks in the deployed app:

1. **Predict:** score the default example and verify an attendance probability, no-show probability, risk band, reliability, associated factors, and model version appear.
2. **Batch score:** load the official 100-row snapshot, score it, and verify valid registrations equal 100, rejected equals 0, and expected attendees plus expected no-shows equals 100.
3. **Scenario lab:** change one organizer-controlled field and verify the comparison is labelled non-causal.
4. **Model card:** verify champion metrics, calibration, normalized confusion matrix, 25-fold stability, threshold curve, and ablation are visible.
5. **Data & operations:** verify the overlap audit and anonymous log counters render without exposing student IDs or raw inputs.
6. Refresh the app and repeat one prediction to catch path or write-permission failures.

Runtime SQLite data on Community Cloud is demonstration state. Files generated during use are not guaranteed to persist across sessions or restarts, so the dashboard must not be presented as durable production monitoring.

## Submission release gate

1. Merge only a commit whose GitHub Actions checks are green.
2. Make the GitHub repository public immediately before submission, unless evaluator accounts have explicitly been granted access.
3. Make the Streamlit app public in its sharing settings.
4. Open the repository URL, deployed app URL, and Google Drive video URL in a signed-out/private browser.
5. Confirm the README clone commands work from that public view.
6. Confirm the demo video has link viewing enabled and does not expose local paths, tokens, or personal test data.
7. Add the final deployed-app and video URLs to the README only after both are verified.

## Rollback

If a release commit breaks the hosted app, do not delete model or evaluation history. Revert the faulty commit through Git, push the revert, and let Community Cloud rebuild from the restored `main`. Keep the last known-green commit hash in the submission checklist until final release.

## Official platform references

- [Deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [App dependencies and dependency-file priority](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Connect GitHub and authorize private repositories](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account)
- [Share a public or private app](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)
- [Community Cloud status and limitations](https://docs.streamlit.io/deploy/streamlit-community-cloud/status)
