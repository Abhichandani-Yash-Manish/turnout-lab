# Model card — Turnout Lab 0.1.0

## Model details

Turnout Lab uses a sigmoid-calibrated random forest trained on eight registration-time fields. Median/mode imputation, categorical encoding, tuning, calibration, and threshold selection are encapsulated in the fitted scikit-learn pipeline. `student_id` is excluded from model features.

- Model family: random forest
- Feature set: normalized raw inputs
- Evaluation rows (leakage-safe cohort): 397
- Connected validation groups: 396
- Shipped-model training rows (refit): 496
- Target: attended (`1`) versus no-show (`0`)
- Classification threshold: 0.590
- Model version: `turnout-lab-0.1.0`

## Intended use

Use the model as a weak planning signal for supportive reminders, volunteer allocation, or capacity estimates. Compare probabilities and risk bands at an aggregate level and review low-reliability rows manually.

Prohibited uses include denying registrations, ranking students, disciplinary action, automated messaging without oversight, or treating a probability as evidence of intent.

## Training and evaluation data

The official training sheet contains 508 rows and the official test sheet contains 100. Every test identity and normalized feature row appears in training, so 101 matching training rows are quarantined before any evaluation. After missing-target removal and deduplication, 397 rows remain, and every metric in this card is measured on that cohort. The shipped model is afterwards refit on all 496 labelled rows; model family, threshold, and risk bands are frozen beforehand, so the refit changes no reported number.

Reported performance uses five outer grouped folds across five fixed seeds. Connected rows sharing an identity or normalized feature fingerprint remain in one group. Inner group-safe splits perform tuning, sigmoid calibration, and classification-threshold selection. Official test labels are not used for model development or evaluation.

## Performance

| Metric | Mean across 25 outer folds | Fold SD |
|---|---:|---:|
| ROC-AUC | 0.635 | 0.050 |
| Macro-F1 | 0.584 | 0.050 |
| Balanced accuracy | 0.591 | 0.046 |
| Attendance precision | 0.705 | 0.047 |
| Attendance recall | 0.706 | 0.096 |
| No-show precision | 0.485 | 0.102 |
| No-show recall | 0.476 | 0.135 |
| Brier score | 0.221 | 0.012 |
| Brier skill vs. baseline | +5.1% | — |
| Top-20% no-show lift | 1.52× | 0.25 |

These results indicate modest discrimination and small calibration improvement over prevalence. They do not justify high-stakes individual decisions.

## Decision diagnostics

- The final attendance threshold is 0.590 and was selected only from leakage-safe development out-of-fold probabilities.
- The threshold curve reports attendance/no-show Precision, Recall, and F1, macro-F1, and predicted-attendance rate from 0.20 to 0.80.
- The normalized confusion matrix uses repeated outer-fold predictions. Each of the 397 development rows is evaluated once per seed, producing 1,985 diagnostic predictions.
- The fold-stability view displays ROC-AUC, macro-F1, and Brier score across all 25 grouped outer folds rather than hiding variation behind one mean.

Headline Precision/Recall/F1 values average the fold-local thresholds selected without each validation fold. The diagnostic confusion matrix instead applies the final 0.590 deployment threshold uniformly to all repeated out-of-fold probabilities, so its class recalls are not expected to equal the headline fold means exactly.

The raw-feature random forest also passed an explicit ablation: ROC-AUC 0.638 and macro-F1 0.590 versus 0.628 and 0.577 for its engineered-feature counterpart. Engineered fields were excluded because they did not improve grouped validation—not because feature engineering was skipped.

## Output interpretation

- **Attendance probability** is the calibrated model estimate.
- **Predicted attendance** applies the learned 0.590 threshold.
- **No-show risk band** is capacity-based: high is the top 20% of development out-of-fold no-show probabilities; medium is the 50th–80th percentile.
- **Reliability** checks the input contract. Missing values, unseen categories, logical inconsistencies, or values outside the development distribution lower reliability independently of the probability.
- **Associated factors** are one-field-at-a-time changes against a reference profile. They are sensitivity checks, not causal explanations.
- **Expected batch turnout** is the sum of valid individual attendance probabilities. Rejected rows are excluded, and the result is a planning expectation rather than a guaranteed count.

## Limitations and risks

- Only 397 leakage-safe labeled rows are available.
- No event ID, organizer, calendar date, capacity, reminder history, or event-level grouping is supplied.
- The official test split cannot estimate independent generalization.
- Missing demographics prevent subgroup fairness assessment; absence of a sensitive column does not guarantee fairness.
- Behavioral patterns may change between clubs, semesters, and event types.
- Scenario estimates are non-causal and should not be used to claim that changing a field will change behavior.
- Inputs outside the observed contract may still receive a score; the app marks these for review.

## Privacy and monitoring

The operational database stores no identity or raw registration fields. Logged data is limited to timestamp, source, model version, probability, risk band, reliability, and warning codes. A real deployment should add outcome monitoring, drift checks, access control, retention limits, and periodic recalibration before operational use.

## Reproducibility

The exact raw-file hashes, evaluation protocol, fold metrics, calibration points, model artifact, and 100-row output are versioned in the repository. Run `uv run turnout-lab train` to rebuild them from the snapshots.
