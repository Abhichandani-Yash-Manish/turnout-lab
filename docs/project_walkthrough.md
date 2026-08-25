# Turnout Lab — project walkthrough

This is the learning-oriented explanation of the project. Read it once from top to bottom, then use the shorter speaking versions at the end for rehearsal.

## 1. Problem framing

The task is binary classification: given information available when a student registers for a club event, estimate whether the student will attend.

- Target: `attended`, where `1` means attended and `0` means no-show.
- Output: calibrated attendance probability plus a threshold-based class.
- Operational use: reminder prioritization and aggregate turnout planning.
- Prohibited use: rejecting registrations or penalizing students.

The model uses event type, registration lead time, previous registrations and attendances, club membership, event day, event time, and travel distance. `student_id` is retained only to map predictions back to rows; it is never a model feature.

## 2. What the raw data revealed

The supplied training file has 508 rows and the test file has 100. The audit found:

- five missing training targets;
- seven exact duplicate training rows;
- missing predictor values;
- inconsistent category casing;
- three histories where attended events exceed registered events;
- two negative registration lead times;
- a 120 km distance value;
- every official test identity and normalized feature row also present in training.

The final point is the most important. Looking up the matching training labels would produce impressive-looking test predictions, but it would be target leakage rather than generalization.

## 3. Leakage quarantine

Before selecting targets for modeling, the pipeline normalizes the feature representation and constructs a fingerprint. A training row is quarantined when either:

1. its student ID appears in the official test file; or
2. its normalized feature fingerprint appears in the official test file.

This quarantines 101 training rows. Their labels are never used for tuning, threshold selection, or any reported evaluation. Once those choices are frozen, the selected pipeline is refit on all 496 labelled rows for final scoring — the usual refit-before-predict step, which widens the training data without touching a single reported metric.

After quarantine, four remaining missing-target rows and six remaining exact duplicates are removed. The leakage-safe cohort contains 397 rows and 396 connected groups.

### Why not evaluate on the official test set?

An evaluation set must represent unseen outcomes. Because every test row is already represented in training, it cannot estimate independent generalization. The official test file is used only to produce the required 100 predictions.

## 4. Cleaning decisions

| Issue | Decision | Reason |
|---|---|---|
| Missing target | Remove row | Supervised training requires a known outcome |
| Exact duplicate | Keep one development copy | Prevent repeated evidence from receiving extra weight |
| Category casing/space | Normalize inside the pipeline | Treat `YES`, `Yes`, and `yes` consistently |
| Missing numeric value | Fold-local median imputation | Robust and prevents validation leakage |
| Missing category | Fold-local most-frequent imputation | Simple treatment for a small dataset |
| Unseen category | One-hot encoder ignores unknown value and reliability falls | Allows safe scoring without pretending the input is familiar |
| Attended > registered | Preserve raw source; invalidate derived rate; flag | Avoid silently rewriting source data |
| Negative lead time | Preserve raw source; clip only derived value; flag | Separate source truth from modeling convenience |
| 120 km distance | Retain and flag | It is unusual but not demonstrably impossible |

Imputation and encoding are fitted inside training folds. Fitting them once on all development data would let validation distributions influence training.

## 5. Raw and engineered features

Two feature representations were evaluated.

- Raw: the eight supplied predictors after normalization and fold-local preprocessing.
- Engineered: previous attendance rate, previous no-show count, history availability, weekend flag, event hour, log distance, late-registration flag, and anomaly flags.

Feature engineering was treated as a hypothesis, not an automatic improvement. The raw random forest achieved ROC-AUC 0.638 and macro-F1 0.590; the engineered version achieved 0.628 and 0.577. The raw representation was therefore retained.

## 6. Why grouped nested validation?

A normal random split could place the same student or a duplicate feature row on both sides. Turnout Lab creates connected groups: rows connected by student ID or normalized feature fingerprint receive the same group.

Validation then uses:

- five outer `StratifiedGroupKFold` folds;
- five fixed random seeds, producing 25 outer results;
- four group-safe inner folds for hyperparameters;
- inner-only probability calibration and threshold selection.

The outer folds estimate generalization. The inner folds make modeling decisions without consulting the outer validation outcomes. Repeating the outer split across seeds shows stability instead of relying on one fortunate split.

## 7. Candidate models

The locked candidates were:

1. prevalence `DummyClassifier`;
2. regularized logistic regression;
3. balanced random forest;
4. gradient boosting.

Each non-dummy family was evaluated with raw and engineered features. The selection rule required positive discrimination, Brier score no worse than baseline, and macro-F1 within 0.03 of the best eligible candidate. Near-tied ROC-AUC candidates were resolved by Brier score and then simplicity.

The raw random forest won because it had the strongest eligible ROC-AUC and macro-F1 while retaining acceptable calibration.

## 8. Why no SMOTE or larger model?

The development cohort is 63.5% attendance and 36.5% no-show. This is imbalanced but not so extreme that synthetic examples are automatically justified. SMOTE would also have to synthesize mixed categorical and numeric registration profiles, which can create unrealistic combinations.

Instead, the comparison includes class weighting, class-specific metrics, balanced accuracy, macro-F1, no-show metrics, and threshold selection. Adding XGBoost, CatBoost, or a neural network would expand the search space on only 397 rows without fixing the data limitations.

## 9. Calibration and threshold

### Calibration

A rank can be useful while its probabilities are poor. Sigmoid calibration maps raw model scores to probabilities that better correspond to observed attendance rates. Brier score measures the squared error of those probabilities.

The calibrated model achieved:

- ROC-AUC: 0.635 ± 0.050;
- macro-F1: 0.584 ± 0.050;
- Brier score: 0.221 ± 0.012;
- Brier skill versus prevalence: +5.1%;
- top-20% no-show lift: 1.52×.

This is modest signal, not certainty.

### Classification threshold

The final attendance threshold is 0.59, chosen on development-only out-of-fold probabilities to maximize macro-F1. Increasing the threshold makes the system more reluctant to predict attendance: attendance recall generally falls while no-show recall generally rises.

The dashboard shows this trade-off instead of presenting 0.59 as a universal truth.

The headline Precision/Recall/F1 numbers average policies selected independently inside each outer fold. The normalized diagnostic confusion matrix reapplies the final 0.59 policy to all repeated out-of-fold predictions. Those two valid summaries answer different questions, so their class recalls differ slightly.

## 10. Probability, risk, and reliability

These are deliberately separate:

- Probability: the calibrated estimate of attendance.
- Predicted class: whether probability clears the 0.59 decision threshold.
- No-show risk band: relative capacity tier based on development out-of-fold no-show probabilities.
- Reliability: input-contract quality, lowered by missing, unseen, inconsistent, or out-of-distribution values.

A registration can have a high predicted probability but low reliability if its inputs are unfamiliar. Reliability is not the probability that the prediction is correct.

## 11. Explanations and scenarios

Individual factors are one-field-at-a-time sensitivity comparisons against a documented reference profile. They say how the fitted estimate changes when one input is replaced while other inputs stay fixed.

They are labelled “factors associated with this score” because they are not causal explanations. The Scenario Lab follows the same rule: it helps form hypotheses, not promises that changing an event field will change attendance.

## 12. Three useful club insights

The descriptive analysis includes sample sizes and bootstrap 95% intervals.

1. Workshops recorded 75.9% attendance (`n=112`, 95% interval 67.9%–83.0%) versus 53.8% for social events (`n=65`, 41.5%–64.6%). This supports studying what workshop formats do well, not simply converting every event into a workshop.
2. Club members recorded 68.9% (`n=251`, 62.9%–74.5%) versus 53.2% for non-members (`n=141`, 45.4%–61.7%). Non-members may benefit from clearer onboarding and reminders.
3. Registrations 8–14 days early recorded 73.4% (`n=188`, 66.5%–79.8%) versus 48.6% for 1–3 days early (`n=72`, 37.5%–59.7%). Organizers can test earlier promotion in a prospective experiment.

The supplied data does not contain a separate technical/non-technical flag. Turnout Lab does not invent a mapping from event type to that missing field.

## 13. Product and privacy architecture

The saved scikit-learn pipeline is shared by CLI, single prediction, batch scoring, and Streamlit. Batch expected turnout is the sum of valid individual probabilities; rejected rows are excluded.

SQLite stores anonymous operational fields only: timestamp, source, model version, probability, risk band, reliability, and warning codes. Student IDs and raw features are not persisted.

No LLM or external API participates in runtime predictions.

## 14. Limitations and responsible use

- Only 397 leakage-safe labeled rows remain.
- The official test split is not independent.
- There are no event IDs or dates for event-level or temporal validation.
- No demographic fields are available for subgroup fairness assessment.
- Collection procedures and population coverage are undocumented.
- Associations can reflect confounding and should not be treated as causes.

A production version would collect prospective outcomes, add event/date identifiers, validate on a later semester, monitor drift, assess relevant subgroups with consent, and recalibrate periodically.

## 15. Speaking versions

### 30-second version

“Turnout Lab predicts whether a student registration will convert into attendance. The key discovery was that all 100 official test rows already appeared in training, so I quarantined every matching identity or feature fingerprint before using labels. I compared four model families with grouped nested validation and selected a calibrated random forest. Its ROC-AUC is 0.635, so I present it honestly as a modest reminder-prioritization signal. The Streamlit app supports single and batch scoring, uncertainty warnings, model diagnostics, and anonymous operations.”

### Two-minute version

Start with the problem and operational use. Explain the 100/100 overlap and 101-row quarantine. Describe fold-local cleaning, connected groups, five folds across five seeds, and inner-only tuning/calibration. State why the raw random forest beat the engineered version. Explain ROC-AUC, macro-F1, Brier score, and the 0.59 threshold. Finish with the five app views, three descriptive insights, privacy boundary, and modest-signal limitation.

### Five-minute version

Use sections 1–13 as the order:

1. problem, target, and permitted use;
2. audit and leakage quarantine;
3. cleaning and preprocessing;
4. grouped nested validation;
5. candidate comparison and raw-feature ablation;
6. calibration, threshold, and error/stability diagnostics;
7. single, batch, scenario, model-card, and operations flows;
8. three insights, privacy, limitations, and future work.

Do not memorize every sentence. Memorize the decision chain: **audit → quarantine → fold-local pipeline → grouped evaluation → calibration → decision policy → responsible product**.
