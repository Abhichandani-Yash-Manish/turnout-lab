# Turnout Lab — viva and evaluator questions

Use these answers as concepts, not scripts. Give the first sentence, then expand only if the evaluator asks.

## Problem and data

### 1. What problem are you solving?

Binary classification of whether a registered student will attend, with calibrated probability output for reminder and turnout planning.

### 2. Why did you choose Task 1?

It offered a real data-quality and evaluation problem, not just model fitting. The compromised official split created an opportunity to demonstrate responsible ML engineering.

### 3. What is one row?

One student registration for one event. The missing event identifier is an important limitation.

### 4. What was the most important finding?

All 100 official test IDs and normalized feature rows occur in training. That makes the supplied test split unsuitable for independent accuracy measurement.

### 5. Why quarantine 101 training rows for 100 test rows?

At least one test-linked identity or fingerprint maps to more than one training row. Quarantine operates on matching training rows, not only unique test rows.

### 6. Why not use the matching labels and get perfect predictions?

That would be target leakage: reproducing known labels instead of estimating unseen behavior. It would invalidate both the score and the project’s credibility.

### 7. Why exclude `student_id`?

It is an identifier, not a generalizable behavioral feature. Including it encourages memorization and cannot help with genuinely new students.

### 8. The document mentions technical/non-technical events. Where is that feature?

The supplied files do not contain a separate technical/non-technical column. I used the actual `event_type` categories and did not invent an unsupported mapping.

## Cleaning and preprocessing

### 9. How did you handle missing values?

Targets were removed because supervised learning needs known labels. Numeric predictors use median imputation and categories use most-frequent imputation, fitted only inside each training fold.

### 10. Why must preprocessing happen inside folds?

Fitting imputation or encoding on the full dataset lets validation information influence training. A pipeline guarantees the transformation is learned only from the current training fold.

### 11. How did you handle inconsistent casing?

Whitespace and categorical casing are normalized before encoding, so variants such as `YES`, `Yes`, and `yes` represent the same value.

### 12. Why not silently fix impossible histories?

There is no authoritative value to replace them with. I preserve source truth, prevent an invalid derived rate, and expose an anomaly flag or input validation warning.

### 13. Why retain the 120 km row?

It is unusual but not impossible. Deleting it without evidence would be arbitrary, so it is retained, flagged, and handled robustly in the engineered representation.

### 14. How are unseen categories handled?

The one-hot encoder ignores unknown categories so scoring can continue, while the reliability layer marks the registration for review.

## Validation and modeling

### 15. Why grouped cross-validation?

Rows sharing an identity or normalized fingerprint must not cross train and validation. Connected groups enforce that boundary.

### 16. Why nested validation?

Inner folds make choices—hyperparameters, calibration, and threshold—while outer folds estimate performance. This separates model selection from evaluation.

### 17. Why repeat across five seeds?

With only 397 rows, one split can be lucky or unlucky. Twenty-five outer results show variability and make stability visible.

### 18. Which models did you compare?

A prevalence baseline, regularized logistic regression, balanced random forest, and gradient boosting, with raw and engineered feature variants where applicable.

### 19. Why did random forest win?

Under the locked selection rule, the raw-feature random forest had the strongest eligible ROC-AUC and macro-F1 while keeping Brier score better than the prevalence baseline.

### 20. Why did raw features beat engineered features?

The derived fields were mostly transformations of information the forest could already partition. On this small cohort they added complexity without validation improvement.

### 21. Why no SMOTE?

The 63.5/36.5 class balance is not extreme, and synthesizing mixed categorical/numeric registrations can create unrealistic examples. Class weighting, class-specific metrics, and threshold selection were safer.

### 22. Why no XGBoost, CatBoost, or neural network?

The bottleneck is small, compromised data—not model capacity. More complex candidates would enlarge the search space and explanation burden without creating independent evidence.

### 23. How did you control overfitting?

Leakage quarantine, connected groups, nested validation, small locked grids, repeated outer folds, fold-local preprocessing, baseline comparison, and no post-result changes to the selection rule.

## Metrics and decisions

### 24. Why is accuracy insufficient?

Attendance is the majority class. Accuracy can hide poor no-show detection, so I report macro-F1, balanced accuracy, both classes’ precision/recall/F1, ranking, and calibration metrics.

### 25. What is Precision?

Of the rows predicted as a class, Precision is the fraction actually in that class. No-show precision answers how many reminder-priority predictions were genuine no-shows.

### 26. What is Recall?

Of all actual members of a class, Recall is the fraction found. No-show recall answers how many real no-shows the policy identifies.

### 27. What is F1-score?

The harmonic mean of Precision and Recall. Macro-F1 averages both classes equally, preventing the attendance majority from dominating the decision.

### 28. What does ROC-AUC 0.635 mean?

The model has modest ranking ability: a randomly chosen attendee tends to receive a higher attendance score than a randomly chosen no-show about 63.5% of the time.

### 29. What is PR-AUC?

It summarizes precision-recall performance across thresholds and is useful when class prevalence matters. I report it for attendance and no-show perspectives.

### 30. What is Brier score?

The mean squared error of predicted probabilities. Lower is better; the model’s 0.221 is about 5.1% better than the prevalence baseline.

### 31. Why calibrate probabilities?

Classification models can rank correctly while producing overconfident scores. Sigmoid calibration makes percentage outputs more defensible.

### 32. Why is the threshold 0.59 instead of 0.5?

It was selected on leakage-safe development out-of-fold predictions to maximize macro-F1. The dashboard shows how other thresholds change Precision and Recall.

### 33. What does the confusion matrix show?

The normalized matrix shows class-specific correct and incorrect decision rates at the final 0.59 threshold, using repeated outer-fold predictions rather than training predictions. Its rates differ slightly from headline mean Recall because the headline averages fold-local thresholds, while the matrix applies one final policy uniformly.

### 34. What does 1.52× no-show lift mean?

The highest-risk 20% contains no-shows at about 1.52 times the base no-show rate. It supports prioritization but does not capture every no-show.

## Product and responsibility

### 35. What is the difference between probability and reliability?

Probability is model output. Reliability describes whether the input resembles the development contract. An unfamiliar input can receive a probability but low reliability.

### 36. Are the local explanations causal?

No. They are one-field-at-a-time sensitivity deltas against a reference profile and are explicitly labelled as associations.

### 37. How is expected batch turnout calculated?

By summing calibrated attendance probabilities for valid rows. Rejected rows are excluded. The result is an expectation, not a guaranteed count.

### 38. What data is persisted?

Only anonymous operational fields: timestamp, source, model version, probability, risk band, reliability, and warning codes. No student ID or raw feature row is stored.

### 39. What fairness claims can you make?

Very limited ones. The data has no demographic attributes, so subgroup performance cannot be assessed. Omitting sensitive features does not prove fairness.

### 40. How was AI used?

AI assisted with development, review, and documentation. The modeling decisions, data audit, metrics, tests, and behavior were verified by the author. No LLM participates in runtime prediction.

### 41. What would you improve with more time and data?

Collect prospective event IDs and dates, validate on a later semester, capture outcomes after deployment, monitor drift and calibration, evaluate relevant subgroups with consent, and test reminder interventions experimentally.

### 42. What is your most honest conclusion?

The dataset contains modest signal. The project’s value is not a spectacular score; it is a traceable process that avoids leakage, quantifies uncertainty, and limits the model to a responsible decision-support role.
