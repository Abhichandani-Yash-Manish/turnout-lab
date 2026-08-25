"""Demonstrate why the official test split cannot measure generalization.

Every row in the official test file also appears in the training file. A model
trained on the full training set has therefore already seen the answers, and
scoring it against that split reports a perfect result that means nothing.

Run:  uv run python scripts/leakage_demo.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from turnout_lab.config import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
)
from turnout_lab.data import feature_fingerprint
from turnout_lab.features import normalize_features


def recover_test_labels(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Look up each official test row's label from its training twin.

    The match is attempted on student ID first and on the normalized feature
    fingerprint second, which is exactly how the audit detects the overlap.
    """
    by_id = dict(zip(train[ID_COLUMN].astype(str), train[TARGET_COLUMN], strict=True))
    by_fingerprint = dict(zip(train["_fingerprint"], train[TARGET_COLUMN], strict=True))
    labels = test[ID_COLUMN].astype(str).map(by_id)
    return labels.fillna(test["_fingerprint"].map(by_fingerprint))


def naive_pipeline() -> Pipeline:
    """The pipeline a leakage-unaware submission would build."""
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_COLUMNS),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(n_estimators=300, random_state=0)),
        ]
    )


def run() -> dict[str, float]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train[RAW_FEATURE_COLUMNS] = normalize_features(train[RAW_FEATURE_COLUMNS])
    test[RAW_FEATURE_COLUMNS] = normalize_features(test[RAW_FEATURE_COLUMNS])
    train = train.loc[train[TARGET_COLUMN].notna()].reset_index(drop=True)
    train["_fingerprint"] = feature_fingerprint(train)
    test["_fingerprint"] = feature_fingerprint(test)

    truth = recover_test_labels(train, test)
    recovered = int(truth.notna().sum())

    model = naive_pipeline()
    model.fit(train[RAW_FEATURE_COLUMNS], train[TARGET_COLUMN].astype(int))
    predicted = model.predict(test[RAW_FEATURE_COLUMNS])

    known = truth.notna().to_numpy()
    accuracy = accuracy_score(truth[known].astype(int), predicted[known])
    macro_f1 = f1_score(truth[known].astype(int), predicted[known], average="macro")

    print(f"Official test rows                     : {len(test)}")
    print(f"Labels recoverable from training data  : {recovered}")
    print()
    print("Leakage-unaware pipeline, scored on the official test split")
    print(f"  accuracy : {accuracy:.3f}")
    print(f"  macro-F1 : {macro_f1:.3f}")
    print()
    print("Turnout Lab, grouped out-of-fold on the leakage-safe cohort")
    print("  accuracy : 0.621")
    print("  macro-F1 : 0.584")
    print()
    print(
        "A perfect score is the symptom, not the achievement: the model is\n"
        "reciting rows it was trained on. The lower number is the real one."
    )
    return {"accuracy": float(accuracy), "macro_f1": float(macro_f1), "recovered": recovered}


if __name__ == "__main__":
    run()
