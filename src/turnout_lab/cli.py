"""Reproducible command-line interface for audit, training, and prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from turnout_lab.config import (
    ARTIFACTS_DIR,
    FEATURE_CONTRACT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH,
    QUALITY_REPORT_PATH,
    TEST_PATH,
    TRAIN_PATH,
)
from turnout_lab.data import prepare_datasets, write_json
from turnout_lab.insights import build_insights
from turnout_lab.modeling import (
    evaluate_calibrated_champion,
    evaluate_candidates,
    train_final_bundle,
    write_training_artifacts,
)
from turnout_lab.prediction import AttendancePredictor


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run_audit(train_path: Path = TRAIN_PATH, test_path: Path = TEST_PATH) -> dict[str, Any]:
    prepared = prepare_datasets(train_path, test_path)
    write_json(QUALITY_REPORT_PATH, prepared.quality_report)
    write_json(FEATURE_CONTRACT_PATH, prepared.feature_contract)
    prepared.quarantine_index.to_csv(
        QUALITY_REPORT_PATH.parent.parent / "data" / "processed" / "quarantine_index.csv",
        index=False,
    )
    print(json.dumps(prepared.quality_report, indent=2, sort_keys=True))
    return prepared.quality_report


def run_training(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    quick: bool = False,
) -> dict[str, Any]:
    prepared = prepare_datasets(train_path, test_path)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(QUALITY_REPORT_PATH, prepared.quality_report)
    write_json(FEATURE_CONTRACT_PATH, prepared.feature_contract)
    prepared.quarantine_index.to_csv(
        QUALITY_REPORT_PATH.parent.parent / "data" / "processed" / "quarantine_index.csv",
        index=False,
    )

    summaries, candidate_folds, champion = evaluate_candidates(prepared.development, quick=quick)
    calibrated = evaluate_calibrated_champion(prepared.development, champion, quick=quick)
    bundle, final_oof = train_final_bundle(
        prepared.development, champion, prepared.feature_contract, quick=quick
    )

    dummy_summary = next(
        summary for summary in summaries if summary["candidate"] == "dummy"
    )
    calibrated_brier = calibrated["summary"]["brier"]["mean"]
    dummy_brier = dummy_summary["brier"]["mean"]
    calibrated["summary"]["brier_skill"] = {
        "mean": float(1 - calibrated_brier / dummy_brier),
        "std": None,
    }

    metrics: dict[str, Any] = {
        "generated_from_quick_run": bool(quick),
        "dataset": prepared.quality_report["development"],
        "overlap_audit": prepared.quality_report["overlap"],
        "evaluation_protocol": {
            "outer_folds": 3 if quick else 5,
            "outer_seeds": [11] if quick else [11, 22, 33, 44, 55],
            "inner_folds": 3 if quick else 4,
            "grouping": "connected components sharing student_id or normalized feature fingerprint",
            "calibration": "sigmoid calibration fit only on group-safe training folds",
            "threshold": "chosen inside training folds to maximize macro-F1",
        },
        "candidate_summaries": summaries,
        "candidate_fold_metrics": candidate_folds,
        "champion": {
            "candidate": champion.name,
            "feature_mode": champion.feature_mode,
            "model_version": bundle["model_version"],
            "decision_threshold": bundle["decision_threshold"],
            "risk_thresholds": bundle["risk_thresholds"],
        },
        "calibrated_champion": calibrated,
        "final_fit_oof": {
            "rows": int(len(final_oof)),
            "mean_attendance_probability": float(final_oof.mean()),
        },
        "descriptive_insights": build_insights(prepared.development),
        "interpretation": {
            "signal_label": "weak"
            if calibrated["summary"]["roc_auc"]["mean"] < 0.60
            else "modest"
            if calibrated["summary"]["roc_auc"]["mean"] < 0.70
            else "useful",
            "warning": "Associations and model estimates are not causal. Use scores for supportive outreach, never punitive decisions.",
        },
    }
    metrics = _json_safe(metrics)
    write_training_artifacts(MODEL_PATH, METRICS_PATH, bundle, metrics)

    predictor = AttendancePredictor(bundle)
    predictions = predictor.score_dataframe(prepared.test.drop(columns=["_fingerprint"]))
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(
        json.dumps(
            {
                "champion": metrics["champion"],
                "calibrated_summary": metrics["calibrated_champion"]["summary"],
                "predictions": str(PREDICTIONS_PATH),
                "prediction_rows": int(len(predictions)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return metrics


def run_prediction(input_path: Path, output_path: Path, model_path: Path = MODEL_PATH) -> None:
    predictor = AttendancePredictor.from_path(model_path)
    inputs = pd.read_csv(input_path)
    outputs = predictor.score_dataframe(inputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outputs.to_csv(output_path, index=False)
    print(f"Wrote {len(outputs)} predictions to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="turnout-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Run the data-quality and overlap audit.")
    audit.add_argument("--train", type=Path, default=TRAIN_PATH)
    audit.add_argument("--test", type=Path, default=TEST_PATH)

    train = subparsers.add_parser("train", help="Evaluate candidates and train the final model.")
    train.add_argument("--train", type=Path, default=TRAIN_PATH)
    train.add_argument("--test", type=Path, default=TEST_PATH)
    train.add_argument("--quick", action="store_true", help="Use reduced folds and tree counts for smoke tests.")

    predict = subparsers.add_parser("predict", help="Score a CSV using the saved model bundle.")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--model", type=Path, default=MODEL_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "audit":
        run_audit(args.train, args.test)
    elif args.command == "train":
        run_training(args.train, args.test, quick=args.quick)
    elif args.command == "predict":
        run_prediction(args.input, args.output, args.model)


if __name__ == "__main__":
    main()

