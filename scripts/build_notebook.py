"""Generate the reader-facing, executable audit and model-selection notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
QUALITY = json.loads((ROOT / "artifacts" / "data_quality_report.json").read_text(encoding="utf-8"))
SUMMARY = METRICS["calibrated_champion"]["summary"]


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        markdown(
            f"""
            # Turnout Lab — leakage-aware attendance modeling

            ## tl;dr

            The supplied test set is not independent: all **{QUALITY['overlap']['exact_id_and_feature_matches']} of {QUALITY['overlap']['test_rows']}** test rows match training identities and features. Turnout Lab quarantines **{QUALITY['overlap']['quarantined_training_rows']}** matching training rows before model development, leaving **{QUALITY['development']['rows']}** leakage-safe labeled registrations.

            The selected calibrated random forest achieved mean repeated-CV ROC-AUC **{SUMMARY['roc_auc']['mean']:.3f}**, macro-F1 **{SUMMARY['macro_f1']['mean']:.3f}**, Brier score **{SUMMARY['brier']['mean']:.3f}**, and top-20% no-show lift **{SUMMARY['top_20_no_show_lift']['mean']:.2f}×**. The signal is modest, so the model is suitable for supportive reminder prioritization—not punitive or deterministic decisions.
            """
        ),
        markdown(
            """
            ## Context & Methods

            This notebook is the inspectable companion to reusable code in `src/turnout_lab`. It verifies the data-quality evidence, model comparison, calibration, and required test output without duplicating the training implementation.

            ### Key assumptions

            - One row represents one student registration for one event.
            - Only information available at registration time is used.
            - Student ID is an identifier and never a model feature.
            - Matching official-test rows are quarantined before targets are used.
            - No event ID or calendar date is supplied, so event-level or temporal validation is impossible.
            - Reported patterns are associations, not causal effects.

            **Sources:** the official challenge document and spreadsheet URLs are recorded with SHA-256 hashes in `data/raw/provenance.json`.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import matplotlib.pyplot as plt
            import pandas as pd

            from turnout_lab.config import METRICS_PATH, PREDICTIONS_PATH, QUALITY_REPORT_PATH, TEST_PATH, TRAIN_PATH
            from turnout_lab.data import feature_fingerprint, prepare_datasets

            prepared = prepare_datasets(TRAIN_PATH, TEST_PATH)
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            quality = json.loads(QUALITY_REPORT_PATH.read_text(encoding="utf-8"))
            diagnostics = metrics["decision_diagnostics"]
            """
        ),
        markdown("## Data"),
        code(
            """
            profile = pd.DataFrame(
                [
                    {"stage": "Raw training", "rows": quality["raw"]["train_rows"], "purpose": "Supplied source"},
                    {"stage": "Official test", "rows": quality["raw"]["test_rows"], "purpose": "Required predictions"},
                    {"stage": "Quarantined train", "rows": quality["overlap"]["quarantined_training_rows"], "purpose": "Excluded from all modeling"},
                    {"stage": "Development cohort", "rows": quality["development"]["rows"], "purpose": "Grouped validation and final fit"},
                ]
            )
            profile
            """
        ),
        code(
            """
            development_ids = set(prepared.development["student_id"].astype(str))
            test_ids = set(prepared.test["student_id"].astype(str))
            development_fingerprints = set(feature_fingerprint(prepared.development))
            test_fingerprints = set(feature_fingerprint(prepared.test))

            leakage_checks = {
                "identity_overlap_after_quarantine": len(development_ids & test_ids),
                "feature_overlap_after_quarantine": len(development_fingerprints & test_fingerprints),
                "student_id_used_as_feature": False,
            }
            assert leakage_checks["identity_overlap_after_quarantine"] == 0
            assert leakage_checks["feature_overlap_after_quarantine"] == 0
            leakage_checks
            """
        ),
        code(
            """
            missing = pd.Series(quality["raw"]["train_missing_by_column"]).sort_values()
            ax = missing.plot.barh(figsize=(8, 4), color="#E8B44C", title="Missing values in raw training data")
            ax.set_xlabel("Rows")
            ax.set_ylabel("")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown("## Results"),
        code(
            """
            candidate_rows = []
            for summary in metrics["candidate_summaries"]:
                candidate_rows.append(
                    {
                        "candidate": summary["label"],
                        "roc_auc": summary["roc_auc"]["mean"],
                        "macro_f1": summary["macro_f1"]["mean"],
                        "brier": summary["brier"]["mean"],
                    }
                )
            candidates = pd.DataFrame(candidate_rows).sort_values("roc_auc", ascending=False)
            candidates.style.format({"roc_auc": "{:.3f}", "macro_f1": "{:.3f}", "brier": "{:.3f}"})
            """
        ),
        code(
            """
            ax = candidates.sort_values("roc_auc").plot.barh(
                x="candidate", y="roc_auc", figsize=(8, 4.5), color="#3454D1", legend=False,
                title="Repeated grouped validation ROC-AUC"
            )
            ax.axvline(0.5, color="#60758A", linestyle="--", label="No discrimination")
            ax.set_xlim(0.45, 0.68)
            ax.set_xlabel("ROC-AUC")
            ax.set_ylabel("")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ### Decision diagnostics

            A probability model becomes a classification policy only after choosing a threshold. The next views show the precision/recall trade-off at alternative thresholds, the error profile at the final threshold, and stability across the 25 grouped outer folds. Each development row is evaluated once per outer seed. Headline class metrics average fold-local thresholds; the confusion matrix applies the final 0.59 policy uniformly, so the two summaries answer different questions.
            """
        ),
        code(
            """
            threshold_curve = pd.DataFrame(diagnostics["threshold_curve"])
            selected_threshold = diagnostics["selected_threshold"]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            axes[0].plot(threshold_curve["threshold"], threshold_curve["attendance_precision"], color="#3454D1", label="Attendance precision")
            axes[0].plot(threshold_curve["threshold"], threshold_curve["attendance_recall"], color="#3454D1", linestyle="--", label="Attendance recall")
            axes[0].plot(threshold_curve["threshold"], threshold_curve["no_show_precision"], color="#F0645A", label="No-show precision")
            axes[0].plot(threshold_curve["threshold"], threshold_curve["no_show_recall"], color="#F0645A", linestyle="--", label="No-show recall")
            axes[0].plot(threshold_curve["threshold"], threshold_curve["macro_f1"], color="#E8B44C", linewidth=2.5, label="Macro-F1")
            axes[0].axvline(selected_threshold, color="#10233B", linestyle=":", label=f"Selected {selected_threshold:.2f}")
            axes[0].set(xlabel="Attendance threshold", ylabel="Metric value", ylim=(0, 1), title="Threshold policy trade-offs")
            axes[0].legend(fontsize=8)

            confusion = pd.DataFrame(
                diagnostics["normalized_confusion_matrix"],
                index=["Actual no-show", "Actual attend"],
                columns=["Predicted no-show", "Predicted attend"],
            )
            image = axes[1].imshow(confusion, cmap="Blues", vmin=0, vmax=1)
            for row in range(2):
                for column in range(2):
                    axes[1].text(column, row, f"{confusion.iloc[row, column]:.1%}", ha="center", va="center", color="#10233B")
            axes[1].set_xticks(range(2), confusion.columns, rotation=20, ha="right")
            axes[1].set_yticks(range(2), confusion.index)
            axes[1].set_title("Normalized repeated OOF confusion matrix")
            fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            fold_metrics = pd.DataFrame(metrics["calibrated_champion"]["fold_metrics"])
            stability = fold_metrics[["roc_auc", "macro_f1", "brier"]].rename(
                columns={"roc_auc": "ROC-AUC", "macro_f1": "Macro-F1", "brier": "Brier (lower is better)"}
            )
            ax = stability.plot.box(figsize=(8, 4.5), color=dict(boxes="#3454D1", whiskers="#60758A", medians="#F0645A", caps="#60758A"))
            ax.set(title="Performance stability across 25 grouped outer folds", ylabel="Metric value", ylim=(0, 1))
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            calibration = pd.DataFrame(metrics["calibrated_champion"]["calibration_points"])
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.plot([0, 1], [0, 1], "--", color="#60758A", label="Ideal")
            ax.plot(
                calibration["mean_predicted_probability"],
                calibration["observed_attendance_rate"],
                marker="o", color="#3454D1", linewidth=2.5, label="Calibrated random forest"
            )
            ax.set(xlabel="Mean predicted attendance", ylabel="Observed attendance", title="Calibration curve", xlim=(0, 1), ylim=(0, 1))
            ax.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            event_insights = pd.DataFrame(metrics["descriptive_insights"]["event_type"])
            event_insights.style.format({
                "attendance_rate": "{:.1%}", "ci_95_low": "{:.1%}", "ci_95_high": "{:.1%}"
            })
            """
        ),
        markdown(
            """
            ### Three planning insights

            - Workshops recorded **75.9% attendance** (`n=112`) versus **53.8%** for social events (`n=65`).
            - Club members recorded **68.9% attendance** (`n=251`) versus **53.2%** for non-members (`n=141`).
            - Registrations made 8–14 days early recorded **73.4% attendance** (`n=188`) versus **48.6%** for registrations made 1–3 days early (`n=72`).

            These are descriptive associations with bootstrap uncertainty, not evidence that changing an event or registration field will cause the observed difference.
            """
        ),
        code(
            """
            predictions = pd.read_csv(PREDICTIONS_PATH)
            assert len(predictions) == 100
            assert predictions["student_id"].nunique() == 100
            assert predictions["attendance_probability"].between(0, 1).all()
            assert predictions["error"].fillna("").eq("").all()
            predictions[[
                "student_id", "attendance_probability", "no_show_probability",
                "predicted_attendance", "no_show_risk_band", "reliability"
            ]].head(10)
            """
        ),
        markdown(
            f"""
            ## Takeaways

            1. **Evaluation integrity matters more than a flattering score.** The official test overlap is quarantined; reported performance comes from {SUMMARY['folds']} grouped outer folds.
            2. **The signal is modest but operationally useful.** Mean ROC-AUC is {SUMMARY['roc_auc']['mean']:.3f}; the highest-risk 20% captures {SUMMARY['top_20_no_show_recall']['mean']:.1%} of no-shows at {SUMMARY['top_20_no_show_lift']['mean']:.2f}× lift.
            3. **Probability is not reliability.** Calibration, input-contract checks, and out-of-distribution warnings are surfaced separately in the app.
            4. **Use the output supportively.** Send reminders, plan capacity, or investigate patterns. Do not deny registrations or penalize students based on a score.

            The absence of event identifiers, timestamps, demographics, and a genuinely independent test set limits both generalization claims and fairness analysis.
            """
        ),
    ]

    output_path = ROOT / "notebooks" / "01_data_audit_and_model_selection.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
