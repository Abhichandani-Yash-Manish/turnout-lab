"""Turnout Lab Streamlit decision dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Streamlit Community Cloud installs dependencies but does not always install the
# repository itself as a package. Adding src/ keeps the import working whether or
# not `pip install .` ran.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from turnout_lab.config import (
    DATABASE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    QUALITY_REPORT_PATH,
    TEST_PATH,
)
from turnout_lab.database import (
    log_batch,
    log_batch_predictions,
    log_prediction,
    operations_summary,
)
from turnout_lab.prediction import AttendancePredictor, summarize_batch
from turnout_lab.schemas import AttendanceInput, PredictionResult

INK = "#10233B"
COBALT = "#3454D1"
CORAL = "#F0645A"
GOLD = "#E8B44C"
MIST = "#EAF0F6"
SLATE = "#60758A"


st.set_page_config(
    page_title="Turnout Lab",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    f"""
    <style>
    :root {{
        --ink: {INK}; --cobalt: {COBALT}; --coral: {CORAL};
        --gold: {GOLD}; --mist: {MIST}; --slate: {SLATE};
    }}
    .stApp {{ background: #F7FAFC; color: var(--ink); }}
    .block-container {{ max-width: 1240px; padding-top: 1.7rem; padding-bottom: 4rem; }}
    h1, h2, h3 {{ font-family: "Avenir Next", "Trebuchet MS", sans-serif; color: var(--ink); }}
    p, label, .stMarkdown {{ font-family: "DM Sans", system-ui, sans-serif; }}
    [data-testid="stMetric"] {{
        background: white; border: 1px solid #D9E3EC; border-top: 4px solid var(--cobalt);
        border-radius: 10px; padding: .8rem 1rem;
    }}
    [data-testid="stMetricLabel"] {{ color: var(--slate); letter-spacing: .02em; }}
    .tl-header {{
        background: white; border: 1px solid #D9E3EC; border-radius: 16px;
        padding: 1.35rem 1.55rem 1.25rem; margin-bottom: 1rem;
        box-shadow: 0 8px 26px rgba(16,35,59,.06);
    }}
    .tl-eyebrow {{
        color: var(--cobalt); font: 700 .72rem/1.2 ui-monospace, monospace;
        text-transform: uppercase; letter-spacing: .14em; margin-bottom: .45rem;
    }}
    .tl-title {{
        color: var(--ink); font: 750 2.45rem/1 "Avenir Next", "Trebuchet MS", sans-serif;
        letter-spacing: -.055em; margin: 0;
    }}
    .tl-subtitle {{ color: var(--slate); max-width: 760px; margin: .65rem 0 0; font-size: 1rem; }}
    .tl-stamp {{
        display: inline-block; color: var(--ink); background: #EDF1FF; border: 1px dashed var(--cobalt);
        border-radius: 999px; padding: .25rem .62rem; font: 700 .7rem ui-monospace, monospace;
        margin-top: .8rem;
    }}
    .ticket {{
        background: white; border: 1px solid #D9E3EC; border-radius: 14px;
        padding: 1.25rem 1.35rem; position: relative; overflow: hidden;
    }}
    .ticket::after {{
        content: ""; position: absolute; right: -1px; top: 0; bottom: 0;
        border-right: 6px dotted #D9E3EC;
    }}
    .probability {{ font: 760 3.2rem/1 "Avenir Next", sans-serif; color: var(--ink); letter-spacing: -.06em; }}
    .probability-label {{ color: var(--slate); font-size: .85rem; margin-top: .25rem; }}
    .runway {{
        height: 18px; background: #E7EDF3; border-radius: 999px; position: relative;
        margin: 1.05rem 0 .45rem; overflow: visible;
    }}
    .runway-fill {{ height: 100%; border-radius: 999px; background: var(--cobalt); }}
    .runway-marker {{
        position: absolute; top: -5px; width: 2px; height: 28px; background: var(--coral);
    }}
    .runway-marker::after {{
        content: "decision"; position: absolute; top: 29px; left: 50%; transform: translateX(-50%);
        font: 650 .62rem ui-monospace, monospace; color: var(--coral);
    }}
    .runway-labels {{ display: flex; justify-content: space-between; color: var(--slate); font-size: .72rem; }}
    .pill {{
        display: inline-block; padding: .28rem .62rem; margin: .25rem .3rem .2rem 0;
        border-radius: 999px; font: 700 .72rem ui-monospace, monospace; text-transform: uppercase;
    }}
    .pill-low {{ background: #E8F2FF; color: #244BA9; }}
    .pill-medium {{ background: #FFF2D2; color: #815A00; }}
    .pill-high {{ background: #FFE5E2; color: #A72F27; }}
    .note {{
        border-left: 4px solid var(--gold); background: #FFF9E9; padding: .75rem .9rem;
        color: #5F4A17; border-radius: 0 8px 8px 0; margin: .7rem 0;
    }}
    .source-line {{ color: var(--slate); font: .72rem ui-monospace, monospace; }}
    div[data-baseweb="tab-list"] {{ gap: .2rem; }}
    button[data-baseweb="tab"] {{ padding: .65rem .8rem; }}
    .stButton > button, .stDownloadButton > button {{ border-radius: 8px; font-weight: 700; }}
    :focus-visible {{ outline: 3px solid {GOLD} !important; outline-offset: 2px; }}
    @media (max-width: 700px) {{
        .tl-title {{ font-size: 2rem; }} .probability {{ font-size: 2.5rem; }}
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    }}
    @media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior: auto !important; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> AttendancePredictor:
    return AttendancePredictor.from_path(MODEL_PATH)


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_official_test() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH)


def percent(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def clean_chart(
    figure: go.Figure, height: int = 350, top_margin: int = 45
) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=top_margin, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=INK),
        hoverlabel=dict(bgcolor="white"),
    )
    figure.update_xaxes(gridcolor="#E2E9F0", zeroline=False)
    figure.update_yaxes(gridcolor="#E2E9F0", zeroline=False)
    return figure


def render_result(result: PredictionResult) -> None:
    risk = result.no_show_risk_band.value
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(
            f"""
            <div class="ticket">
              <div class="tl-eyebrow">Attendance estimate</div>
              <div class="probability">{percent(result.attendance_probability, 0)}</div>
              <div class="probability-label">likely to attend · calibrated probability</div>
              <div class="runway">
                <div class="runway-fill" style="width:{result.attendance_probability * 100:.2f}%"></div>
                <div class="runway-marker" style="left:{result.decision_threshold * 100:.2f}%"></div>
              </div>
              <div class="runway-labels"><span>0% attendance</span><span>100% attendance</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        verdict = "Likely to attend" if result.predicted_attendance else "Prioritize a reminder"
        st.subheader(verdict)
        st.markdown(
            f'<span class="pill pill-{risk}">{risk} no-show risk</span>'
            f'<span class="pill pill-{result.reliability.value}">{result.reliability.value} reliability</span>',
            unsafe_allow_html=True,
        )
        st.write(f"No-show probability: **{percent(result.no_show_probability)}**")
        if result.status.value == "review_required":
            st.warning("Treat this score as a review prompt: at least one input is outside the development contract.")

    st.markdown("#### Factors associated with this score")
    if result.reason_codes:
        for reason in result.reason_codes:
            st.markdown(f"- {reason}")
    else:
        st.caption("No single field moved the estimate by at least one percentage point from the reference profile.")
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)
    st.caption("These are one-field-at-a-time scenario deltas, not causal explanations.")


def candidate_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for summary in metrics["candidate_summaries"]:
        rows.append(
            {
                "Candidate": summary["label"],
                "ROC-AUC": summary["roc_auc"]["mean"],
                "Macro-F1": summary["macro_f1"]["mean"],
                "Brier": summary["brier"]["mean"],
            }
        )
    return pd.DataFrame(rows)


if not MODEL_PATH.exists() or not METRICS_PATH.exists() or not QUALITY_REPORT_PATH.exists():
    st.error("Model artifacts are missing. Run `uv run turnout-lab train` before starting the dashboard.")
    st.stop()

predictor = load_predictor()
metrics = load_json(METRICS_PATH)
quality = load_json(QUALITY_REPORT_PATH)
contract = predictor.contract
champion_summary = metrics["calibrated_champion"]["summary"]
diagnostics = metrics.get("decision_diagnostics")

st.markdown(
    """
    <div class="tl-header">
      <div class="tl-eyebrow">Leakage-aware decision instrument</div>
      <div class="tl-title">Turnout Lab</div>
      <p class="tl-subtitle">Forecast attendance, identify registrations that may benefit from a reminder, and inspect exactly where the evidence becomes uncertain.</p>
      <span class="tl-stamp">TEST OVERLAP QUARANTINED BEFORE MODELING</span>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_columns = st.columns(4)
overview_columns[0].metric("Leakage-safe rows", quality["development"]["rows"], "from 508 raw")
overview_columns[1].metric("Repeated CV ROC-AUC", f"{champion_summary['roc_auc']['mean']:.3f}")
overview_columns[2].metric("Macro-F1", f"{champion_summary['macro_f1']['mean']:.3f}")
overview_columns[3].metric("Brier skill", percent(champion_summary["brier_skill"]["mean"]), "vs prevalence")
st.caption(
    f"Model: {metrics['champion']['candidate'].replace('_', ' ')} · "
    f"{metrics['champion']['feature_mode']} features · {metrics['champion']['model_version']}"
)

view_names = ["Predict", "Batch score", "Scenario lab", "Model card", "Data & operations"]


def keep_view(view: str) -> None:
    st.session_state["active_view"] = view


active_view = st.session_state.get("active_view", "Predict")
if active_view not in view_names:
    active_view = "Predict"
predict_tab, batch_tab, scenario_tab, model_tab, data_tab = st.tabs(
    view_names, default=active_view
)

with predict_tab:
    st.markdown("### Score one registration")
    st.write("Enter information known at registration time. The student ID is returned for mapping but is never stored.")
    reference = contract["reference_profile"]
    with st.form("single_prediction_form"):
        student_id = st.text_input("Student ID (optional)", placeholder="DEMO-001")
        first, second = st.columns(2)
        with first:
            event_type = st.selectbox(
                "Event type", contract["categorical"]["event_type"]["allowed_values"]
            )
            event_day = st.selectbox(
                "Event day", contract["categorical"]["event_day"]["allowed_values"]
            )
            event_time = st.selectbox(
                "Event time", contract["categorical"]["event_time"]["allowed_values"]
            )
            club_member = st.selectbox(
                "Club member", contract["categorical"]["club_member"]["allowed_values"]
            )
        with second:
            registration_days = st.number_input(
                "Registration days before", min_value=0, max_value=365, value=int(reference["registration_days_before"])
            )
            previous_registered = st.number_input(
                "Previous events registered", min_value=0, max_value=100, value=int(reference["previous_events_registered"])
            )
            previous_attended = st.number_input(
                "Previous events attended", min_value=0, max_value=100, value=int(reference["previous_events_attended"])
            )
            distance = st.number_input(
                "Travel distance (km)", min_value=0.0, max_value=1000.0, value=float(reference["travel_distance_km"]), step=0.5
            )
        submitted = st.form_submit_button(
            "Estimate turnout",
            type="primary",
            width="stretch",
            on_click=keep_view,
            args=("Predict",),
        )

    if submitted:
        try:
            attendance_input = AttendanceInput(
                student_id=student_id or None,
                event_type=event_type,
                registration_days_before=registration_days,
                previous_events_registered=previous_registered,
                previous_events_attended=previous_attended,
                club_member=club_member,
                event_day=event_day,
                event_time=event_time,
                travel_distance_km=distance,
            )
            result = predictor.predict(attendance_input)
            st.session_state["latest_result"] = result
            st.session_state["latest_input"] = attendance_input
            if not log_prediction(DATABASE_PATH, result):
                st.warning("Prediction completed, but the anonymous activity log is temporarily unavailable.")
        except ValidationError as error:
            st.error(error.errors()[0]["msg"])
    if "latest_result" in st.session_state:
        render_result(st.session_state["latest_result"])

with batch_tab:
    st.markdown("### Score a registration list")
    st.write("Upload the challenge schema as CSV, or run the bundled official test snapshot.")
    uploaded = st.file_uploader(
        "Registration CSV", type=["csv"], on_change=keep_view, args=("Batch score",)
    )
    if uploaded is not None:
        try:
            st.session_state["batch_source"] = pd.read_csv(uploaded)
            st.session_state["batch_source_name"] = uploaded.name
        except Exception as error:  # noqa: BLE001 - user-facing upload boundary
            st.error(f"The CSV could not be read: {error}")
    use_official = st.button(
        "Use official 100-row test snapshot", on_click=keep_view, args=("Batch score",)
    )
    if use_official:
        st.session_state["batch_source"] = load_official_test()
        st.session_state["batch_source_name"] = "official_test.csv"
    source_frame: pd.DataFrame | None = st.session_state.get("batch_source")
    source_name = st.session_state.get("batch_source_name", "")
    if source_frame is not None:
        st.caption(f"Loaded {len(source_frame)} rows from {source_name}.")
        if st.button(
            "Score loaded rows", type="primary", on_click=keep_view, args=("Batch score",)
        ):
            try:
                outputs = predictor.score_dataframe(source_frame)
                st.session_state["batch_outputs"] = outputs
                batch_summary = summarize_batch(outputs, predictor.bundle["model_version"])
                st.session_state["batch_summary"] = batch_summary
                batch_logged = log_batch(
                    DATABASE_PATH, predictor.bundle["model_version"], batch_summary
                )
                predictions_logged = log_batch_predictions(DATABASE_PATH, outputs)
                if not batch_logged or not predictions_logged:
                    st.warning(
                        "Batch scoring completed, but the anonymous activity log is temporarily unavailable."
                    )
            except ValueError as error:
                st.error(str(error))
    if "batch_outputs" in st.session_state:
        outputs = st.session_state["batch_outputs"]
        batch_summary = st.session_state.get("batch_summary") or summarize_batch(
            outputs, predictor.bundle["model_version"]
        )
        planning_cards = st.columns(3)
        planning_cards[0].metric("Valid registrations", batch_summary.valid_rows)
        planning_cards[1].metric(
            "Expected attendees", f"{batch_summary.expected_attendees:.1f}"
        )
        planning_cards[2].metric(
            "Expected no-shows", f"{batch_summary.expected_no_shows:.1f}"
        )
        quality_cards = st.columns(3)
        quality_cards[0].metric("High no-show risk", batch_summary.high_risk_count)
        quality_cards[1].metric(
            "Review required", batch_summary.review_required_rows
        )
        quality_cards[2].metric("Rejected", batch_summary.rejected_rows)
        st.markdown(
            '<div class="note">Expected totals are sums of individual calibrated probabilities. '
            "They support aggregate planning; they are not guaranteed attendance counts.</div>",
            unsafe_allow_html=True,
        )
        display = outputs.copy()
        display["likely_to_attend"] = display["attendance_probability"].map(
            lambda value: "—" if pd.isna(value) else f"{value * 100:.0f}%"
        )
        display_columns = [
            "student_id",
            "likely_to_attend",
            "no_show_risk_band",
            "reliability",
            "status",
            "error",
        ]
        st.dataframe(display[display_columns].head(100), width="stretch", hide_index=True)
        st.caption(
            "The downloadable CSV keeps full-precision probabilities; this table rounds them for reading."
        )
        st.download_button(
            "Download scored CSV",
            data=outputs.to_csv(index=False).encode("utf-8"),
            file_name="turnout_lab_predictions.csv",
            mime="text/csv",
        )

with scenario_tab:
    st.markdown("### Compare organizer-controlled scenarios")
    st.write("Hold the attendee profile fixed and change event design choices. The delta is predictive, not causal.")
    base_column, proposal_column = st.columns(2)

    def scenario_controls(container, prefix: str, title: str) -> dict:
        with container:
            st.markdown(f"#### {title}")
            return {
                "event_type": st.selectbox(
                    "Event type",
                    contract["categorical"]["event_type"]["allowed_values"],
                    key=f"{prefix}_type",
                    on_change=keep_view,
                    args=("Scenario lab",),
                ),
                "event_day": st.selectbox(
                    "Event day",
                    contract["categorical"]["event_day"]["allowed_values"],
                    key=f"{prefix}_day",
                    on_change=keep_view,
                    args=("Scenario lab",),
                ),
                "event_time": st.selectbox(
                    "Event time",
                    contract["categorical"]["event_time"]["allowed_values"],
                    key=f"{prefix}_time",
                    on_change=keep_view,
                    args=("Scenario lab",),
                ),
                "travel_distance_km": st.slider(
                    "Travel distance (km)",
                    0.0,
                    30.0,
                    float(min(reference["travel_distance_km"], 30)),
                    0.5,
                    key=f"{prefix}_distance",
                    on_change=keep_view,
                    args=("Scenario lab",),
                ),
            }

    baseline_fields = scenario_controls(base_column, "baseline", "Baseline event")
    proposal_fields = scenario_controls(proposal_column, "proposal", "Proposed event")
    if st.button(
        "Compare scenarios", type="primary", on_click=keep_view, args=("Scenario lab",)
    ):
        shared = {
            "registration_days_before": reference["registration_days_before"],
            "previous_events_registered": int(reference["previous_events_registered"]),
            "previous_events_attended": int(reference["previous_events_attended"]),
            "club_member": reference["club_member"],
        }
        baseline_result = predictor.predict(AttendanceInput(**shared, **baseline_fields))
        proposal_result = predictor.predict(AttendanceInput(**shared, **proposal_fields))
        comparison = pd.DataFrame(
            {
                "Scenario": ["Baseline", "Proposed"],
                "Attendance probability": [
                    baseline_result.attendance_probability,
                    proposal_result.attendance_probability,
                ],
            }
        )
        delta = proposal_result.attendance_probability - baseline_result.attendance_probability
        st.metric("Estimated attendance change", f"{delta * 100:+.1f} points")
        figure = px.bar(
            comparison,
            x="Scenario",
            y="Attendance probability",
            color="Scenario",
            color_discrete_map={"Baseline": SLATE, "Proposed": COBALT},
            range_y=[0, 1],
            text_auto=".1%",
        )
        figure.update_layout(showlegend=False, title="Attendance probability by scenario")
        st.plotly_chart(clean_chart(figure, 320), use_container_width=True)
        st.markdown('<div class="note">Scenario changes do not establish that an event choice causes the modelled difference. Use this view to form hypotheses, not promises.</div>', unsafe_allow_html=True)

with model_tab:
    st.markdown("### Model card")
    st.write("Every number below is loaded from the saved evaluation artifact—nothing is hardcoded into the interface.")
    st.caption(
        f"Measured on the {metrics['dataset']['rows']}-row leakage-safe cohort. "
        f"The shipped model is refit on all {metrics['champion']['deployment_refit_rows']} labelled rows "
        "for prediction only; model choice, threshold, and risk bands are frozen before that refit."
    )
    score_columns = st.columns(4)
    score_columns[0].metric("ROC-AUC", f"{champion_summary['roc_auc']['mean']:.3f}", f"± {champion_summary['roc_auc']['std']:.3f}")
    score_columns[1].metric("Attendance precision", percent(champion_summary["attendance_precision"]["mean"]))
    score_columns[2].metric("Attendance recall", percent(champion_summary["attendance_recall"]["mean"]))
    score_columns[3].metric("No-show F1", f"{champion_summary['no_show_f1']['mean']:.3f}")

    candidates = candidate_frame(metrics).sort_values("ROC-AUC")
    champion_label = (
        f"{metrics['champion']['candidate']} · {metrics['champion']['feature_mode']}"
    )
    candidate_chart = go.Figure(
        go.Bar(
            x=candidates["ROC-AUC"],
            y=candidates["Candidate"],
            orientation="h",
            marker_color=[
                COBALT if label == champion_label else "#A8B6C4"
                for label in candidates["Candidate"]
            ],
            text=candidates["ROC-AUC"].map(lambda value: f"{value:.3f}"),
            textposition="outside",
            hovertemplate="%{y}<br>ROC-AUC %{x:.3f}<extra></extra>",
        )
    )
    candidate_chart.update_layout(
        title="Model comparison · repeated grouped validation",
        xaxis_title="ROC-AUC",
        yaxis_title=None,
        xaxis_range=[0.45, max(0.70, candidates["ROC-AUC"].max() + 0.04)],
    )
    st.plotly_chart(clean_chart(candidate_chart, 410), use_container_width=True)

    raw_forest = candidates.loc[candidates["Candidate"].eq("random_forest · raw")].iloc[0]
    engineered_forest = candidates.loc[
        candidates["Candidate"].eq("random_forest · engineered")
    ].iloc[0]
    st.markdown(
        '<div class="note"><strong>Feature ablation:</strong> the raw-feature random forest '
        f"outperformed its engineered counterpart by "
        f"{raw_forest['ROC-AUC'] - engineered_forest['ROC-AUC']:+.3f} ROC-AUC and "
        f"{raw_forest['Macro-F1'] - engineered_forest['Macro-F1']:+.3f} macro-F1. "
        "Derived fields were tested, not assumed useful, and were excluded from the champion.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Decision diagnostics")
    if diagnostics is None:
        st.warning("Decision diagnostics are missing. Run `uv run turnout-lab train` to rebuild them.")
    else:
        threshold_frame = pd.DataFrame(diagnostics["threshold_curve"])
        threshold_chart = go.Figure()
        threshold_series = [
            ("Attendance precision", "attendance_precision", COBALT, "solid"),
            ("Attendance recall", "attendance_recall", COBALT, "dash"),
            ("No-show precision", "no_show_precision", CORAL, "solid"),
            ("No-show recall", "no_show_recall", CORAL, "dash"),
            ("Macro-F1", "macro_f1", GOLD, "dot"),
        ]
        for label, column, color, dash in threshold_series:
            threshold_chart.add_trace(
                go.Scatter(
                    x=threshold_frame["threshold"],
                    y=threshold_frame[column],
                    mode="lines",
                    name=label,
                    line=dict(color=color, dash=dash, width=3 if label == "Macro-F1" else 2),
                    hovertemplate=f"Threshold %{{x:.2f}}<br>{label} %{{y:.1%}}<extra></extra>",
                )
            )
        threshold_chart.add_vline(
            x=diagnostics["selected_threshold"],
            line_color=INK,
            line_dash="dash",
            annotation_text=f"Selected {diagnostics['selected_threshold']:.2f}",
            annotation_position="top right",
        )
        threshold_chart.update_layout(
            title="Classification policy across thresholds",
            xaxis_title="Attendance decision threshold",
            yaxis_title="Metric value",
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(
            clean_chart(threshold_chart, 430, top_margin=85), use_container_width=True
        )
        st.caption(
            "Higher thresholds make attendance harder to predict, trading attendance recall "
            "for attendance precision and no-show recall. The marker is the final development-only threshold."
        )

        diagnostic_left, diagnostic_right = st.columns(2)
        confusion_values = diagnostics["normalized_confusion_matrix"]
        confusion_chart = go.Figure(
            go.Heatmap(
                z=confusion_values,
                x=["Predicted no-show", "Predicted attend"],
                y=["Actual no-show", "Actual attend"],
                zmin=0,
                zmax=1,
                colorscale=[[0, "#F7FAFC"], [1, COBALT]],
                text=[[f"{value:.1%}" for value in row] for row in confusion_values],
                texttemplate="%{text}",
                hovertemplate="%{y}<br>%{x}<br>%{z:.1%}<extra></extra>",
                colorbar=dict(title="Rate", tickformat=".0%"),
            )
        )
        confusion_chart.update_layout(
            title="Normalized repeated OOF confusion matrix",
            xaxis_title=None,
            yaxis_title=None,
        )
        diagnostic_left.plotly_chart(
            clean_chart(confusion_chart, 360), use_container_width=True
        )

        fold_frame = pd.DataFrame(metrics["calibrated_champion"]["fold_metrics"])
        stability_chart = go.Figure()
        for label, column, color in [
            ("ROC-AUC", "roc_auc", COBALT),
            ("Macro-F1", "macro_f1", CORAL),
            ("Brier ↓", "brier", GOLD),
        ]:
            stability_chart.add_trace(
                go.Box(
                    y=fold_frame[column],
                    name=label,
                    marker_color=color,
                    line_color=color,
                    boxpoints="all",
                    jitter=0.28,
                    pointpos=0,
                    hovertemplate=f"{label} %{{y:.3f}}<extra></extra>",
                )
            )
        stability_chart.update_layout(
            title="Performance across 25 grouped outer folds",
            yaxis_title="Metric value",
            yaxis_range=[0, 1],
            showlegend=False,
        )
        diagnostic_right.plotly_chart(
            clean_chart(stability_chart, 360), use_container_width=True
        )
        st.caption(
            f"Confusion rates and threshold curves use {diagnostics['repeated_oof_predictions']:,} "
            f"predictions: each of {diagnostics['development_rows']} rows is evaluated once for "
            f"each of {len(diagnostics['outer_seeds'])} outer seeds. The matrix applies the final "
            f"{diagnostics['selected_threshold']:.2f} policy uniformly; headline recall averages "
            "fold-local policies. Brier is lower-is-better."
        )

    chart_left, chart_right = st.columns(2)
    calibration = pd.DataFrame(metrics["calibrated_champion"]["calibration_points"])
    calibration_chart = go.Figure()
    calibration_chart.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line=dict(color=SLATE, dash="dash"))
    )
    calibration_chart.add_trace(
        go.Scatter(
            x=calibration["mean_predicted_probability"],
            y=calibration["observed_attendance_rate"],
            mode="lines+markers",
            name="Random forest",
            line=dict(color=COBALT, width=3),
            marker=dict(size=8),
        )
    )
    calibration_chart.update_layout(title="Calibration", xaxis_title="Predicted", yaxis_title="Observed")
    chart_left.plotly_chart(clean_chart(calibration_chart, 350), use_container_width=True)

    importance = pd.DataFrame(metrics["calibrated_champion"]["feature_importance"]).sort_values("importance_mean")
    importance_chart = px.bar(
        importance,
        x="importance_mean",
        y="feature",
        orientation="h",
        error_x="importance_std",
        color_discrete_sequence=[CORAL],
        title="Validation-set permutation importance",
    )
    importance_chart.update_xaxes(title="ROC-AUC decrease after permutation")
    importance_chart.update_yaxes(title=None)
    chart_right.plotly_chart(clean_chart(importance_chart, 350), use_container_width=True)

    with st.expander("Evaluation protocol and selection rule"):
        st.json(metrics["evaluation_protocol"])
        st.write(
            "The official test overlap is excluded before development. Candidate preprocessing and tuning occur inside grouped folds; the champion is then evaluated with sigmoid calibration and fold-local threshold selection."
        )
    st.markdown('<div class="note">Signal is modest. This model ranks registrations for supportive outreach; it does not justify denying access, penalizing students, or claiming individual certainty.</div>', unsafe_allow_html=True)

with data_tab:
    st.markdown("### Data trust and operations")
    audit_cards = st.columns(4)
    audit_cards[0].metric("Raw training rows", quality["raw"]["train_rows"])
    audit_cards[1].metric("Quarantined rows", quality["overlap"]["quarantined_training_rows"])
    audit_cards[2].metric("Exact test matches", quality["overlap"]["exact_id_and_feature_matches"])
    audit_cards[3].metric("Development rows", quality["development"]["rows"])
    st.error(
        "The official test set is not independent: all 100 rows match training identities and features. Matching training rows are quarantined from evaluation, so no metric shown here is measured on them. The shipped model is refit on all labelled rows for prediction only."
    )

    missing = pd.DataFrame(
        {
            "Feature": list(quality["raw"]["train_missing_by_column"].keys()),
            "Missing rows": list(quality["raw"]["train_missing_by_column"].values()),
        }
    ).sort_values("Missing rows")
    missing_chart = px.bar(
        missing,
        x="Missing rows",
        y="Feature",
        orientation="h",
        color_discrete_sequence=[GOLD],
        title="Raw training missingness",
    )
    st.plotly_chart(clean_chart(missing_chart, 390), use_container_width=True)

    st.markdown("#### Descriptive patterns")
    insight_name = st.selectbox(
        "Segment",
        list(metrics["descriptive_insights"].keys()),
        format_func=lambda value: value.replace("_", " ").title(),
        on_change=keep_view,
        args=("Data & operations",),
    )
    insight = pd.DataFrame(metrics["descriptive_insights"][insight_name])
    insight_chart = go.Figure(
        go.Bar(
            x=insight["segment"],
            y=insight["attendance_rate"],
            marker_color=COBALT,
            error_y=dict(
                type="data",
                symmetric=False,
                array=insight["ci_95_high"] - insight["attendance_rate"],
                arrayminus=insight["attendance_rate"] - insight["ci_95_low"],
            ),
            customdata=insight[["n"]],
            hovertemplate="%{x}<br>Attendance %{y:.1%}<br>n=%{customdata[0]}<extra></extra>",
        )
    )
    insight_chart.update_layout(title="Observed attendance with bootstrap 95% intervals", yaxis_tickformat=".0%")
    st.plotly_chart(clean_chart(insight_chart, 360), use_container_width=True)
    st.caption("Descriptive associations only; small segments have wide intervals and should not drive policy.")

    st.markdown("#### Anonymous local operations")
    operations = operations_summary(DATABASE_PATH)
    if not operations["available"]:
        st.warning(
            "Anonymous operations analytics are temporarily unavailable. "
            "Prediction and batch scoring remain fully functional."
        )
    operation_cards = st.columns(4)
    prediction_total = int(operations["predictions"]["total"] or 0)
    average_probability = operations["predictions"]["average_probability"]
    operation_cards[0].metric("Logged predictions", prediction_total)
    operation_cards[1].metric("Average attendance", percent(average_probability))
    operation_cards[2].metric("High-risk scores", int(operations["predictions"]["high_risk"] or 0))
    operation_cards[3].metric("Batch runs", int(operations["batches"]["total"] or 0))
    risk_distribution = pd.DataFrame(operations["risk_distribution"])
    if not risk_distribution.empty:
        risk_order = ["low", "medium", "high"]
        risk_chart = px.bar(
            risk_distribution,
            x="no_show_risk_band",
            y="count",
            color="no_show_risk_band",
            color_discrete_map={"low": COBALT, "medium": GOLD, "high": CORAL},
            category_orders={"no_show_risk_band": risk_order},
            title="Operational scores by no-show risk band",
        )
        risk_chart.update_layout(showlegend=False)
        st.plotly_chart(clean_chart(risk_chart, 320), use_container_width=True)
    else:
        st.info("No scores have been logged yet. Use Predict or Batch score to populate this view.")
    st.markdown('<div class="source-line">Runtime logs contain no student IDs and no raw registration fields.</div>', unsafe_allow_html=True)

st.divider()
st.caption(
    "Turnout Lab uses a challenge-data snapshot. Predictions support reminder prioritization and event planning; they are not causal findings or guarantees of individual behavior."
)
