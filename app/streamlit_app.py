"""Local, read-only consumption risk analytics. No model loading or training."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gridguard.capacity_planning import FINANCIAL_NOTICE, capacity_plan, financial_scenarios
from gridguard.context_dashboard import (
    PROJECT_ROOT,
    context_signature,
    customer_events,
    filter_context,
    inspection_frame,
    load_context_view,
    local_path,
    verified_context_history,
)
from gridguard.dashboard import (
    discover_runs,
    file_signature,
    filter_queue,
    load_anomaly_benchmark,
    load_explainability,
    load_reliability,
    load_run,
    optional_report,
    prediction_metrics,
    private_export,
    quality_groups,
    queue_metrics,
)
from gridguard.data import load_customer_history
from gridguard.decision_gate import STATUSES, _day_end
from gridguard.explainability_metrics import IMPORTANCE_NOTICE
from gridguard.i18n import (
    CATALOG,
    NAVIGATION,
    QUESTIONS,
    diagnostic_text,
    display_label,
    format_count,
    format_number,
    group_for_workspace,
    localize_figure,
    localize_frame,
    metric_help,
    translate,
)
from gridguard.modeling import ranking_metrics
from gridguard.operational import time_value
from gridguard.reliability import ECE_NOTICE

language = st.session_state.get("language", "tr")
st.set_page_config(page_title="GridGuard AI", page_icon=":material/bolt:", layout="wide")


def tr(message, **values):
    return translate(message, language, **values)


def label(value):
    return display_label(value, language)


def option_label(value):
    return str(label(value))


def diagnostic(message, *, warning=False):
    (st.warning if warning else st.caption)(diagnostic_text(message, language))
    if message not in CATALOG:
        with st.expander(tr("Original diagnostic details")):
            st.code(message, language=None)


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def read_artifacts(folder, signature):
    run = load_run(folder)
    return run, prediction_metrics(run.predictions)


@st.cache_data(ttl=300, max_entries=32, show_spinner=False)
def read_optional(path, modified):
    return optional_report(path)


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def customer_history(path, customer_id, modified):
    return load_customer_history(path, customer_id)


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def read_capacity_curve(predictions):
    capacities = np.linspace(0, len(predictions), min(len(predictions) + 1, 101), dtype=int)
    return capacity_plan(predictions, capacities)


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def read_reliability(path, signature):
    return load_reliability(path)


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def read_anomaly(path, signature):
    return load_anomaly_benchmark(path)


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def read_explainability(path, signature):
    return load_explainability(path)


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def read_context(decision_dir, context_dir, root, predictions_path, signature):
    return load_context_view(
        decision_dir, context_dir, root=root, predictions_path=predictions_path or None
    )


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def context_history(path, customer_id, expected_sha256, signature):
    return verified_context_history(path, customer_id, expected_sha256)


def number(value, percent=False):
    return format_number(value, language, percent=percent)


def cards(items):
    with st.container(horizontal=True):
        for label, value in items:
            st.metric(
                display_label(label, language),
                display_label(value, language),
                help=metric_help(label, language),
                border=True,
            )


def chart(figure, title):
    figure = localize_figure(figure, language)
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#DCE5F1",
        colorway=["#38BDF8", "#2DD4BF", "#FBBF24", "#FB7185"],
        margin=dict(l=16, r=16, t=48, b=24),
        height=350,
        title=dict(text=title, font_size=16),
        legend_title_text="",
    )
    figure.update_xaxes(gridcolor="#253449")
    figure.update_yaxes(gridcolor="#253449")
    st.plotly_chart(figure, width="stretch", theme=None, alt=title, config={"displaylogo": False})


def table(frame, label):
    if len(frame) > 200:
        page_number = st.number_input(
            tr("Page"),
            min_value=1,
            max_value=(len(frame) + 199) // 200,
            value=1,
            key=f"table_page_{label}",
        )
        st.caption(
            tr(
                "Showing {shown} of {total} rows.",
                shown=min(200, len(frame) - (page_number - 1) * 200),
                total=len(frame),
            )
        )
        frame = frame.iloc[(page_number - 1) * 200 : page_number * 200]
    st.dataframe(
        localize_frame(frame, language),
        hide_index=True,
        width="stretch",
        alt=label,
        column_config={column: display_label(column, language) for column in frame},
    )


def optional(path):
    path = Path(path)
    result, warning = read_optional(str(path), path.stat().st_mtime_ns if path.is_file() else None)
    if warning:
        diagnostic(warning)
    return result


language = st.sidebar.segmented_control(
    "Dil / Language",
    ["tr", "en"],
    default=language,
    required=True,
    format_func=str.upper,
    key="language",
)
st.sidebar.title(tr(":material/bolt: GridGuard AI"))
st.sidebar.caption(tr("CONSUMPTION INTELLIGENCE"))
initial_workspace = os.environ.get("GRIDGUARD_WORKSPACE", "Overview")
groups = list(NAVIGATION)
section = st.sidebar.radio(
    tr("Workspace"),
    groups,
    format_func=lambda key: tr(f"navigation.{key}"),
    index=groups.index(group_for_workspace(initial_workspace)),
    key="section",
)
with st.sidebar.expander(tr("Experiment source")):
    options = ["Custom path", *discover_runs()]
    selected = st.selectbox(
        tr("Saved experiment"),
        options,
        format_func=option_label,
        key="ui_saved_experiment",
    )
    default = os.environ.get("GRIDGUARD_ARTIFACT_DIR", "artifacts")
    artifact_dir = (
        st.text_input(
            tr("Artifact directory"),
            value=default,
            key="ui_artifact_directory",
        )
        if selected == "Custom path"
        else selected
    )
st.sidebar.caption(tr("Local artifacts · Read-only analytics"))
st.title(tr(f"navigation.{section}"))
st.caption(tr(QUESTIONS[section]))
views = NAVIGATION[section]
page = (
    views[0]
    if len(views) == 1
    else st.segmented_control(
        tr("Analysis view"),
        views,
        default=initial_workspace if initial_workspace in views else views[0],
        required=True,
        format_func=option_label,
        key=f"view_{section}",
        persist_state="session",
    )
)
if page == "Operational Context":
    st.warning(
        tr(
            "SYNTHETIC operational context only. Even verified records are simulated. Decisions support human review; they do not establish innocence, fraud or improved ML performance."
        )
    )
    root = os.environ.get("GRIDGUARD_CONTEXT_ROOT", str(PROJECT_ROOT))
    with st.expander(tr("Local context sources"), expanded=False):
        decision_dir = st.text_input(
            tr("Saved gate output directory"),
            os.environ.get("GRIDGUARD_CONTEXT_DECISIONS", "reports/context-sgcc-phase1"),
            key="context_decision_dir",
        )
        context_dir = st.text_input(
            tr("SYNTHETIC operational dataset directory"),
            os.environ.get("GRIDGUARD_CONTEXT_DIR", "artifacts/context-sgcc-phase1"),
            key="context_dataset_dir",
        )
        predictions_path = st.text_input(
            tr("Original prediction CSV (optional integrity check)"),
            os.environ.get("GRIDGUARD_CONTEXT_PREDICTIONS", ""),
            key="context_prediction_path",
        )
        st.caption(
            tr("Local files within the server-configured data root. No uploads or file changes.")
        )
    try:
        signature = context_signature(
            decision_dir, context_dir, root=root, predictions_path=predictions_path or None
        )
        with st.spinner(tr("Validating saved SYNTHETIC context…")):
            context = read_context(decision_dir, context_dir, root, predictions_path, signature)
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        st.info(
            tr(
                "No valid saved operational context is available. Select local directories containing context_decisions.csv, context_evaluation.json and the SYNTHETIC datasets."
            )
        )
        st.caption(
            tr(
                "Check file schemas, fingerprints and the evaluation window. Other workspaces remain available."
            )
        )
        st.stop()
    for warning in context.warnings:
        diagnostic(warning, warning=True)
    evaluation = context.report
    counts = evaluation["customers_per_decision_status"]
    cards(
        [
            (
                "Total evaluated customers",
                format_count(evaluation["total_customers_evaluated"], language),
            ),
            ("Eligible for review", format_count(counts["eligible_for_review"], language)),
            (
                "Needs context verification",
                format_count(counts["needs_context_verification"], language),
            ),
            ("Insufficient data", format_count(counts["insufficient_data"], language)),
        ]
    )
    with st.expander(tr("Coverage and high-risk details"), expanded=False):
        cards(
            [
                (
                    "High-risk requiring verification",
                    format_count(
                        evaluation["high_risk_customers_requiring_verification"], language
                    ),
                ),
                (
                    "Context coverage",
                    number(
                        evaluation["context_coverage"].get("subscription_window_coverage_rate"),
                        True,
                    ),
                ),
            ]
        )
    st.caption(tr("These are synthetic region assignments, not actual network geography."))
    st.caption(
        tr(
            "SYNTHETIC gate window: {p0} to {p1} · Available as of {p2}. Cards describe the complete saved cohort. Context coverage is subscription coverage of the entire window; outage/meter absence remains unknown.",
            p0=label(evaluation["window_start"]),
            p1=label(evaluation["window_end"]),
            p2=label(evaluation["evaluation_date"]),
        )
    )
    st.caption(
        tr(
            "High risk: original score ≥ {p0:g}/100. Missing-event policy: {p1}. Historical ML predictions are unchanged and are not prospective scores at this cutoff.",
            p0=label(evaluation["rules"]["high_risk_threshold"]),
            p1=label(evaluation["rules"]["missing_event_policy"]),
        )
    )
    with st.container(border=True):
        st.subheader(tr("Inspect SYNTHETIC context decisions"))
        risk_range = st.slider(tr("Original ML risk score"), 0, 100, (0, 100), key="context_risk")
        with st.expander(tr("Advanced filters"), expanded=False):
            statuses = st.multiselect(
                tr("Decision status"),
                list(STATUSES),
                default=list(STATUSES),
                key="context_status",
                format_func=option_label,
            )
            event_types = st.multiselect(
                tr("Matched event type"),
                sorted(context.events.event_type.unique()),
                key="context_event_type",
                format_func=option_label,
            )
            verifications = st.multiselect(
                tr("Event verification status"),
                ["verified", "unverified", "missing"],
                key="context_verification",
                format_func=option_label,
            )
            detailed_columns = st.checkbox(tr("Detailed evidence columns"), key="context_details")
        with st.container(horizontal=True):
            include_missing = st.checkbox(
                tr("Include missing or invalid risk scores"), True, key="context_missing_risk"
            )
            reveal_ids = st.checkbox(
                tr("Reveal customer and event IDs in this local view"),
                False,
                key="context_reveal_ids",
            )
        search = (
            st.text_input(tr("Find customer ID locally"), key="context_search")
            if reveal_ids
            else ""
        )
        st.caption(
            tr(
                "Empty event/verification filters include all customers. Combined filters must match the same event. Unknown coverage is not proof that no event occurred. IDs are hidden until explicitly revealed."
            )
        )
        filtered = filter_context(
            context,
            risk_range=risk_range,
            statuses=statuses,
            event_types=event_types,
            verifications=verifications,
            search=search,
            include_missing_scores=include_missing,
        )
        st.caption(
            tr(
                "{p0:,} of {p1:,} customers match. Saved input order is retained.",
                p0=label(len(filtered)),
                p1=label(len(context.decisions)),
            )
        )
        if filtered.empty:
            st.info(tr("No customers match these context filters."))
            st.stop()
        view_key = str(hash((signature, tuple(filtered.index), reveal_ids)))
        page_number = st.number_input(
            tr("Inspection page"),
            min_value=1,
            max_value=max(1, (len(filtered) + 199) // 200),
            value=1,
            step=1,
            key=f"context_page_{view_key}",
        )
        visible = filtered.iloc[(page_number - 1) * 200 : page_number * 200]
        browser_frame = localize_frame(
            inspection_frame(visible, context.events, reveal_ids=reveal_ids),
            language,
            anonymous=not reveal_ids,
        )
        if not detailed_columns:
            browser_frame = browser_frame.drop(columns="reason_codes")
        selection = st.dataframe(
            browser_frame,
            hide_index=True,
            width="stretch",
            column_config={
                "customer_id": label("Customer ID" if reveal_ids else "Customer (ID hidden)"),
                "original_risk_score": st.column_config.NumberColumn(tr("Original ML risk score")),
                "decision_status": tr("Decision status"),
                "evidence_status": tr("Evidence status"),
                "provenance": label("provenance"),
                "reason_codes": st.column_config.ListColumn(tr("Reason codes")),
                "matched_operational_events": st.column_config.ListColumn(
                    tr("Matched operational events")
                ),
            },
            on_select="rerun",
            selection_mode="single-row",
            key=f"context_table_{view_key}_{page_number}",
            alt=label(
                "SYNTHETIC context decisions; customer identifiers hidden unless explicitly revealed"
            ),
        )
        selected_rows = selection.selection.rows
        index = selected_rows[0] if selected_rows and selected_rows[0] < len(visible) else 0
        chosen_index = st.selectbox(
            tr("Customer to inspect"),
            visible.index.tolist(),
            index=index,
            format_func=lambda value: str(browser_frame.loc[value, "customer_id"]),
            key=f"context_customer_{view_key}_{page_number}_{index}",
        )
    chosen = context.decisions.loc[chosen_index]
    chosen_score = pd.to_numeric(pd.Series([chosen.original_risk_score]), errors="coerce").iloc[0]
    score_label = (
        str(chosen.original_risk_score)
        if np.isfinite(chosen_score) and 0 <= chosen_score <= 100
        else "Unavailable"
    )
    st.subheader(tr("Customer context drill-down · SYNTHETIC"))
    cards(
        [
            ("Original risk score / 100", score_label),
            ("Gate decision", chosen.decision_status),
            ("Evidence status", chosen.evidence_status),
        ]
    )
    st.caption(" · ".join(label(code) for code in chosen.reason_codes))
    with st.expander(tr("Technical details"), expanded=False):
        st.code(
            "\n".join([chosen.decision_status, chosen.evidence_status, *chosen.reason_codes]),
            language=None,
        )
    selected_events = customer_events(context, chosen.customer_id, reveal_ids=reveal_ids)
    resolved = len(selected_events)
    st.caption(
        tr(
            "{p0} of {p1} saved matched events available. All periods below are SYNTHETIC.",
            p0=label(resolved),
            p1=label(len(chosen.matched_event_ids)),
        )
    )
    if selected_events.empty:
        st.info(
            tr(
                "Operational evidence is unavailable or no records matched. Context remains unknown."
            )
        )
    else:
        timeline = px.timeline(
            selected_events,
            x_start="start",
            x_end="end",
            y="event_type",
            color="verification_status",
            color_discrete_map={
                "verified": "#2DD4BF",
                "unverified": "#FBBF24",
                "missing": "#FB7185",
            },
            labels={
                "event_type": "SYNTHETIC event",
                "verification_status": "Synthetic verification",
            },
        )
        timeline.update_yaxes(autorange="reversed")
        if st.checkbox(tr("Focus timeline on the gate window"), True, key="context_focus_window"):
            timeline.update_xaxes(
                range=[time_value(evaluation["window_start"]), _day_end(evaluation["window_end"])]
            )
        chart(timeline, tr("SYNTHETIC outages, meter events and subscription timeline"))
        table(
            selected_events, tr("SYNTHETIC matched operational evidence for the selected customer")
        )
        st.caption(
            tr(
                "Timeline endpoints are exclusive; subscription and meter end dates include their full source day."
            )
        )
    with st.form("context_history_form"):
        raw_path = st.text_input(
            tr("SGCC source CSV for consumption history"),
            os.environ.get("GRIDGUARD_CONTEXT_SOURCE", "data/consumption.csv"),
            key="context_source_path",
        )
        submitted = st.form_submit_button(tr("Load source consumption history"))
    if submitted:
        try:
            path = local_path(raw_path, root, suffix=".csv")
            history = context_history(
                str(path),
                chosen.customer_id,
                evaluation["source_sha256"],
                file_signature(path.parent, names=(path.name,)),
            )
            if history.empty:
                st.info(tr("The selected customer has no history in this source."))
            else:
                figure = px.line(
                    history, x="date", y="kwh", labels={"kwh": "Daily consumption (kWh)"}
                )
                for event in selected_events.itertuples():
                    if event.event_type not in ("active", "terminated"):
                        figure.add_vrect(
                            x0=event.start,
                            x1=event.end,
                            fillcolor="#FBBF24",
                            opacity=0.15,
                            line_width=0,
                        )
                chart(figure, tr("Source consumption · shaded periods are SYNTHETIC events"))
                chart(
                    px.scatter(
                        history.assign(
                            availability=history.kwh.isna().map(
                                {True: "Missing / invalid", False: "Observed"}
                            )
                        ),
                        x="date",
                        y="availability",
                    ),
                    tr("Source reading availability · gaps preserved"),
                )
                st.caption(
                    tr(
                        "Source SHA-256 verified. Readings are unchanged; SYNTHETIC overlap does not establish a causal explanation."
                    )
                )
        except (OSError, ValueError, KeyError, TypeError):
            st.info(
                tr(
                    "History unavailable. Select the exact local source CSV recorded by this context run."
                )
            )
    st.stop()

if page == "Model Explainability":
    path = Path(
        st.text_input(
            tr("Saved explainability report"),
            value=os.environ.get(
                "GRIDGUARD_EXPLAINABILITY_REPORT", str(Path(artifact_dir) / "explainability.json")
            ),
            key="explain_path",
        )
    )
    try:
        explanation = read_explainability(
            str(path),
            file_signature(
                path.parent,
                names=(
                    path.name,
                    "split_manifest.json",
                    "explainability_predictions.csv",
                    "permutation_repeats.json",
                ),
            ),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.info(
            tr(
                "No valid saved explainability report is available. Run the optional explain workflow first."
            )
        )
        diagnostic(str(exc))
        st.stop()
    st.warning(
        tr(
            "Development-only analysis of a fresh fixed RF, not the historical final model. Importance and error analysis use separate customer cohorts. The inspected final holdout was excluded; independent final performance is not established."
        )
    )
    if explanation["dataset"].get("source") == "synthetic":
        st.warning(
            tr(
                "Synthetic demonstration; simulated outcomes do not establish operational performance."
            )
        )
    st.caption(
        tr(
            "Scores are uncalibrated ranking signals. Outcomes refer to historical positive labels, not confirmed theft. Label noise, inspection selection and missing event dates limit interpretation. These are global associations, not individual or causal explanations."
        )
    )
    importance, errors = explanation["importance"], explanation["errors"]
    cards(
        [
            ("Importance customers", str(importance["customers"])),
            ("Error evaluation customers", str(errors["customers"])),
            ("Importance cohort AP", number(importance["baseline_average_precision"])),
            ("Error cohort AP", number(errors["ranking"]["average_precision"])),
        ]
    )
    st.subheader(tr("Global permutation importance"))
    importance_frame = pd.DataFrame(importance["features"]).sort_values(
        "mean_decrease", ascending=False, na_position="last"
    )
    table(importance_frame, tr("AP permutation importance, standard deviation and negative means"))
    if importance["status"] == "available":
        chart(
            px.bar(
                importance_frame,
                x="mean_decrease",
                y="feature",
                orientation="h",
                error_x="std_decrease",
                color="negative_mean",
                labels={"mean_decrease": "Mean AP decrease", "feature": "Original feature"},
            ),
            tr("AP permutation importance: mean with repeat standard deviation"),
        )
    else:
        st.info(tr("Permutation importance unavailable: {p0}.", p0=label(importance["status"])))
    st.caption(tr(IMPORTANCE_NOTICE))
    st.caption(
        tr(
            "Saved repeats: {p0}; seed: {p1}. One repeat yields zero empirical SD, not certainty.",
            p0=label(importance["n_repeats"]),
            p1=label(importance["seed"]),
        )
    )
    with st.expander(tr("Threshold, group and distribution details"), expanded=False):
        st.subheader(tr("Errors at predeclared thresholds"))
        thresholds = explanation["configuration"]["thresholds"]
        threshold = st.selectbox(
            tr("Saved decision threshold"),
            thresholds,
            index=thresholds.index(0.5),
            key="explain_threshold",
            format_func=option_label,
        )
        current = next(row for row in errors["thresholds"] if row["threshold"] == threshold)
        cards(
            [
                (key.replace("_", " ").title(), str(current[key]))
                for key in ("true_positive", "false_positive", "true_negative", "false_negative")
            ]
        )
        if current["confusion_matrix"] is not None:
            chart(
                go.Figure(
                    go.Heatmap(
                        z=current["confusion_matrix"],
                        x=["Predicted negative", "Predicted positive"],
                        y=["Observed negative", "Observed positive"],
                        texttemplate="%{z}",
                        showscale=False,
                    )
                ),
                tr("Confusion matrix at saved threshold {p0:g}", p0=label(threshold)),
            )
        table(
            pd.DataFrame(
                [
                    {key: value for key, value in row.items() if key != "confusion_matrix"}
                    for row in errors["thresholds"]
                ]
            ),
            tr("Precision, recall and errors at fixed thresholds"),
        )
        table(
            pd.DataFrame(
                [
                    {"requested_k": k, **values}
                    for k, values in errors["ranking"]["ranking_at_k"].items()
                ]
            ),
            tr("Error cohort Precision and Recall at predefined inspection capacities"),
        )
        st.caption(
            tr(
                "Threshold equality is positive. Only saved thresholds/capacities are shown; no threshold optimization occurs. Undefined rates remain unavailable."
            )
        )
        st.subheader(tr("Data-quality error associations at primary threshold 0.5"))
        fixed = errors["fixed_error_analysis"]
        if fixed:
            grouped = fixed["data_quality_groups"]
            grouping = st.selectbox(
                tr("Predefined error groups"),
                list(grouped),
                key="explain_groups",
                format_func=option_label,
            )
            group_rows = []
            for name, group in grouped[grouping].get("groups", {}).items():
                reliable = group["rate_status"] == "available"
                group_rows.append(
                    {
                        "group": name,
                        "customers": group["customers"],
                        "positives": group.get("positives"),
                        "false_positive": group.get("false_positive"),
                        "false_negative": group.get("false_negative"),
                        "false_positive_rate": group.get("false_positive_rate")
                        if reliable
                        else None,
                        "false_negative_rate": group.get("false_negative_rate")
                        if reliable
                        else None,
                        "rate_status": group["rate_status"],
                    }
                )
            group_frame = pd.DataFrame(group_rows)
            table(
                group_frame,
                tr("Group support, false positives, false negatives and rate reliability"),
            )
            if not group_frame.empty:
                chart(
                    px.bar(
                        group_frame,
                        x="group",
                        y=["false_positive_rate", "false_negative_rate"],
                        barmode="group",
                    ),
                    tr("Error rates by predefined group at threshold 0.5"),
                )
        st.caption(
            tr(
                "Groups always use primary threshold 0.5, independently of the threshold selector above. Small groups below {p0} are suppressed; rates need at least five positives and negatives. Counts are descriptive; missingness can reflect outages or collection problems. Group differences do not establish causal data-quality effects.",
                p0=label(errors["min_group_size"]),
            )
        )
        st.subheader(tr("Score and feature distributions by outcome"))
        feature = st.selectbox(
            tr("Original feature"),
            importance_frame.feature.tolist(),
            key="explain_feature",
            format_func=option_label,
        )
        for selected_feature in ("risk_score", feature):
            distributions = [
                row
                for row in errors["outcome_distributions"]
                if row["threshold"] == threshold and row["feature"] == selected_feature
            ]
            bins = []
            for row in distributions:
                if row["distribution"]:
                    bins.extend(
                        {
                            "outcome": row["outcome"],
                            "value": item["lower"] / 2 + item["upper"] / 2,
                            "fraction": item["count"] / row["observed_values"],
                        }
                        for item in row["distribution"]["histogram"]
                    )
            if bins:
                chart(
                    px.bar(
                        pd.DataFrame(bins),
                        x="value",
                        y="fraction",
                        color="outcome",
                        barmode="overlay",
                        opacity=0.55,
                    ),
                    tr(
                        "{p0}: binned distributions at threshold {p1:g}",
                        p0=label(selected_feature),
                        p1=label(threshold),
                    ),
                )
            table(
                pd.DataFrame(
                    [
                        {key: value for key, value in row.items() if key != "distribution"}
                        for row in distributions
                    ]
                ),
                tr(
                    "{p0} outcome counts, observed values and suppression status",
                    p0=label(selected_feature),
                ),
            )
        st.caption(
            tr(
                "Shared bins compare fractions of observed values within each outcome, not outcome prevalence. Features remain un-imputed in diagnostics; missing counts stay visible. Small observed cohorts have no value distributions. Risk score is rounded 100 times the raw ranking score, not a calibrated probability."
            )
        )
        st.subheader(tr("Configuration and provenance"))
        table(
            pd.DataFrame(
                [
                    {"Property": key, "Value": str(value)}
                    for key, value in {
                        "experiment": explanation["experiment_id"],
                        "source": explanation["dataset"]["source"],
                        "dataset fingerprint": explanation["dataset"]["fingerprint_sha256"],
                        "feature configuration": explanation["feature_config"],
                        "model parameters": explanation["model_parameters"],
                        "partitions": explanation["split_sizes"],
                        "configuration": explanation["configuration"],
                        "methodology": explanation["methodology"],
                        "software": explanation["software"],
                    }.items()
                ]
            ),
            tr("Explainability methodology, partition roles, seeds and software"),
        )
        for warning in errors["warnings"]:
            diagnostic(warning)
        st.caption(
            tr(
                "Saved aggregate diagnostics only. No customer identifiers, model loading, fitting or model rescoring in this workspace."
            )
        )
    st.stop()

if page == "Anomaly Detection":
    path = Path(
        st.text_input(
            tr("Saved anomaly benchmark"),
            value=os.environ.get(
                "GRIDGUARD_ANOMALY_REPORT", str(Path(artifact_dir) / "anomaly_benchmark.json")
            ),
            key="anomaly_path",
        )
    )
    try:
        anomaly = read_anomaly(
            str(path),
            file_signature(
                path.parent, names=(path.name, "split_manifest.json", "anomaly_predictions.csv")
            ),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.info(
            tr(
                "No valid saved anomaly benchmark is available. Run the optional benchmark workflow first."
            )
        )
        diagnostic(str(exc))
        st.stop()
    st.warning(
        tr(
            "Development-only benchmark. Independent final performance is not established. The inspected final holdout was excluded. The RF comparator is newly fitted, not the historical published model. No method or threshold was selected on evaluation outcomes."
        )
    )
    if anomaly["dataset"].get("source") == "synthetic":
        st.warning(
            tr(
                "Synthetic demonstration; simulated labels do not establish operational detection quality."
            )
        )
    st.caption(
        tr(
            "Anomaly scores are not probabilities or confirmed theft evidence. Unusual consumption can reflect outages, missing readings or collection errors. Disagreement does not establish that either method is wrong. Historical customer-level labels lack event dates; human review is required."
        )
    )
    partition = st.selectbox(
        tr("Saved development cohort"),
        ["development_evaluation", "development_validation"],
        key="anomaly_partition",
        format_func=option_label,
    )
    evaluation = anomaly["evaluations"][partition]
    cards(
        [
            ("Development customers", str(evaluation["customers"])),
            ("Observed positive labels", str(evaluation["positive_customers"])),
            ("Training customers", str(anomaly["split_sizes"]["train"])),
            ("Excluded holdout customers", str(anomaly["split_sizes"]["excluded_holdout"])),
        ]
    )
    for warning in evaluation["warnings"]:
        diagnostic(warning)
    rows, rankings = [], []
    for method, metrics in evaluation["methods"].items():
        rows.append(
            {
                "method": method,
                "AP": metrics["average_precision"],
                "ROC-AUC": metrics["roc_auc"],
                "distinct scores": metrics["distribution"]["distinct_scores"],
            }
        )
        rankings.extend(
            {"method": method, "requested_k": k, **values}
            for k, values in metrics["ranking_at_k"].items()
        )
    summary = pd.DataFrame(rows)
    table(summary, tr("Supervised versus unsupervised development discrimination"))
    chart(
        px.bar(summary, x="method", y=["AP", "ROC-AUC"], barmode="group"),
        tr("Development discrimination comparison"),
    )
    table(pd.DataFrame(rankings), tr("Development Precision, Recall and Lift at predeclared K"))
    with st.expander(tr("Comparison details"), expanded=False):
        st.subheader(tr("Top-K overlap and complementarity"))
        correlations, overlap = [], []
        for pair in evaluation["complementarity"]:
            correlations.append(
                {
                    "left": pair["left_method"],
                    "right": pair["right_method"],
                    "Spearman rho": pair["spearman_rho"],
                }
            )
            for row in pair["top_k"]:
                overlap.append(
                    {
                        "left": pair["left_method"],
                        "right": pair["right_method"],
                        **{
                            key: value
                            for key, value in row.items()
                            if key not in ("left_only", "right_only")
                        },
                        "left_only_customers": row["left_only"]["customers"],
                        "right_only_customers": row["right_only"]["customers"],
                        "left_only_positive_rate": row["left_only"]["positive_rate"],
                        "right_only_positive_rate": row["right_only"]["positive_rate"],
                        "left_rate_status": row["left_only"]["status"],
                        "right_rate_status": row["right_only"]["status"],
                    }
                )
        table(pd.DataFrame(correlations), tr("Tie-aware Spearman rank correlation"))
        table(pd.DataFrame(overlap), tr("Top-K intersections, Jaccard and nonoverlap label rates"))
        st.caption(
            tr(
                "For RF comparisons, right-only customers are anomaly-only and left-only customers are supervised-only. Spearman uses average score ranks, without ID tie breaking; constant scores have undefined correlation. Top-K ties use customer ID. Small nonoverlap groups have unavailable label rates. No ensemble is fitted."
            )
        )
        st.subheader(tr("Native score distributions"))
        for method, metrics in evaluation["methods"].items():
            histogram = pd.DataFrame(metrics["distribution"]["histogram"])
            if not histogram.empty:
                histogram["score"] = histogram.lower / 2 + histogram.upper / 2
                chart(
                    px.bar(histogram, x="score", y="count"),
                    tr("{p0}: native score distribution", p0=label(method)),
                )
        st.caption(
            tr(
                "Each method has its own score units and axis. Absolute score magnitudes are not comparable across methods."
            )
        )
        st.subheader(tr("Missing-reading-rate groups"))
        group_rows = []
        for group in evaluation["missing_reading_groups"]:
            for method in evaluation["methods"]:
                metrics = group["methods"][method] if group["methods"] else {}
                group_rows.append(
                    {
                        "group": group["group"],
                        "customers": group["customers"],
                        "positive_customers": group["positive_customers"],
                        "status": group["status"],
                        "method": method,
                        "AP": metrics.get("average_precision"),
                        "ROC-AUC": metrics.get("roc_auc"),
                    }
                )
        table(
            pd.DataFrame(group_rows),
            tr("Development discrimination by fixed missing-reading-rate groups"),
        )
        st.caption(
            tr(
                "Existing missing-rate bins are reused. Group discrimination needs at least {p0} customers and {p1} per class. These are descriptive safeguards, not power calculations, fairness conclusions or formal privacy guarantees. Null metrics are unavailable.",
                p0=label(evaluation["min_group_size"]),
                p1=label(evaluation["min_class_support"]),
            )
        )
        st.subheader(tr("Configuration and provenance"))
        table(
            pd.DataFrame(
                [
                    {"Property": key, "Value": str(value)}
                    for key, value in {
                        "experiment": anomaly["experiment_id"],
                        "source": anomaly["dataset"]["source"],
                        "dataset fingerprint": anomaly["dataset"]["fingerprint_sha256"],
                        "feature configuration": anomaly["feature_config"],
                        "partitions": anomaly["split_sizes"],
                        "model parameters": anomaly["model_parameters"],
                        "score definitions": anomaly["score_definitions"],
                        "methodology": anomaly["methodology"],
                        "software": anomaly["software"],
                    }.items()
                ]
            ),
            tr("Fixed method configurations, score definitions and provenance"),
        )
        st.caption(
            tr(
                "This view reads verified saved aggregates, exposes no customer identifiers, and never loads or trains a model."
            )
        )
    st.stop()

if page == "Model Reliability":
    path = Path(
        st.text_input(
            tr("Saved reliability report"),
            value=os.environ.get(
                "GRIDGUARD_RELIABILITY_REPORT", str(Path(artifact_dir) / "reliability.json")
            ),
            key="reliability_path",
        )
    )
    try:
        reliability = read_reliability(
            str(path),
            file_signature(
                path.parent, names=(path.name, "split_manifest.json", "calibration_predictions.csv")
            ),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.info(
            tr(
                "No valid saved reliability evaluation is available. Run the optional calibration workflow first."
            )
        )
        diagnostic(str(exc))
        st.stop()
    st.warning(
        tr(
            "Development-only calibration diagnostics. Independent final performance is not established. The previously inspected final holdout was excluded from this experiment."
        )
    )
    if reliability["dataset"].get("source") == "synthetic":
        st.warning(
            tr(
                "Synthetic demonstration; these diagnostics do not establish real-world reliability."
            )
        )
    st.caption(
        tr(
            "Calibrated outputs estimate the dataset's historical positive-label frequency under modeling assumptions. They are not confirmed theft probabilities. Dataset shift, label quality and population representativeness can invalidate calibration; independent human review is required."
        )
    )
    selected_method = reliability["selected_method"]
    methods = reliability["methods"]
    cards(
        [
            ("Selected mapping", selected_method),
            (
                "Development Brier score",
                number(methods[selected_method]["development_evaluation"]["brier_score"]),
            ),
            (
                "Development log loss",
                number(methods[selected_method]["development_evaluation"]["log_loss"]),
            ),
            ("Development ECE", number(methods[selected_method]["development_evaluation"]["ece"])),
        ]
    )
    st.caption(
        tr(
            "The saved mapping was selected by Brier score on calibration_selection, before diagnostic evaluation. No fitting occurs in this view."
        )
    )
    summaries = pd.DataFrame(
        [
            {
                "method": method,
                **{
                    key: metrics["development_evaluation"][key]
                    for key in (
                        "customers",
                        "brier_score",
                        "log_loss",
                        "ece",
                        "average_precision",
                        "roc_auc",
                        "distinct_probabilities",
                    )
                },
            }
            for method, metrics in methods.items()
        ]
    )
    table(summaries, tr("Development-only probability and discrimination comparison"))
    chart(
        px.bar(summaries, x="method", y="brier_score"),
        tr("Development Brier score comparison (lower is better)"),
    )
    diagram = go.Figure()
    diagram.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Reference diagonal", line_dash="dash")
    )
    bins = []
    ranking = []
    for method, result in methods.items():
        metrics = result["development_evaluation"]
        populated = [item for item in metrics["bins"] if item["count"]]
        diagram.add_trace(
            go.Scatter(
                x=[item["mean_probability"] for item in populated],
                y=[item["positive_fraction"] for item in populated],
                mode="lines+markers",
                name=method,
                text=[f"Customers: {item['count']}" for item in populated],
                hovertemplate="Mean probability %{x:.3f}<br>Positive-label fraction %{y:.3f}<br>%{text}<extra>%{fullData.name}</extra>",
            )
        )
        bins.extend({"method": method, **item} for item in metrics["bins"])
        ranking.extend(
            {"method": method, "requested_k": k, **values}
            for k, values in metrics["ranking_at_k"].items()
        )
        for warning in metrics["warnings"]:
            st.caption(tr("{p0}: {p1}", p0=label(method), p1=label(warning)))
    diagram.update_xaxes(title="Mean predicted positive-label probability", range=[0, 1])
    diagram.update_yaxes(title="Observed positive-label fraction", range=[0, 1])
    chart(diagram, tr("Development reliability diagram"))
    with st.expander(tr("Calibration details"), expanded=False):
        table(pd.DataFrame(bins), tr("Fixed reliability bins, including counts and empty bins"))
        st.caption(tr(ECE_NOTICE))
        st.subheader(tr("Development ranking comparison"))
        table(pd.DataFrame(ranking), tr("Precision, recall and lift before and after calibration"))
        st.caption(
            tr(
                "AP and ROC-AUC appear above. Isotonic can introduce ties; sigmoid can also change rankings through reversal or numerical saturation. Actual probabilities and customer-ID tie breaks are used throughout."
            )
        )
        st.subheader(tr("Saved methodology and provenance"))
        table(
            pd.DataFrame(
                [
                    {"Property": key, "Value": str(value)}
                    for key, value in {
                        "experiment": reliability["experiment_id"],
                        "base model": reliability["selected_model"],
                        "feature configuration": reliability["feature_config"],
                        "source": reliability["dataset"]["source"],
                        "dataset fingerprint": reliability["dataset"]["fingerprint_sha256"],
                        "partitions": reliability["split_sizes"],
                        "methodology": reliability["methodology"],
                        "software": reliability["software"],
                        "skipped methods": reliability["skipped_methods"],
                    }.items()
                ]
            ),
            tr("Calibration methods, partition roles, seed and software metadata"),
        )
    st.stop()

try:
    run, computed = read_artifacts(artifact_dir, file_signature(artifact_dir))
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.info(
        tr(
            "Select a local experiment containing metrics.json and, for holdout runs, test_predictions.csv."
        )
    )
    diagnostic(str(exc))
    st.stop()

report, predictions = run.report, run.predictions
dataset = report.get("dataset", {})
validation_only = report.get("test") is None
validation = report.get("validation", {})
evaluation = (
    validation.get(report.get("selected_model"), {})
    if validation_only
    else computed or report.get("test", {})
)
st.caption(
    tr(
        "{p0}  |  {p1}  |  {p2}",
        p0=label(run.kind),
        p1=label(report.get("experiment_id", Path(artifact_dir).name)),
        p2=label(report.get("selected_model", "Model unavailable").replace("_", " ")),
    )
)
if validation_only:
    st.info(
        tr(
            "This is a validation-only experiment. No test predictions or customer queue are available."
        )
    )
if dataset.get("source") == "synthetic":
    st.warning(
        tr(
            "Synthetic demonstration. Simulated readings and labels do not measure real-world performance."
        )
    )
st.caption(
    tr(
        "Scores are uncalibrated ranking signals, not evidence of electricity theft. Retrospective metrics do not predict future inspection outcomes."
    )
)
for warning in run.warnings:
    diagnostic(warning, warning=True)

if page == "Overview":
    cards(
        [
            (
                "Validation AP" if validation_only else "Observed holdout AP",
                number(evaluation.get("average_precision")),
            ),
            ("ROC-AUC", number(evaluation.get("roc_auc"))),
            (
                "Customers in experiment",
                format_count(dataset["customers"], language)
                if "customers" in dataset
                else "Unavailable",
            ),
            ("Missing readings · experiment", number(dataset.get("missing_reading_rate"), True)),
        ]
    )
    if not predictions.empty:
        st.info(
            tr(
                "Start with {count} high-risk customers in this historical cohort. Operational context uses a separate bundle.",
                count=int(predictions.risk_score.ge(80).sum()),
            )
        )
        left, right = st.columns(2)
        with left, st.container(border=True):
            bands = predictions.risk_band.value_counts().reindex(
                ["Low", "Medium", "High"], fill_value=0
            )
            chart(
                px.bar(
                    x=bands.index,
                    y=bands.values,
                    color=bands.index,
                    color_discrete_map={"Low": "#38BDF8", "Medium": "#FBBF24", "High": "#FB7185"},
                    labels={"x": "Illustrative band", "y": "Customers"},
                ),
                tr("Risk distribution"),
            )
        with right, st.container(border=True):
            chart(
                px.histogram(
                    predictions,
                    x="model_score",
                    nbins=30,
                    labels={"model_score": "Raw model score"},
                ),
                tr("Score distribution"),
            )
    with st.expander(tr("Experiment context"), expanded=False):
        table(
            pd.DataFrame(
                [
                    {"Property": key.replace("_", " "), "Value": str(value)}
                    for key, value in {
                        "source": dataset.get("source", "Unavailable"),
                        "feature_configuration": report.get("feature_config", "baseline (legacy)"),
                        "reading_columns": dataset.get("reading_days", "Unavailable"),
                        "split_sizes": report.get("split_sizes", "Unavailable"),
                        "fitted_on": report.get("methodology", {}).get("fitted_on", "Not recorded"),
                        "provenance": run.provenance,
                    }.items()
                ]
            ),
            tr("Experiment context"),
        )
        st.caption(
            tr(
                "Historical customer-level labels lack event onset dates. The reported holdout has been inspected; this is retrospective analysis, not a prospective detection study. Bands (high ≥80, medium ≥50) are illustrative. Human verification is required."
            )
        )

elif page == "Model performance":
    cards(
        [
            ("Average precision", number(evaluation.get("average_precision"))),
            ("ROC-AUC", number(evaluation.get("roc_auc"))),
            ("Precision · threshold 0.5", number(evaluation.get("precision_0_5"), True)),
            ("Recall · threshold 0.5", number(evaluation.get("recall_0_5"), True)),
        ]
    )
    st.caption(
        tr(
            "F1 at 0.5: {p0}. Fixed illustrative threshold; no tuning here.",
            p0=label(number(evaluation.get("f1_0_5"))),
        )
    )
    if evaluation.get("confusion_matrix") is not None:
        chart(
            go.Figure(
                go.Heatmap(
                    z=evaluation["confusion_matrix"],
                    x=["Predicted negative", "Predicted positive"],
                    y=["Actual negative", "Actual positive"],
                    colorscale="Teal",
                    texttemplate="%{z}",
                    showscale=False,
                )
            ),
            tr("Confusion matrix · threshold 0.5"),
        )
    if not predictions.empty and "true_label" in predictions:
        rows = [
            {
                "K requested": k,
                **ranking_metrics(
                    predictions.true_label.to_numpy(),
                    predictions.model_score.to_numpy(),
                    k,
                    customer_ids=predictions.customer_id,
                ),
            }
            for k in (50, 100, 200)
        ]
        ranking = pd.DataFrame(rows)
        table(ranking, tr("Retrospective ranking metrics"))
        chart(
            px.line(
                ranking,
                x="evaluated_k",
                y=["precision", "recall"],
                markers=True,
                labels={"evaluated_k": "Actual inspection capacity", "value": "Fraction"},
            ),
            tr("Retrospective precision and recall at capacity"),
        )
        chart(px.bar(ranking, x="K requested", y="lift"), tr("Lift over cohort prevalence"))
        st.caption(
            tr(
                "K is capped by available customers. Recall uses all positives in the displayed cohort; lift divides precision by cohort prevalence. Calculated from saved unrounded scores."
            )
        )
    with st.expander(tr("Technical details"), expanded=False):
        st.subheader(tr("Validation comparison"))
        comparison = (
            pd.DataFrame.from_dict(validation, orient="index").rename_axis("model").reset_index()
        )
        columns = [
            col
            for col in (
                "model",
                "average_precision",
                "roc_auc",
                "precision_at_50",
                "recall_0_5",
                "f1_0_5",
            )
            if col in comparison
        ]
        table(comparison[columns], tr("Validation model comparison"))
        if "average_precision" in comparison:
            chart(
                px.bar(comparison, x="model", y="average_precision"),
                tr("Validation average precision"),
            )
        st.caption(
            tr(
                "Selection uses validation AP. Holdout analysis here neither trains nor tunes a model."
            )
        )

elif page == "Inspection queue":
    if predictions.empty:
        st.info(tr("Customer predictions are unavailable for this run."))
    else:
        with st.container(border=True):
            capacity = st.number_input(
                tr("Inspection capacity (K)"),
                min_value=1,
                max_value=len(predictions),
                value=min(50, len(predictions)),
                step=1,
                key="ui_inspection_capacity_k",
            )
            bands = st.multiselect(
                tr("Risk bands"),
                ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
                format_func=option_label,
                key="ui_risk_bands",
            )
            search = st.text_input(
                tr("Search customer ID"),
                help=label("Literal, case-insensitive local search"),
                key="ui_search_customer_id",
            )
        queue = filter_queue(predictions, capacity=capacity, bands=bands, search=search)
        metrics = queue_metrics(queue, predictions)
        cards(
            [
                ("Customers in queue", str(len(queue))),
                ("Retrospective precision", number(metrics["precision"], True)),
                ("Recall of cohort positives", number(metrics["recall"], True)),
                ("Lift over cohort prevalence", number(metrics["lift"])),
            ]
        )
        st.caption(
            tr(
                "Eligibility filters apply before capacity. Global rank is retained. Raw score descending; ties by customer ID ascending. Filtered queue metrics are not unfiltered Precision@K."
            )
        )
        if queue.empty:
            st.info(tr("No customers match these filters."))
        else:
            table(
                queue[
                    [
                        col
                        for col in (
                            "rank",
                            "customer_id",
                            "risk_score",
                            "model_score",
                            "risk_band",
                            "review_signal",
                        )
                        if col in queue
                    ]
                ],
                tr("Local inspection queue"),
            )
            include = st.checkbox(
                tr("Include customer IDs in private local CSV"),
                value=False,
                key="ui_include_customer_ids_in_private_local_csv",
            )
            st.download_button(
                tr("Download displayed queue"),
                private_export(queue, include_ids=include),
                "inspection_candidates.csv",
                "text/csv",
                icon=":material/download:",
            )
            st.caption(
                tr(
                    "Export omits ground-truth labels and reading indicators. Without IDs it is not anonymous: ranks and scores may still link records. Keep all exports private."
                )
            )

elif page == "Inspection Capacity Planning":
    if predictions.empty:
        st.info(tr("Inspection capacity planning requires saved customer predictions."))
    else:
        st.info(
            tr(
                "Retrospective decision support on the full selected cohort. Historical positive labels are observed outcomes, not proof of theft or future inspection success. Choose capacity from operational constraints; do not tune models on this holdout."
            )
        )
        st.caption(
            tr(
                "Cohort: {p0:,} customers | Provenance: {p1}",
                p0=label(len(predictions)),
                p1=label(run.provenance),
            )
        )
        capacity = st.slider(
            tr("Inspection capacity (K)"),
            min_value=0,
            max_value=len(predictions),
            value=min(50, len(predictions)),
            step=1,
            key="planning_capacity",
        )
        comparison = st.text_input(
            tr("Comparison capacities (comma-separated)"),
            "0, 50, 100, 200",
            max_chars=400,
            help=label("Nonnegative whole numbers. Requests above cohort size are capped."),
            key="planning_comparison",
        )
        try:
            capacities = [int(value.strip()) for value in comparison.split(",") if value.strip()]
            scenarios = capacity_plan(predictions, [capacity, *capacities])
        except ValueError:
            st.error(
                tr("Enter comma-separated nonnegative whole-number capacities, such as 0, 50, 100.")
            )
            st.stop()
        current = scenarios.loc[scenarios.requested_k.eq(capacity)].iloc[0]
        cards(
            [
                ("Retrospective Precision@K", number(current.precision, True)),
                ("Retrospective Recall@K", number(current.recall, True)),
                ("Retrospective Lift@K", number(current.lift)),
                ("Observed positives in top K", number(current.positives_found)),
                ("Random expected positives", number(current.random_expected_positives)),
            ]
        )
        if "true_label" in predictions:
            curve = read_capacity_curve(predictions)
            left, right = st.columns(2)
            for container, metric, baseline in (
                (left, "precision", "random_precision"),
                (right, "recall", "random_recall"),
            ):
                with container, st.container(border=True):
                    figure = px.line(
                        curve.rename(
                            columns={
                                metric: "Model ranking",
                                baseline: "Random selection (expected)",
                            }
                        ),
                        x="evaluated_k",
                        y=["Model ranking", "Random selection (expected)"],
                        labels={"evaluated_k": "Inspections (K)", "value": f"{metric.title()}@K"},
                    )
                    figure.update_yaxes(range=[0, 1])
                    figure.add_vline(x=capacity, line_dash="dot")
                    chart(figure, tr("Retrospective {p0}@K", p0=label(metric.title())))
            figure = px.line(
                curve.rename(
                    columns={
                        "positives_found": "Model ranking (observed)",
                        "random_expected_positives": "Random selection (expected)",
                    }
                ),
                x="evaluated_k",
                y=["Model ranking (observed)", "Random selection (expected)"],
                labels={"evaluated_k": "Inspections (K)", "value": "Positive labels"},
            )
            figure.add_vline(x=capacity, line_dash="dot")
            chart(figure, tr("Retrospective model selection versus random baseline"))
        else:
            st.info(
                tr(
                    "Ground-truth labels are unavailable. Outcome metrics, random baselines and financial scenarios cannot be calculated from raw scores."
                )
            )
        st.caption(
            tr(
                "Random selection means uniform sampling without replacement from this same cohort: expected positives = actual K × observed prevalence. It is an expectation, not a simulated outcome. Curves sample at most 101 capacities; scenario rows use exact K. At K=0, precision and lift are undefined. With no positives, recall and lift are undefined."
            )
        )
        st.subheader(tr("Operational scenario comparison"))
        table(
            scenarios[
                [
                    "requested_k",
                    "evaluated_k",
                    "precision",
                    "recall",
                    "lift",
                    "positives_found",
                    "random_expected_positives",
                    "marginal_inspections",
                    "marginal_positives",
                    "marginal_yield",
                ]
            ].rename(
                columns={
                    "requested_k": "Requested K",
                    "evaluated_k": "Actual K",
                    "precision": "Precision@K",
                    "recall": "Recall@K",
                    "lift": "Lift@K",
                    "positives_found": "Observed positives",
                    "random_expected_positives": "Random expected positives",
                    "marginal_inspections": "Additional inspections",
                    "marginal_positives": "Additional observed positives",
                    "marginal_yield": "Marginal yield",
                }
            ),
            tr("Retrospective operational capacity scenarios"),
        )
        st.caption(
            tr(
                "Rows are sorted by requested K. Marginal yield is additional observed positives / additional inspections since the previous row (the first starts at zero). Capped requests can have identical actual K and undefined marginal yield. This view uses the full cohort; Inspection queue eligibility filters do not apply."
            )
        )
        with st.expander(tr("Illustrative financial scenario"), expanded=False):
            st.subheader(tr("Illustrative financial scenario"))
            financial = st.checkbox(
                tr("Enable illustrative financial calculations"),
                value=False,
                disabled="true_label" not in predictions,
                key="planning_financial",
            )
            if financial and "true_label" in predictions:
                st.warning(tr(FINANCIAL_NOTICE))
                st.caption(
                    tr(
                        "Use the same currency units for both inputs. Every actual inspection incurs the entered cost. Each observed positive is assigned the hypothetical recovery value. Random selection uses the same capacity and cost. Net = recovery minus inspection costs; other costs, collection rates and timing are excluded."
                    )
                )
                cost = st.number_input(
                    tr("Cost per inspection (currency units)"),
                    min_value=0.0,
                    value=0.0,
                    key="planning_cost",
                )
                recovery = st.number_input(
                    tr("Hypothetical recovery per observed positive (currency units)"),
                    min_value=0.0,
                    value=0.0,
                    key="planning_recovery",
                )
                try:
                    finances = financial_scenarios(
                        scenarios, inspection_cost=cost, recovery_per_positive=recovery
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    table(
                        finances.drop(
                            columns=["positives_found", "random_expected_positives", "notice"]
                        ).rename(
                            columns={
                                "requested_k": "Requested K",
                                "evaluated_k": "Actual K",
                                "inspection_cost_total": "Total inspection cost",
                                "hypothetical_model_recovery": "Hypothetical model recovery",
                                "hypothetical_random_recovery": "Hypothetical random recovery",
                                "illustrative_model_net": "Illustrative model net",
                                "illustrative_random_net": "Illustrative random net",
                            }
                        ),
                        tr("Illustrative financial scenarios, not validated financial forecasts"),
                    )
        st.subheader(tr("Prioritized inspection candidates"))
        # load_run has already validated and ordered the full cohort by raw score and ID.
        queue = predictions.head(capacity)
        table(
            queue[["rank", "customer_id", "model_score", "risk_score", "risk_band"]],
            tr("Selected top-K inspection candidates without ground-truth labels"),
        )
        include = st.checkbox(
            tr("Include customer IDs in private local CSV"), value=False, key="planning_include_ids"
        )
        st.download_button(
            tr("Download capacity inspection candidates"),
            private_export(queue, include_ids=include),
            "inspection_candidates.csv",
            "text/csv",
            icon=":material/download:",
        )
        st.caption(
            tr(
                "Raw score descending, customer ID ascending for ties; rounded risk scores do not determine order. CSVs always omit labels and reading signals. IDs are excluded by default. Ranks and scores can still link records; keep exports private. A ranking alone never confirms electricity theft."
            )
        )

elif page == "Customer analytics":
    if predictions.empty:
        st.info(tr("Individual customer analytics require saved customer predictions."))
    else:
        search = st.text_input(
            tr("Find a customer"),
            help=label("Search locally to narrow the first 100 matches"),
            key="ui_find_a_customer",
        )
        matches = predictions.loc[
            predictions.customer_id.str.contains(search, case=False, regex=False)
        ].head(100)
        if matches.empty:
            st.info(tr("No matching customer."))
        else:
            chosen_id = st.selectbox(
                tr("Customer"),
                matches.customer_id.tolist(),
                key="ui_customer",
            )
            chosen = matches.loc[matches.customer_id == chosen_id].iloc[0]
            cards(
                [
                    ("Risk score / 100", str(chosen.risk_score)),
                    ("Illustrative band", chosen.risk_band),
                    ("Raw score", f"{chosen.model_score:.6f}"),
                    ("Global rank", str(chosen["rank"])),
                ]
            )
            st.markdown(tr("**Descriptive reading signals**"))
            st.write(label(chosen.get("review_signal", "No saved descriptive signals")))
            st.caption(
                tr("Heuristic summaries, not model feature attribution or causal explanations.")
            )
            with st.form("history"):
                raw_path = st.text_input(
                    tr("Original CSV on this machine"),
                    dataset.get("input_path", ""),
                    key="ui_original_csv_on_this_machine",
                )
                submitted = st.form_submit_button(tr("Load consumption history"))
            if submitted:
                try:
                    path = Path(raw_path)
                    history = customer_history(str(path), chosen_id, path.stat().st_mtime_ns)
                    if history.empty:
                        st.info(tr("Customer not found in this file."))
                    else:
                        observed = history.kwh.notna().sum()
                        cards(
                            [
                                ("Calendar days", str(len(history))),
                                (
                                    "Missing or invalid days",
                                    number(float(history.kwh.isna().mean()), True),
                                ),
                                (
                                    "Zero readings / observed",
                                    number(
                                        float(history.kwh.eq(0).sum() / observed)
                                        if observed
                                        else None,
                                        True,
                                    ),
                                ),
                            ]
                        )
                        chart(
                            px.line(
                                history,
                                x="date",
                                y="kwh",
                                labels={"kwh": "Daily consumption (kWh)"},
                            ),
                            tr("Consumption history · gaps preserved"),
                        )
                        chart(
                            px.scatter(
                                history.assign(
                                    status=history.kwh.isna().map(
                                        {True: "Missing / invalid", False: "Observed"}
                                    )
                                ),
                                x="date",
                                y="status",
                            ),
                            tr("Reading availability"),
                        )
                        st.caption(
                            tr(
                                "Negative, nonnumeric and nonfinite values become missing; calendar gaps remain missing. The selected CSV is a local lookup, not verified against the experiment fingerprint."
                            )
                        )
                except (OSError, ValueError) as exc:
                    st.warning(tr("History unavailable: {p0}", p0=label(exc)))

elif page == "Data quality":
    st.subheader(tr("Experiment sample"))
    cards(
        [
            ("Missing-reading fraction", number(dataset.get("missing_reading_rate"), True)),
            ("Positive label fraction", number(dataset.get("positive_rate"), True)),
            ("Missing calendar days", str(dataset.get("missing_calendar_days", "Unavailable"))),
        ]
    )
    if "positive_customers" in dataset and "customers" in dataset:
        chart(
            px.bar(
                x=["Negative label", "Positive label"],
                y=[
                    dataset["customers"] - dataset["positive_customers"],
                    dataset["positive_customers"],
                ],
                labels={"x": "Historical label", "y": "Customers"},
            ),
            tr("Experiment label distribution"),
        )
    if "missing_ratio" in predictions:
        chart(
            px.histogram(predictions, x="missing_ratio", nbins=20),
            tr("Holdout missing-reading distribution"),
        )
    st.subheader(tr("Optional source-data audit"))
    quality_path = st.text_input(
        tr("Preflight report JSON"),
        "reports/data-quality.json",
        key="ui_preflight_report_json",
    )
    quality = optional(quality_path)
    if quality:
        st.caption(
            tr(
                "Source-file audit, not necessarily the experiment sample. Compare source and cohort metadata before use."
            )
        )
        keys = [
            "customers",
            "reading_columns",
            "missing_reading_rate",
            "zero_readings",
            "negative_readings_masked_in_training",
            "nonnumeric_nonempty_cells",
            "nonfinite_readings",
            "missing_calendar_days",
        ]
        table(
            pd.DataFrame(
                [
                    {
                        "Measure": key.replace("_", " "),
                        "Value": str(quality.get(key, "Not recorded")),
                    }
                    for key in keys
                ]
            ),
            tr("Source data quality audit"),
        )
    st.subheader(tr("Aggregate quality groups"))
    report_dir = st.text_input(
        tr("Aggregate report directory"),
        str(Path("reports") / Path(artifact_dir).name),
        key="ui_aggregate_report_directory",
    )
    analysis_path = (
        Path(artifact_dir) / "validation_error_analysis.json"
        if validation_only
        else Path(report_dir) / "error_analysis.json"
    )
    st.caption(
        tr(
            "{p0} groups · {p1}",
            p0=label("Validation" if validation_only else "Holdout"),
            p1=label(analysis_path),
        )
    )
    analysis = optional(analysis_path)
    feature = st.selectbox(
        tr("Group by"),
        ["missing_ratio", "coefficient_variation", "mean_kwh"],
        format_func=option_label,
        key="ui_group_by",
    )
    groups = quality_groups(analysis, feature)
    if groups.empty:
        st.info(tr("No aggregate group metrics available for this selection."))
    else:
        table(groups, tr("Aggregate quality group metrics"))
        if "average_precision" in groups:
            chart(
                px.bar(groups, x="group", y="average_precision"),
                tr("Average precision by quality group"),
            )
        st.caption(
            tr(
                "Suppressed groups have no rates. Null metrics are undefined, not zero. AP varies with prevalence; group differences do not establish causality or fairness. No customer records are loaded from these reports."
            )
        )
