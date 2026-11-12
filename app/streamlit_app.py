"""Local, read-only consumption risk analytics. No model loading or training."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gridguard.capacity_planning import FINANCIAL_NOTICE, capacity_plan, financial_scenarios
from gridguard.dashboard import (
    discover_runs,
    file_signature,
    filter_queue,
    load_run,
    optional_report,
    prediction_metrics,
    private_export,
    quality_groups,
    queue_metrics,
)
from gridguard.data import load_customer_history
from gridguard.modeling import ranking_metrics

st.set_page_config(
    page_title="GridGuard AI | Risk analytics", page_icon=":material/bolt:", layout="wide"
)


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


def number(value, percent=False):
    return (
        "Unavailable"
        if value is None or pd.isna(value)
        else (f"{value:.1%}" if percent else f"{value:,.3f}")
    )


def cards(items):
    with st.container(horizontal=True):
        for label, value in items:
            st.metric(label, value, border=True)


def chart(figure, title):
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
    st.dataframe(frame, hide_index=True, width="stretch", alt=label)


def optional(path):
    path = Path(path)
    result, warning = read_optional(str(path), path.stat().st_mtime_ns if path.is_file() else None)
    if warning:
        st.caption(warning)
    return result


st.sidebar.title(":material/bolt: GridGuard AI")
st.sidebar.caption("CONSUMPTION INTELLIGENCE")
options = ["Custom path", *discover_runs()]
selected = st.sidebar.selectbox("Saved experiment", options)
default = os.environ.get("GRIDGUARD_ARTIFACT_DIR", "artifacts")
artifact_dir = (
    st.sidebar.text_input("Artifact directory", value=default)
    if selected == "Custom path"
    else selected
)
page = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "Model performance",
        "Inspection queue",
        "Inspection Capacity Planning",
        "Customer analytics",
        "Data quality",
    ],
)
st.sidebar.caption("Local artifacts · Read-only analytics")
st.title(page)
st.caption("GridGuard AI / Electricity consumption risk analytics")
try:
    run, computed = read_artifacts(artifact_dir, file_signature(artifact_dir))
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.info(
        "Select a local experiment containing metrics.json and, for holdout runs, test_predictions.csv."
    )
    st.caption(str(exc))
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
    f"{run.kind}  |  {report.get('experiment_id', Path(artifact_dir).name)}  |  "
    f"{report.get('selected_model', 'Model unavailable').replace('_', ' ')}"
)
if validation_only:
    st.info(
        "This is a validation-only experiment. No test predictions or customer queue are available."
    )
if dataset.get("source") == "synthetic":
    st.warning(
        "Synthetic demonstration. Simulated readings and labels do not measure real-world performance."
    )
st.caption(
    "Scores are uncalibrated ranking signals, not evidence of electricity theft. "
    "Retrospective metrics do not predict future inspection outcomes."
)
for warning in run.warnings:
    st.warning(warning)

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
                f"{dataset.get('customers', 0):,}" if "customers" in dataset else "Unavailable",
            ),
            ("Missing readings · experiment", number(dataset.get("missing_reading_rate"), True)),
        ]
    )
    if not predictions.empty:
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
                "Risk distribution",
            )
        with right, st.container(border=True):
            chart(
                px.histogram(
                    predictions,
                    x="model_score",
                    nbins=30,
                    labels={"model_score": "Raw model score"},
                ),
                "Score distribution",
            )
    with st.container(border=True):
        st.subheader("Experiment context")
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
            "Experiment context",
        )
        st.caption(
            "Historical customer-level labels lack event onset dates. The reported holdout has been "
            "inspected; this is retrospective analysis, not a prospective detection study. "
            "Bands (high ≥80, medium ≥50) are illustrative. Human verification is required."
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
        f"F1 at 0.5: {number(evaluation.get('f1_0_5'))}. Fixed illustrative threshold; no tuning here."
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
            "Confusion matrix · threshold 0.5",
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
        table(ranking, "Retrospective ranking metrics")
        chart(
            px.line(
                ranking,
                x="evaluated_k",
                y=["precision", "recall"],
                markers=True,
                labels={"evaluated_k": "Actual inspection capacity", "value": "Fraction"},
            ),
            "Retrospective precision and recall at capacity",
        )
        chart(px.bar(ranking, x="K requested", y="lift"), "Lift over cohort prevalence")
        st.caption(
            "K is capped by available customers. Recall uses all positives in the displayed cohort; "
            "lift divides precision by cohort prevalence. Calculated from saved unrounded scores."
        )
    st.subheader("Validation comparison")
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
    table(comparison[columns], "Validation model comparison")
    if "average_precision" in comparison:
        chart(px.bar(comparison, x="model", y="average_precision"), "Validation average precision")
    st.caption(
        "Selection uses validation AP. Holdout analysis here neither trains nor tunes a model."
    )

elif page == "Inspection queue":
    if predictions.empty:
        st.info("Customer predictions are unavailable for this run.")
    else:
        with st.container(border=True):
            capacity = st.number_input(
                "Inspection capacity (K)",
                min_value=1,
                max_value=len(predictions),
                value=min(50, len(predictions)),
                step=1,
            )
            bands = st.multiselect(
                "Risk bands", ["High", "Medium", "Low"], default=["High", "Medium", "Low"]
            )
            search = st.text_input(
                "Search customer ID", help="Literal, case-insensitive local search"
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
            "Eligibility filters apply before capacity. Global rank is retained. Raw score descending; "
            "ties by customer ID ascending. Filtered queue metrics are not unfiltered Precision@K."
        )
        if queue.empty:
            st.info("No customers match these filters.")
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
                "Local inspection queue",
            )
            include = st.checkbox("Include customer IDs in private local CSV", value=False)
            st.download_button(
                "Download displayed queue",
                private_export(queue, include_ids=include),
                "inspection_candidates.csv",
                "text/csv",
                icon=":material/download:",
            )
            st.caption(
                "Export omits ground-truth labels and reading indicators. Without IDs it is not anonymous: "
                "ranks and scores may still link records. Keep all exports private."
            )

elif page == "Inspection Capacity Planning":
    if predictions.empty:
        st.info("Inspection capacity planning requires saved customer predictions.")
    else:
        st.info(
            "Retrospective decision support on the full selected cohort. Historical positive labels "
            "are observed outcomes, not proof of theft or future inspection success. "
            "Choose capacity from operational constraints; do not tune models on this holdout."
        )
        st.caption(f"Cohort: {len(predictions):,} customers | Provenance: {run.provenance}")
        capacity = st.slider(
            "Inspection capacity (K)",
            min_value=0,
            max_value=len(predictions),
            value=min(50, len(predictions)),
            step=1,
            key="planning_capacity",
        )
        comparison = st.text_input(
            "Comparison capacities (comma-separated)",
            "0, 50, 100, 200",
            max_chars=400,
            help="Nonnegative whole numbers. Requests above cohort size are capped.",
            key="planning_comparison",
        )
        try:
            capacities = [int(value.strip()) for value in comparison.split(",") if value.strip()]
            scenarios = capacity_plan(predictions, [capacity, *capacities])
        except ValueError:
            st.error(
                "Enter comma-separated nonnegative whole-number capacities, such as 0, 50, 100."
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
                    chart(figure, f"Retrospective {metric.title()}@K")
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
            chart(figure, "Retrospective model selection versus random baseline")
        else:
            st.info(
                "Ground-truth labels are unavailable. Outcome metrics, random baselines and "
                "financial scenarios cannot be calculated from raw scores."
            )
        st.caption(
            "Random selection means uniform sampling without replacement from this same cohort: "
            "expected positives = actual K × observed prevalence. It is an expectation, not a "
            "simulated outcome. Curves sample at most 101 capacities; scenario rows use exact K. "
            "At K=0, precision and lift are undefined. With no positives, recall and lift are undefined."
        )
        st.subheader("Operational scenario comparison")
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
            "Retrospective operational capacity scenarios",
        )
        st.caption(
            "Rows are sorted by requested K. Marginal yield is additional observed positives / "
            "additional inspections since the previous row (the first starts at zero). "
            "Capped requests can have identical actual K and undefined marginal yield. "
            "This view uses the full cohort; Inspection queue eligibility filters do not apply."
        )
        st.subheader("Illustrative financial scenario")
        financial = st.checkbox(
            "Enable illustrative financial calculations",
            value=False,
            disabled="true_label" not in predictions,
            key="planning_financial",
        )
        if financial and "true_label" in predictions:
            st.warning(FINANCIAL_NOTICE)
            st.caption(
                "Use the same currency units for both inputs. Every actual inspection incurs the "
                "entered cost. Each observed positive is assigned the hypothetical recovery value. "
                "Random selection uses the same capacity and cost. Net = recovery minus inspection "
                "costs; other costs, collection rates and timing are excluded."
            )
            cost = st.number_input(
                "Cost per inspection (currency units)",
                min_value=0.0,
                value=0.0,
                key="planning_cost",
            )
            recovery = st.number_input(
                "Hypothetical recovery per observed positive (currency units)",
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
                    "Illustrative financial scenarios, not validated financial forecasts",
                )
        st.subheader("Prioritized inspection candidates")
        # load_run has already validated and ordered the full cohort by raw score and ID.
        queue = predictions.head(capacity)
        table(
            queue[["rank", "customer_id", "model_score", "risk_score", "risk_band"]],
            "Selected top-K inspection candidates without ground-truth labels",
        )
        include = st.checkbox(
            "Include customer IDs in private local CSV", value=False, key="planning_include_ids"
        )
        st.download_button(
            "Download capacity inspection candidates",
            private_export(queue, include_ids=include),
            "inspection_candidates.csv",
            "text/csv",
            icon=":material/download:",
        )
        st.caption(
            "Raw score descending, customer ID ascending for ties; rounded risk scores do not "
            "determine order. CSVs always omit labels and reading signals. IDs are excluded by "
            "default. Ranks and scores can still link records; keep exports private. "
            "A ranking alone never confirms electricity theft."
        )

elif page == "Customer analytics":
    if predictions.empty:
        st.info("Individual customer analytics require saved customer predictions.")
    else:
        search = st.text_input(
            "Find a customer", help="Search locally to narrow the first 100 matches"
        )
        matches = predictions.loc[
            predictions.customer_id.str.contains(search, case=False, regex=False)
        ].head(100)
        if matches.empty:
            st.info("No matching customer.")
        else:
            chosen_id = st.selectbox("Customer", matches.customer_id.tolist())
            chosen = matches.loc[matches.customer_id == chosen_id].iloc[0]
            cards(
                [
                    ("Risk score / 100", str(chosen.risk_score)),
                    ("Illustrative band", chosen.risk_band),
                    ("Raw score", f"{chosen.model_score:.6f}"),
                    ("Global rank", str(chosen["rank"])),
                ]
            )
            st.markdown("**Descriptive reading signals**")
            st.write(chosen.get("review_signal", "No saved descriptive signals"))
            st.caption("Heuristic summaries, not model feature attribution or causal explanations.")
            with st.form("history"):
                raw_path = st.text_input(
                    "Original CSV on this machine", dataset.get("input_path", "")
                )
                submitted = st.form_submit_button("Load consumption history")
            if submitted:
                try:
                    path = Path(raw_path)
                    history = customer_history(str(path), chosen_id, path.stat().st_mtime_ns)
                    if history.empty:
                        st.info("Customer not found in this file.")
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
                            "Consumption history · gaps preserved",
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
                            "Reading availability",
                        )
                        st.caption(
                            "Negative, nonnumeric and nonfinite values become missing; calendar gaps remain missing. "
                            "The selected CSV is a local lookup, not verified against the experiment fingerprint."
                        )
                except (OSError, ValueError) as exc:
                    st.warning(f"History unavailable: {exc}")

elif page == "Data quality":
    st.subheader("Experiment sample")
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
            "Experiment label distribution",
        )
    if "missing_ratio" in predictions:
        chart(
            px.histogram(predictions, x="missing_ratio", nbins=20),
            "Holdout missing-reading distribution",
        )
    st.subheader("Optional source-data audit")
    quality_path = st.text_input("Preflight report JSON", "reports/data-quality.json")
    quality = optional(quality_path)
    if quality:
        st.caption(
            "Source-file audit, not necessarily the experiment sample. Compare source and cohort metadata before use."
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
            "Source data quality audit",
        )
    st.subheader("Aggregate quality groups")
    report_dir = st.text_input(
        "Aggregate report directory", str(Path("reports") / Path(artifact_dir).name)
    )
    analysis_path = (
        Path(artifact_dir) / "validation_error_analysis.json"
        if validation_only
        else Path(report_dir) / "error_analysis.json"
    )
    st.caption(f"{'Validation' if validation_only else 'Holdout'} groups · {analysis_path}")
    analysis = optional(analysis_path)
    feature = st.selectbox("Group by", ["missing_ratio", "coefficient_variation", "mean_kwh"])
    groups = quality_groups(analysis, feature)
    if groups.empty:
        st.info("No aggregate group metrics available for this selection.")
    else:
        table(groups, "Aggregate quality group metrics")
        if "average_precision" in groups:
            chart(
                px.bar(groups, x="group", y="average_precision"),
                "Average precision by quality group",
            )
        st.caption(
            "Suppressed groups have no rates. Null metrics are undefined, not zero. AP varies with prevalence; "
            "group differences do not establish causality or fairness. No customer records are loaded from these reports."
        )
