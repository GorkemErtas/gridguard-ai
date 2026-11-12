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
    load_anomaly_benchmark,
    load_reliability,
    load_run,
    optional_report,
    prediction_metrics,
    private_export,
    quality_groups,
    queue_metrics,
)
from gridguard.data import load_customer_history
from gridguard.modeling import ranking_metrics
from gridguard.reliability import ECE_NOTICE

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


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def read_reliability(path, signature):
    return load_reliability(path)


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def read_anomaly(path, signature):
    return load_anomaly_benchmark(path)


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
        "Model Reliability",
        "Anomaly Detection",
        "Inspection queue",
        "Inspection Capacity Planning",
        "Customer analytics",
        "Data quality",
    ],
)
st.sidebar.caption("Local artifacts · Read-only analytics")
st.title(page)
st.caption("GridGuard AI / Electricity consumption risk analytics")
if page == "Anomaly Detection":
    path = Path(
        st.text_input(
            "Saved anomaly benchmark",
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
            "No valid saved anomaly benchmark is available. Run the optional benchmark workflow first."
        )
        st.caption(str(exc))
        st.stop()
    st.warning(
        "Development-only benchmark. Independent final performance is not established. "
        "The inspected final holdout was excluded. The RF comparator is newly fitted, "
        "not the historical published model. No method or threshold was selected on evaluation outcomes."
    )
    if anomaly["dataset"].get("source") == "synthetic":
        st.warning(
            "Synthetic demonstration; simulated labels do not establish operational detection quality."
        )
    st.caption(
        "Anomaly scores are not probabilities or confirmed theft evidence. Unusual consumption can "
        "reflect outages, missing readings or collection errors. Disagreement does not establish that "
        "either method is wrong. Historical customer-level labels lack event dates; human review is required."
    )
    partition = st.selectbox(
        "Saved development cohort",
        ["development_evaluation", "development_validation"],
        key="anomaly_partition",
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
        st.caption(warning)
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
    table(summary, "Supervised versus unsupervised development discrimination")
    chart(
        px.bar(summary, x="method", y=["AP", "ROC-AUC"], barmode="group"),
        "Development discrimination comparison",
    )
    table(pd.DataFrame(rankings), "Development Precision, Recall and Lift at predeclared K")
    st.subheader("Top-K overlap and complementarity")
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
    table(pd.DataFrame(correlations), "Tie-aware Spearman rank correlation")
    table(pd.DataFrame(overlap), "Top-K intersections, Jaccard and nonoverlap label rates")
    st.caption(
        "For RF comparisons, right-only customers are anomaly-only and left-only customers are "
        "supervised-only. Spearman uses average score ranks, without ID tie breaking; constant "
        "scores have undefined correlation. Top-K ties use customer ID. Small nonoverlap groups "
        "have unavailable label rates. No ensemble is fitted."
    )
    st.subheader("Native score distributions")
    for method, metrics in evaluation["methods"].items():
        histogram = pd.DataFrame(metrics["distribution"]["histogram"])
        if not histogram.empty:
            histogram["score"] = histogram.lower / 2 + histogram.upper / 2
            chart(px.bar(histogram, x="score", y="count"), f"{method}: native score distribution")
    st.caption(
        "Each method has its own score units and axis. Absolute score magnitudes are not comparable across methods."
    )
    st.subheader("Missing-reading-rate groups")
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
        pd.DataFrame(group_rows), "Development discrimination by fixed missing-reading-rate groups"
    )
    st.caption(
        f"Existing missing-rate bins are reused. Group discrimination needs at least {evaluation['min_group_size']} "
        f"customers and {evaluation['min_class_support']} per class. These are descriptive safeguards, not "
        "power calculations, fairness conclusions or formal privacy guarantees. Null metrics are unavailable."
    )
    st.subheader("Configuration and provenance")
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
        "Fixed method configurations, score definitions and provenance",
    )
    st.caption(
        "This view reads verified saved aggregates, exposes no customer identifiers, and never loads or trains a model."
    )
    st.stop()

if page == "Model Reliability":
    path = Path(
        st.text_input(
            "Saved reliability report",
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
            "No valid saved reliability evaluation is available. Run the optional calibration workflow first."
        )
        st.caption(str(exc))
        st.stop()
    st.warning(
        "Development-only calibration diagnostics. Independent final performance is not established. "
        "The previously inspected final holdout was excluded from this experiment."
    )
    if reliability["dataset"].get("source") == "synthetic":
        st.warning(
            "Synthetic demonstration; these diagnostics do not establish real-world reliability."
        )
    st.caption(
        "Calibrated outputs estimate the dataset's historical positive-label frequency under modeling "
        "assumptions. They are not confirmed theft probabilities. Dataset shift, label quality and "
        "population representativeness can invalidate calibration; independent human review is required."
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
        "The saved mapping was selected by Brier score on calibration_selection, before diagnostic evaluation. No fitting occurs in this view."
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
    table(summaries, "Development-only probability and discrimination comparison")
    chart(
        px.bar(summaries, x="method", y="brier_score"),
        "Development Brier score comparison (lower is better)",
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
            st.caption(f"{method}: {warning}")
    diagram.update_xaxes(title="Mean predicted positive-label probability", range=[0, 1])
    diagram.update_yaxes(title="Observed positive-label fraction", range=[0, 1])
    chart(diagram, "Development reliability diagram")
    table(pd.DataFrame(bins), "Fixed reliability bins, including counts and empty bins")
    st.caption(ECE_NOTICE)
    st.subheader("Development ranking comparison")
    table(pd.DataFrame(ranking), "Precision, recall and lift before and after calibration")
    st.caption(
        "AP and ROC-AUC appear above. Isotonic can introduce ties; sigmoid can also change rankings through reversal or numerical saturation. Actual probabilities and customer-ID tie breaks are used throughout."
    )
    st.subheader("Saved methodology and provenance")
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
        "Calibration methods, partition roles, seed and software metadata",
    )
    st.stop()

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
