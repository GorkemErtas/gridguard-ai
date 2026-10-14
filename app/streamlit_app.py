"""Read-only Streamlit report over held-out predictions.

Launch after 'gridguard demo' or 'gridguard train' from the repository root.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="GridGuard AI", page_icon="⚡", layout="wide")


@st.cache_data(show_spinner=False)
def read_artifacts(folder: str) -> tuple[dict, pd.DataFrame]:
    path = Path(folder)
    report = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(path / "test_predictions.csv", dtype={"customer_id": "string"})
    return report, predictions


@st.cache_data(show_spinner=False)
def customer_history(csv_path: str, customer_id: str) -> pd.DataFrame:
    """Scan local CSV in bounded chunks, avoiding a full-file dashboard load."""
    for chunk in pd.read_csv(
        csv_path,
        chunksize=250,
        dtype={"CONS_NO": "string", "customer_id": "string"},
    ):
        id_column = "CONS_NO" if "CONS_NO" in chunk else "customer_id"
        selected = chunk.loc[chunk[id_column].astype(str) == customer_id]
        if selected.empty:
            continue
        record = selected.iloc[0]
        dates = []
        for column in chunk.columns:
            date = pd.to_datetime(column, format="%m/%d/%Y", errors="coerce")
            if pd.isna(date):
                date = pd.to_datetime(column, format="%Y-%m-%d", errors="coerce")
            if pd.notna(date):
                dates.append((date, column))
        dates.sort()
        values = pd.to_numeric(
            pd.Series([record[col] for _, col in dates]), errors="coerce"
        )
        return pd.DataFrame(
            {"date": [date for date, _ in dates], "kwh": values.to_numpy()}
        )
    return pd.DataFrame(columns=["date", "kwh"])


st.title("⚡ GridGuard AI")
st.caption("Electricity consumption intelligence · Customer-level risk ranking · Portfolio demo")
st.sidebar.header("Report")
artifact_dir = st.sidebar.text_input("Artifact directory", value="artifacts")
try:
    report, predictions = read_artifacts(artifact_dir)
except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
    st.info("No trained report found. Run the commands below from the repository root.")
    st.code("gridguard demo\nstreamlit run app/streamlit_app.py", language="bash")
    st.caption(str(exc))
    st.stop()

dataset = report["dataset"]
evaluation = report["test"]
source = dataset["source"]
if source == "synthetic":
    st.warning(
        "SYNTHETIC DEMO: consumption behavior and labels are simulated. "
        "The metrics below are not evidence of real-world electricity theft detection."
    )
else:
    st.info(
        "Risk scores are uncalibrated ranking signals, not proof of theft. "
        "A field investigation requires independent verification."
    )

st.caption(
    f"Source: {source} · Model: {report['selected_model']} · "
    f"Evaluation: untouched customer-level test split · "
    f"{report['split_sizes']['test']:,} test customers"
)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Test average precision", f"{evaluation['average_precision']:.3f}")
col2.metric("Test recall @ 0.5", f"{evaluation['recall_0_5']:.1%}")
col3.metric("Precision @ top 50", f"{evaluation['precision_at_50']:.1%}")
col4.metric("Test class prevalence", f"{evaluation['positive_prevalence']:.1%}")

tab_queue, tab_models, tab_history, tab_about = st.tabs(
    ["Inspection queue", "Model evaluation", "Reading history", "Methodology"]
)

with tab_queue:
    st.subheader("Investigation prioritization")
    st.caption(
        "These are held-out test customers. Bands are illustrative thresholds "
        "(High ≥80, Medium ≥50); they are not operational policy."
    )
    left, right = st.columns(2)
    with left:
        capacity = st.slider(
            "Available inspection slots (top K)", 1, min(200, len(predictions)),
            min(50, len(predictions))
        )
    with right:
        minimum_score = st.slider("Minimum risk score", 0, 100, 0)
    filtered = predictions.loc[predictions["risk_score"] >= minimum_score].head(capacity)
    if filtered.empty:
        st.info("No customer meets the selected score threshold.")
    else:
        retrospective_precision = filtered["true_label"].mean()
        st.metric("Retrospective precision of displayed queue", f"{retrospective_precision:.1%}")
        st.dataframe(
            filtered[["customer_id", "risk_score", "risk_band", "review_signal"]],
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Download inspection candidates (CSV)",
            data=filtered.drop(columns=["true_label"]).to_csv(index=False).encode("utf-8"),
            file_name="gridguard_inspection_candidates.csv",
            mime="text/csv",
        )
    band_counts = (
        predictions["risk_band"]
        .value_counts()
        .reindex(["High", "Medium", "Low"], fill_value=0)
        .rename_axis("Band")
        .reset_index(name="Customers")
    )
    st.plotly_chart(
        px.bar(band_counts, x="Band", y="Customers", title="Test-set risk band distribution"),
        use_container_width=True,
    )

with tab_models:
    st.subheader("Validation model comparison")
    st.caption(
        "Models are selected by validation average precision. "
        "The test partition is used only once, after model selection."
    )
    table = pd.DataFrame.from_dict(report["validation"], orient="index").reset_index(
        names="model"
    )
    st.dataframe(
        table[["model", "average_precision", "roc_auc", "precision_at_50",
               "recall_0_5", "f1_0_5"]],
        hide_index=True,
        use_container_width=True,
    )
    st.subheader("Final test evaluation")
    st.json(evaluation)
    st.caption(
        "Confusion matrix rows: actual [negative, positive]; "
        "columns: predicted [negative, positive] using an illustrative 0.5 cutoff."
    )

with tab_history:
    st.subheader("Customer reading history")
    selected_id = st.selectbox(
        "Held-out customer", predictions["customer_id"].astype(str).tolist()
    )
    chosen = predictions.loc[predictions["customer_id"] == selected_id].iloc[0]
    st.write(
        f"**Risk score:** {chosen['risk_score']}/100 · "
        f"**Band:** {chosen['risk_band']} · "
        f"**Reading indicators:** {chosen['review_signal']}"
    )
    st.caption(
        "Reading indicators are simple heuristics and do not explain "
        "the model's feature contributions."
    )
    raw_path = st.text_input("Original CSV on this machine", dataset["input_path"])
    if Path(raw_path).is_file():
        try:
            history = customer_history(raw_path, selected_id)
            if history.empty:
                st.info("The selected customer was not found in the specified CSV.")
            else:
                st.line_chart(history.set_index("date")["kwh"], y_label="Daily consumption (kWh)")
                st.caption("Missing days remain missing; they are not interpreted as zero usage.")
        except (OSError, ValueError) as exc:
            st.error(f"Could not read consumption history: {exc}")
    else:
        st.info("The original CSV is unavailable. Enter its local file path to view readings.")

with tab_about:
    st.markdown(
        """
        **Data:** one customer per row, at least 60 daily readings, optional binary \`FLAG\`.
        Negative readings are treated as missing. The target is never a feature.

        **Features:** statistical consumption distribution, recent-vs-previous 30-day
        change, weekly use patterns, zero readings, and missingness.

        **Training:** stratified customer-level 60/20/20 split. The logistic regression
        and random forest models are selected based only on validation average precision.
        A constant-prior dummy baseline is included for reference.

        **Interpretation:** \`risk_score\` = rounded model output ×100, not a calibrated
        probability. Risk bands are arbitrary dashboard labels, not safety-critical rules.
        Source labels are customer-level, not timestamps of theft events.
        """
    )
    st.caption(
        "Not for customer accusations, service interruption, or automated enforcement. "
        "See docs/model-card.md for limitations."
    )
