"""TR/EN presentation only: stored schemas, IDs and analytical inputs stay intact."""

import json
import re
from pathlib import Path

import pandas as pd

CATALOG = json.loads(Path(__file__).with_name("translations.json").read_text(encoding="utf-8"))
LANGUAGES = ("tr", "en")
NAVIGATION = {
    "overview": ("Overview",),
    "risk": ("Inspection queue", "Customer analytics"),
    "operations": ("Operational Context", "Inspection Capacity Planning"),
    "model": (
        "Model performance",
        "Model Reliability",
        "Anomaly Detection",
        "Model Explainability",
    ),
    "quality": ("Data quality",),
}
QUESTIONS = dict(
    zip(
        NAVIGATION,
        (
            "What is the overall situation?",
            "Which customers should be examined first?",
            "Which operational circumstances need verification?",
            "How reliable are the saved model results?",
            "How complete and trustworthy are the readings?",
        ),
    )
)


def translate(message, language="tr", **values):
    """Strict UI keys prevent silently incomplete translations."""
    if language not in LANGUAGES:
        raise ValueError("Unsupported display language.")
    result = CATALOG[message][language]
    return result.format(**values) if values else result


def display_label(value, language="tr"):
    """Known display vocabulary only; unknown technical identifiers stay verbatim."""
    if not isinstance(value, str):
        return value
    if value in CATALOG:
        return translate(value, language)
    schema_label = value.replace(" ", "_")
    if schema_label in CATALOG:
        return translate(schema_label, language)
    if "; " in value:
        return "; ".join(display_label(part, language) for part in value.split("; "))
    if " · " in value:
        return " · ".join(display_label(part, language) for part in value.split(" · "))
    return value


def group_for_workspace(workspace):
    return next((group for group, views in NAVIGATION.items() if workspace in views), "overview")


def format_count(value, language="tr"):
    result = f"{value:,}"
    return result.replace(",", ".") if language == "tr" else result


def format_number(value, language="tr", *, percent=False):
    if value is None or pd.isna(value):
        return translate("Unavailable", language)
    result = f"{value:.1%}" if percent else f"{value:,.3f}"
    if language == "tr":
        result = result.translate(str.maketrans(",.", ".,"))
        if percent:
            result = "%" + result.removesuffix("%")
    return result


def localize_frame(frame, language="tr", *, anonymous=False):
    """Copy browser-facing values; machine column names remain for column_config."""
    result = frame.copy(deep=True)
    categories = {
        "risk_band",
        "decision_status",
        "evidence_status",
        "event_type",
        "verification_status",
        "provenance",
        "rate_status",
        "status",
        "outcome",
        "review_signal",
        "Property",
    }
    for column in result:
        if column in ("reason_codes", "matched_operational_events"):
            result[column] = pd.Series(
                [[display_label(item, language) for item in values] for values in result[column]],
                index=result.index,
                dtype=object,
            )
        elif column in categories:
            result[column] = [display_label(value, language) for value in result[column]]
    if anonymous and "customer_id" in result:
        result["customer_id"] = [
            translate("Customer {number}", language, number=index + 1) for index in result.index
        ]
    return result


def _hover_text(value, language):
    if not isinstance(value, str):
        return value
    parts = re.split(r"(%\{[^}]+\}|<[^>]*>)", value)
    for i in range(0, len(parts), 2):
        text = parts[i]
        if "=" in text:
            name, value = text.split("=", 1)
            parts[i] = (
                str(display_label(name, language)) + "=" + str(display_label(value, language))
            )
            continue
        match = re.fullmatch(r"(.*?)([=: ]+)", text)
        if match:
            parts[i] = str(display_label(match[1], language)) + match[2]
        else:
            match = re.fullmatch(r"([^:]+):\s*(\d+)", text)
            parts[i] = (
                (str(display_label(match[1], language)) + ": " + match[2])
                if match
                else display_label(text, language)
            )
    return "".join(parts)


def localize_figure(figure, language="tr"):
    """Detach the figure before changing labels; numerical arrays stay unchanged."""
    import plotly.graph_objects as go

    result = go.Figure(figure)
    for axis in (*result.select_xaxes(), *result.select_yaxes()):
        if axis.title.text:
            axis.title.text = display_label(axis.title.text, language)
    if result.layout.legend.title.text:
        result.layout.legend.title.text = display_label(result.layout.legend.title.text, language)
    for trace in result.data:
        if trace.name:
            trace.name = display_label(trace.name, language)
        for attribute in ("x", "y", "labels", "text"):
            values = getattr(trace, attribute, None)
            if values is not None and not isinstance(values, str):
                if any(isinstance(value, str) for value in values):
                    setattr(trace, attribute, [_hover_text(value, language) for value in values])
        if getattr(trace, "hovertemplate", None):
            trace.hovertemplate = _hover_text(trace.hovertemplate, language)
    result.update_layout(separators=",." if language == "tr" else ".,")
    return result


def diagnostic_text(message, language="tr"):
    if message in CATALOG:
        return translate(message, language)
    if message.startswith("Missing SYNTHETIC context files: "):
        files = message.split(": ", 1)[1].split(". Saved decisions", 1)[0]
        if language == "en":
            return message
        return translate(
            "Partial SYNTHETIC context: {files}. Saved decisions remain available; missing evidence is unknown.",
            language,
            files=files,
        )
    return (
        message if language == "en" else translate("Saved diagnostic requires attention.", language)
    )


def metric_help(message, language="tr"):
    if "AP" in message or "precision" in message.lower() and "average" in message.lower():
        key = "AP summarizes ranking quality for the historical positive labels; it is not trapezoidal PR-AUC."
    elif "ROC-AUC" in message:
        key = "ROC-AUC measures separation across thresholds; assess AP as well when positives are rare."
    elif "precision" in message.lower():
        key = "Precision is the observed positive-label fraction in the selected group."
    elif "recall" in message.lower():
        key = "Recall is the fraction of all cohort positives found in the selected group."
    elif "lift" in message.lower():
        key = "Lift compares precision with the full cohort's positive-label rate."
    elif "Brier" in message:
        key = "Brier score measures squared probability error; lower is better."
    elif "ECE" in message:
        key = "ECE depends on binning and sample size and does not prove calibration."
    else:
        return None
    return translate(key, language)
