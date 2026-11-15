import ast
import json
import re
from pathlib import Path
from string import Formatter

import pandas as pd
import plotly.express as px
import pytest

from gridguard.i18n import (
    CATALOG,
    LANGUAGES,
    NAVIGATION,
    display_label,
    format_count,
    format_number,
    group_for_workspace,
    localize_figure,
    localize_frame,
    translate,
)
from tests.dashboard_support import app_file, open_workspace
from tests.test_context_dashboard import context_files as context_files
from tests.test_dashboard import dashboard_artifacts as dashboard_artifacts

APP = Path(__file__).parents[1] / "app/streamlit_app.py"


def test_catalog_is_complete_and_preserves_format_parameters():
    for message, translations in CATALOG.items():
        assert set(translations) == set(LANGUAGES), message
        assert all(isinstance(value, str) and value.strip() for value in translations.values())
        fields = [
            {name for _, name, _, _ in Formatter().parse(value) if name is not None}
            for value in translations.values()
        ]
        assert fields[0] == fields[1], message
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    literal_keys = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tr"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert not literal_keys - CATALOG.keys()
    with pytest.raises(KeyError):
        translate("an undeclared UI message")


def test_five_groups_cover_every_legacy_view_once():
    assert len(NAVIGATION) == 5
    views = [view for items in NAVIGATION.values() for view in items]
    assert len(views) == len(set(views)) == 10
    assert group_for_workspace("Operational Context") == "operations"
    assert group_for_workspace("Model Reliability") == "model"
    assert group_for_workspace("unrecognized legacy environment value") == "overview"
    assert translate("navigation.overview") == "Genel Bakış"
    assert translate("navigation.operations", "en") == "Operational Review"


@pytest.mark.parametrize("language", LANGUAGES)
def test_display_copies_preserve_ids_scores_codes_and_plot_data(language):
    frame = pd.DataFrame(
        {
            "customer_id": pd.Series(["00001", "NA"], dtype="string"),
            "original_risk_score": [95.123, 18.0],
            "decision_status": ["needs_context_verification", "eligible_for_review"],
            "reason_codes": [["synthetic_context_only"], ["active_subscription_context"]],
            "matched_operational_events": [["unplanned_outage"], []],
        }
    )
    before = frame.copy(deep=True)
    displayed = localize_frame(frame, language)
    assert displayed.customer_id.tolist() == ["00001", "NA"]
    pd.testing.assert_series_equal(displayed.original_risk_score, frame.original_risk_score)
    assert displayed.reason_codes.dtype == displayed.matched_operational_events.dtype == object
    pd.testing.assert_frame_equal(frame, before)
    figure = px.bar(
        pd.DataFrame({"risk_band": ["High", "Low"], "count": [2, 4]}),
        x="risk_band",
        y="count",
        color="risk_band",
    )
    original = figure.to_json()
    localized = localize_figure(figure, language)
    assert [list(trace.y) for trace in localized.data] == [list(trace.y) for trace in figure.data]
    assert figure.to_json() == original
    if language == "tr":
        assert localized.layout.xaxis.title.text == "Risk düzeyi"
        assert localized.data[0].name == "Yüksek"
        assert "Risk düzeyi=%{x}" in localized.data[0].hovertemplate
        assert list(localized.data[0].x) == ["Yüksek"]
    assert display_label("test_predictions.csv", language) == "test_predictions.csv"
    assert display_label("CONS_NO", language) == "CONS_NO"
    assert format_number(0.5, language, percent=True) == ("%50,0" if language == "tr" else "50.0%")
    assert format_count(2000, language) == ("2.000" if language == "tr" else "2,000")


def test_new_session_defaults_to_turkish_and_language_survives_navigation(
    dashboard_artifacts, monkeypatch
):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GRIDGUARD_ARTIFACT_DIR", str(dashboard_artifacts))
    monkeypatch.delenv("GRIDGUARD_WORKSPACE", raising=False)
    app = AppTest.from_file(str(APP), default_timeout=25).run()
    assert not app.exception
    assert app.segmented_control(key="language").value == "tr"
    assert app.title[0].value == "Genel Bakış"
    assert app.sidebar.radio(key="section").options == [
        "Genel Bakış",
        "Risk Analizi",
        "Operasyonel İnceleme",
        "Model Performansı",
        "Veri Kalitesi",
    ]
    import shutil

    alternate = dashboard_artifacts.parent / "alternate-run"
    shutil.copytree(dashboard_artifacts, alternate)
    app.sidebar.text_input(key="ui_artifact_directory").set_value(str(alternate)).run()
    for view in (
        "Inspection queue",
        "Customer analytics",
        "Model performance",
        "Model Reliability",
        "Anomaly Detection",
        "Model Explainability",
        "Data quality",
        "Inspection Capacity Planning",
    ):
        open_workspace(app, view)
        assert not app.exception, view
        assert app.segmented_control(key="language").value == "tr"
    app.segmented_control(key="language").set_value("en").run()
    assert not app.exception
    assert app.title[0].value == "Operational Review"
    assert app.sidebar.text_input(key="ui_artifact_directory").value == str(alternate)
    for view in (
        "Inspection queue",
        "Customer analytics",
        "Model performance",
        "Model Reliability",
        "Anomaly Detection",
        "Model Explainability",
        "Data quality",
        "Inspection Capacity Planning",
    ):
        open_workspace(app, view)
        assert not app.exception, view
        assert app.segmented_control(key="language").value == "en"
    app.segmented_control(key="language").set_value("tr").run()
    assert not app.exception
    assert app.title[0].value == "Operasyonel İnceleme"
    app.slider(key="planning_capacity").set_value(0).run()
    assert app.segmented_control(key="language").value == "tr"


@pytest.mark.parametrize("missing", [None, "outages.csv", "synthetic_context.json"])
def test_turkish_context_and_language_switch_are_read_only(context_files, monkeypatch, missing):
    import joblib
    from sklearn.pipeline import Pipeline

    from gridguard.provenance import file_fingerprint

    if missing:
        (context_files["context_dir"] / missing).unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("The localized dashboard must not load, fit or rescore a model")

    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(Pipeline, "predict_proba", forbidden)
    for name, key in (
        ("GRIDGUARD_CONTEXT_DIR", "context_dir"),
        ("GRIDGUARD_CONTEXT_DECISIONS", "decision_dir"),
        ("GRIDGUARD_CONTEXT_PREDICTIONS", "predictions_path"),
        ("GRIDGUARD_CONTEXT_ROOT", "root"),
    ):
        monkeypatch.setenv(name, str(context_files[key]))
    monkeypatch.setenv("GRIDGUARD_ARTIFACT_DIR", "absent-legacy-experiment")
    monkeypatch.setenv("GRIDGUARD_WORKSPACE", "Operational Context")
    before = {
        path: file_fingerprint(path) for path in context_files["root"].rglob("*") if path.is_file()
    }
    app = app_file(str(APP), language="tr", default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Operasyonel İnceleme"
    assert any("SENTETİK / SYNTHETIC" in value.value for value in app.warning)
    assert any(
        metric.label == "Değerlendirilen müşteri" and metric.value == "6" for metric in app.metric
    )
    assert not any(
        customer in app.dataframe[0].value.to_json() for customer in context_files["ids"]
    )
    if missing:
        assert any("SENTETİK / SYNTHETIC" in value.value for value in app.warning)
    app.segmented_control(key="language").set_value("en").run()
    assert not app.exception
    assert app.title[0].value == "Operational Review"
    assert all(file_fingerprint(path) == checksum for path, checksum in before.items())
    open_workspace(app, "Overview")
    assert not app.exception


def test_readme_metrics_links_and_documented_cli_commands_are_valid():
    root = APP.parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    recorded = json.loads((root / "docs/experiments/pipeline-final.json").read_text())["test"]
    for name in ("average_precision", "roc_auc"):
        assert str(recorded[name]) in readme
    for target in re.findall(r"\]\(([^)]+)\)", readme):
        if "://" not in target:
            assert (root / target.split("#", 1)[0]).is_file(), target
    from gridguard.cli import main

    for command in set(re.findall(r"^gridguard ([a-z-]+)", readme, re.MULTILINE)):
        with pytest.raises(SystemExit) as result:
            main([command, "--help"])
        assert result.value.code == 0
