import json
from pathlib import Path

import pandas as pd
import pytest

from gridguard.context_dashboard import (
    context_signature,
    customer_events,
    filter_context,
    inspection_frame,
    load_context_view,
    local_path,
    verified_context_history,
)
from gridguard.decision_gate import save_decisions
from gridguard.operational import PROVENANCE, GenerationConfig, generate_context
from gridguard.provenance import file_fingerprint, write_json
from tests.dashboard_support import open_workspace


@pytest.fixture
def context_files(tmp_path):
    ids = ["TEST-0006", "TEST-0001", "TEST-0005", "TEST-0002", "TEST-0004", "TEST-0003"]
    source = tmp_path / "source.csv"
    readings = pd.DataFrame(
        10.0, index=range(6), columns=pd.date_range("2014-01-01", periods=120).strftime("%Y-%m-%d")
    )
    readings.iloc[0, 1] = float("nan")
    readings.iloc[0, 2] = 0
    readings.insert(0, "CONS_NO", ids)
    readings["FLAG"] = [1, 0, 1, 0, 0, 1]
    readings.to_csv(source, index=False)
    context = generate_context(
        source,
        config=GenerationConfig(
            regions=2,
            missing_subscription_rate=0,
            terminated_rate=0,
            unverified_rate=0,
            missing_verification_rate=0,
        ),
    )
    mapping = context.tables["customer_regions"]
    mapping["region_id"] = [
        "SYN-REG-001" if customer in ids[:3] else "SYN-REG-002" for customer in mapping.customer_id
    ]
    for name in ("outages", "meter_events"):
        context.tables[name].drop(context.tables[name].index, inplace=True)
    context.tables["outages"].loc[0] = [
        "SYN-OUT-1",
        "SYN-REG-001",
        "2014-02-03T02:00:00",
        "2014-02-03T08:00:00",
        "unplanned",
        "verified",
        "2014-02-04",
        PROVENANCE,
    ]
    for index, (customer, kind, verification, available) in enumerate(
        [
            (ids[1], "communication_failure", "unverified", "2014-02-11"),
            (ids[4], "meter_replacement", "verified", "2014-02-11"),
            (ids[4], "meter_fault", "unverified", "2014-02-11"),
            (ids[3], "communication_failure", "verified", "2014-02-16"),
        ]
    ):
        context.tables["meter_events"].loc[index] = [
            f"SYN-MTR-{index}",
            customer,
            kind,
            "2014-02-10",
            "2014-02-10",
            verification,
            available,
            PROVENANCE,
        ]
    context.tables["subscriptions"].drop(
        context.tables["subscriptions"]
        .loc[context.tables["subscriptions"].customer_id.eq(ids[-1])]
        .index,
        inplace=True,
    )
    subscription = context.tables["subscriptions"]
    active_index = subscription.index[subscription.customer_id.eq(ids[2])][0]
    subscription.loc[active_index, "end_date"] = "2014-02-04"
    subscription.loc[subscription.index.max() + 1] = [
        "SYN-SUB-TERMINATED",
        ids[2],
        "2014-02-05",
        "2014-04-30",
        "terminated",
        "verified",
        "2014-02-05",
        PROVENANCE,
    ]
    folder = tmp_path / "context"
    folder.mkdir()
    for name, frame in context.tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
    write_json(folder / "synthetic_context.json", context.metadata)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "customer_id": ids,
            "risk_score": [92, 88, 65, 42, 85, 32],
            "true_label": [1, 0, 1, 0, 0, 1],
            "private_note": ["DO NOT DISPLAY"] * 6,
        }
    ).to_csv(predictions, index=False)
    output = tmp_path / "decisions"
    save_decisions(
        predictions,
        context_dir=folder,
        output_dir=output,
        evaluation_date="2014-02-15",
        window_start="2014-02-01",
        window_end="2014-02-10",
    )
    return {
        "root": tmp_path,
        "context_dir": folder,
        "decision_dir": output,
        "predictions_path": predictions,
        "source": source,
        "ids": ids,
    }


def view(files):
    return load_context_view(
        **{key: files[key] for key in ("root", "context_dir", "decision_dir", "predictions_path")}
    )


def test_parse_replay_and_preserve_exact_source_files(context_files):
    files = context_files
    before = {path: file_fingerprint(path) for path in files["root"].rglob("*") if path.is_file()}
    parsed = view(files)
    assert not parsed.warnings
    assert parsed.decisions.customer_id.tolist() == files["ids"]
    assert parsed.decisions.original_risk_score.tolist() == [92, 88, 65, 42, 85, 32]
    for column in ("reason_codes", "matched_event_ids"):
        assert parsed.decisions[column].dtype == object
        assert all(isinstance(value, list) for value in parsed.decisions[column])
    assert parsed.report["customers_per_decision_status"] == {
        "eligible_for_review": 1,
        "needs_context_verification": 4,
        "insufficient_data": 1,
    }
    assert "SYN-MTR-3" not in parsed.events.event_id.tolist()  # Future evidence.
    assert {"active", "terminated"} <= set(customer_events(parsed, files["ids"][2]).event_type)
    assert not {"true_label", "private_note"} & set(parsed.decisions)
    assert all(file_fingerprint(path) == fingerprint for path, fingerprint in before.items())


def test_filters_preserve_order_and_require_same_event_match(context_files):
    parsed = view(context_files)
    assert filter_context(parsed, risk_range=(80, 100)).customer_id.tolist() == [
        context_files["ids"][index] for index in (0, 1, 4)
    ]
    assert filter_context(parsed, statuses=[]).empty
    assert filter_context(parsed, search="TEST-0002").customer_id.tolist() == ["TEST-0002"]
    assert filter_context(parsed, search=".*").empty  # Literal, not regular expression.
    assert filter_context(
        parsed, event_types=["meter_fault"], verifications=["unverified"]
    ).customer_id.tolist() == [context_files["ids"][4]]
    assert filter_context(
        parsed, event_types=["meter_replacement"], verifications=["unverified"]
    ).empty
    assert filter_context(
        parsed, event_types=["unplanned_outage"], verifications=["unverified"]
    ).empty
    assert filter_context(parsed, statuses=["insufficient_data"]).customer_id.tolist() == [
        context_files["ids"][-1]
    ]


def test_display_allowlist_masks_ids_and_does_not_mutate(context_files):
    parsed = view(context_files)
    before = parsed.decisions.copy(deep=True)
    masked = inspection_frame(parsed.decisions, parsed.events)
    assert masked.customer_id.tolist() == [f"Customer {i}" for i in range(1, 7)]
    assert not any(customer in masked.to_json() for customer in context_files["ids"])
    assert not {"matched_event_ids", "true_label", "private_note"} & set(masked)
    assert (
        inspection_frame(parsed.decisions, parsed.events, reveal_ids=True).customer_id.tolist()
        == context_files["ids"]
    )
    events = customer_events(parsed, context_files["ids"][1])
    assert "customer_id" not in events and "event_id" not in events
    assert "event_id" in customer_events(parsed, context_files["ids"][1], reveal_ids=True)
    pd.testing.assert_frame_equal(before, parsed.decisions)


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
@pytest.mark.parametrize(
    "missing",
    [None, "meter_events.csv", "outages.csv", "customer_regions.csv", "synthetic_context.json"],
)
def test_inspection_lists_are_objects_and_arrow_round_trip(context_files, storage, missing):
    """Run under both CI dependency jobs, independently of pandas' string default."""
    pa = pytest.importorskip("pyarrow")
    if missing:
        (context_files["context_dir"] / missing).unlink()
    parsed = view(context_files)
    parsed.decisions["customer_id"] = parsed.decisions.customer_id.astype(f"string[{storage}]")
    if storage == "pyarrow":
        # A caller may already carry Arrow list columns; the browser boundary
        # must still produce Python lists rather than preserve Arrow metadata.
        for column in ("reason_codes", "matched_event_ids"):
            parsed.decisions[column] = pd.Series(
                parsed.decisions[column].tolist(),
                index=parsed.decisions.index,
                dtype=pd.ArrowDtype(pa.list_(pa.string())),
            )
    decisions_before, events_before = (
        parsed.decisions.copy(deep=True),
        parsed.events.copy(deep=True),
    )
    # Nonconsecutive indices catch accidental positional realignment after filtering.
    selected = filter_context(parsed, risk_range=(80, 100))
    shown = inspection_frame(selected, parsed.events, reveal_ids=True)
    for column in ("reason_codes", "matched_operational_events"):
        assert shown[column].dtype == object
        assert all(isinstance(value, list) for value in shown[column])
    assert shown.index.tolist() == selected.index.tolist() == [0, 1, 4]
    pd.testing.assert_frame_equal(
        shown[["customer_id", "original_risk_score", "decision_status", "evidence_status"]],
        selected[["customer_id", "original_risk_score", "decision_status", "evidence_status"]],
    )
    assert shown.reason_codes.tolist() == [list(value) for value in selected.reason_codes]
    available = parsed.events.groupby("customer_id").size()
    for index, row in selected.iterrows():
        expected = sorted(
            set(parsed.events.loc[parsed.events.customer_id.eq(row.customer_id), "event_type"])
        )
        if available.get(row.customer_id, 0) < len(row.matched_event_ids):
            expected.append("unknown / unresolved evidence")
        assert shown.matched_operational_events.loc[index] == expected
    # Streamlit AppTest reconstructs browser frames from Arrow. Arrow extension
    # list dtype metadata used to break that reconstruction with PyArrow 26.
    restored = pa.Table.from_pandas(shown).to_pandas()
    for column in ("reason_codes", "matched_operational_events"):
        assert [list(value) for value in restored[column]] == shown[column].tolist()
    pd.testing.assert_frame_equal(decisions_before, parsed.decisions)
    pd.testing.assert_frame_equal(events_before, parsed.events)


@pytest.mark.parametrize("empty_filter", [False, True])
def test_empty_event_lists_keep_object_dtype_and_serialize(context_files, empty_filter):
    pa = pytest.importorskip("pyarrow")
    parsed = view(context_files)
    parsed.events = parsed.events.iloc[:0].copy()
    parsed.decisions["customer_id"] = parsed.decisions.customer_id.astype("string[pyarrow]")
    # No recorded matches is different from recorded matches whose evidence is missing.
    parsed.decisions["matched_event_ids"] = pd.Series(
        [[] for _ in parsed.decisions.index],
        index=parsed.decisions.index,
        dtype=pd.ArrowDtype(pa.list_(pa.null())),
    )
    selected = filter_context(parsed, statuses=[] if empty_filter else ["eligible_for_review"])
    shown = inspection_frame(selected, parsed.events)
    assert shown.matched_operational_events.dtype == object
    assert shown.reason_codes.dtype == object
    assert shown.matched_operational_events.tolist() == [[] for _ in selected.index]
    assert len(pa.Table.from_pandas(shown).to_pandas()) == len(selected)
    if not empty_filter:
        shown.at[shown.index[0], "matched_operational_events"] = ["unknown / unresolved evidence"]
        assert shown.matched_operational_events.iloc[0] == ["unknown / unresolved evidence"]


@pytest.mark.parametrize(
    "filename",
    ["meter_events.csv", "outages.csv", "customer_regions.csv", "synthetic_context.json"],
)
def test_missing_context_retains_saved_decisions_with_explicit_unknowns(context_files, filename):
    (context_files["context_dir"] / filename).unlink()
    parsed = view(context_files)
    assert len(parsed.decisions) == 6
    assert parsed.warnings
    if filename == "synthetic_context.json":
        assert parsed.events.empty
    shown = inspection_frame(parsed.decisions, parsed.events)
    assert any(
        "unknown / unresolved evidence" in values for values in shown.matched_operational_events
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("original_risk_score", 97),
        ("decision_status", "eligible_for_review"),
        ("reason_codes", '["synthetic_context_only"]'),
        ("matched_event_ids", "[]"),
        ("evidence_status", "synthetic_only"),
        ("customer_id", "UNRELATED-ID"),
        ("provenance", "REAL"),
        ("reason_codes", "__import__('os').getcwd()"),
        ("matched_event_ids", '[{"id": "nested"}]'),
    ],
)
def test_tampered_decisions_are_rejected(context_files, column, value):
    path = context_files["decision_dir"] / "context_decisions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        view(context_files)


@pytest.mark.parametrize("kind", ["source", "aggregate", "event", "population", "prediction"])
def test_mixed_or_changed_artifacts_are_rejected(context_files, kind):
    files = context_files
    report_path = files["decision_dir"] / "context_evaluation.json"
    if kind in ("source", "aggregate"):
        report = json.loads(report_path.read_text())
        if kind == "source":
            report["source_sha256"] = "a" * 64
        else:
            report["context_coverage"]["subscription_window_coverage_rate"] = 0.1
        write_json(report_path, report)
    elif kind == "event":
        path = files["context_dir"] / "outages.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "start_time"] = "2014-03-01T02:00:00"
        frame.loc[0, "end_time"] = "2014-03-01T08:00:00"
        frame.to_csv(path, index=False)
    elif kind == "population":
        path = files["context_dir"] / "customer_regions.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "customer_id"] = "SUBSTITUTED-ID"
        frame.to_csv(path, index=False)
    else:
        with files["predictions_path"].open("a") as handle:
            handle.write("\n")
    with pytest.raises(ValueError):
        view(files)


def test_cache_signatures_include_all_inputs_and_deletions(context_files):
    kwargs = {
        key: context_files[key]
        for key in ("root", "decision_dir", "context_dir", "predictions_path")
    }
    before = context_signature(**kwargs)
    path = context_files["context_dir"] / "meter_events.csv"
    with path.open("a") as handle:
        handle.write("\n")
    assert context_signature(**kwargs) != before
    changed = context_signature(**kwargs)
    path.unlink()
    assert context_signature(**kwargs) != changed


@pytest.mark.parametrize(
    "unsafe",
    [
        "../outside.csv",
        "https://example.com/events.csv",
        "//server/share",
        "\\\\server\\share",
        ".git/config.csv",
        ".env.csv",
        "source.csv:secret",
        "\x00bad",
    ],
)
def test_unsafe_paths_are_rejected(context_files, unsafe):
    with pytest.raises((ValueError, OSError)):
        local_path(unsafe, context_files["root"], suffix=".csv")


def test_symlink_escape_and_absolute_outside_paths(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("private")
    with pytest.raises(ValueError, match="root"):
        local_path(outside, root, suffix=".csv")
    try:
        (root / "link.csv").symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this Windows account.")
    with pytest.raises(ValueError, match="root"):
        local_path(root / "link.csv", root, suffix=".csv")


def test_history_checks_fingerprint_and_keeps_gaps(context_files):
    files = context_files
    expected = file_fingerprint(files["source"])
    history = verified_context_history(files["source"], files["ids"][0], expected)
    assert len(history) == 120
    assert pd.isna(history.kwh.iloc[1]) and history.kwh.iloc[2] == 0
    with pytest.raises(ValueError, match="fingerprint"):
        verified_context_history(files["source"], files["ids"][0], "wrong")


def test_missing_prediction_reference_is_explicitly_unverified(context_files):
    parsed = load_context_view(
        context_files["decision_dir"], context_files["context_dir"], root=context_files["root"]
    )
    assert any("score provenance is unverified" in warning for warning in parsed.warnings)


def test_partial_bundle_rejects_private_text_in_reason_codes(context_files):
    (context_files["context_dir"] / "outages.csv").unlink()
    path = context_files["decision_dir"] / "context_decisions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "reason_codes"] = json.dumps(["synthetic_context_only", context_files["ids"][0]])
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="reason codes"):
        view(context_files)


def test_missing_risk_can_be_filtered_without_changing_saved_score(context_files):
    parsed = view(context_files)
    parsed.decisions.loc[5, "original_risk_score"] = float("nan")
    selected = filter_context(parsed, risk_range=(80, 100))
    assert context_files["ids"][5] in selected.customer_id.tolist()
    assert (
        context_files["ids"][5]
        not in filter_context(
            parsed, risk_range=(80, 100), include_missing_scores=False
        ).customer_id.tolist()
    )
    assert pd.isna(inspection_frame(selected, parsed.events).original_risk_score.loc[5])


def app_for(files, monkeypatch):
    pytest.importorskip("streamlit")
    from tests.dashboard_support import app_file

    monkeypatch.setenv("GRIDGUARD_ARTIFACT_DIR", str(files["root"] / "missing-legacy"))
    monkeypatch.setenv("GRIDGUARD_CONTEXT_ROOT", str(files["root"]))
    monkeypatch.setenv("GRIDGUARD_CONTEXT_DIR", str(files["context_dir"]))
    monkeypatch.setenv("GRIDGUARD_CONTEXT_DECISIONS", str(files["decision_dir"]))
    monkeypatch.setenv("GRIDGUARD_CONTEXT_PREDICTIONS", str(files["predictions_path"]))
    monkeypatch.setenv("GRIDGUARD_CONTEXT_SOURCE", str(files["source"]))
    monkeypatch.setenv("GRIDGUARD_WORKSPACE", "Operational Context")
    return app_file(
        str(Path(__file__).parents[1] / "app/streamlit_app.py"), default_timeout=30
    ).run()


def test_workspace_kpis_filters_drilldown_and_read_only(context_files, monkeypatch):
    import joblib
    from sklearn.pipeline import Pipeline

    import gridguard.context_dashboard as helpers

    def forbidden(*args, **kwargs):
        pytest.fail("Operational context must not load, fit or score models")

    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(Pipeline, "predict_proba", forbidden)
    history_loader = helpers.load_customer_history

    def premature_history(*args, **kwargs):
        pytest.fail("Source history must not be read before explicit form submission")

    monkeypatch.setattr(helpers, "load_customer_history", premature_history)
    before = {
        path: file_fingerprint(path) for path in context_files["root"].rglob("*") if path.is_file()
    }
    app = app_for(context_files, monkeypatch)
    assert not app.exception
    assert any("SYNTHETIC" in warning.value for warning in app.warning)
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Total evaluated customers"] == "6"
    assert metrics["Needs context verification"] == "4"
    assert metrics["Context coverage"] == "83.3%"
    assert len(app.get("plotly_chart")) == 1
    assert not app.get("download_button")
    browser = app.dataframe[0].value
    assert not any(customer in browser.to_json() for customer in context_files["ids"])
    app.multiselect(key="context_event_type").set_value(["meter_fault"]).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 1
    app.multiselect(key="context_verification").set_value(["verified"]).run()
    assert not app.exception
    assert any("No customers" in item.value for item in app.info)
    app.multiselect(key="context_verification").set_value([]).run()
    app.checkbox(key="context_reveal_ids").check().run()
    assert app.dataframe[0].value.customer_id.tolist() == [context_files["ids"][4]]
    monkeypatch.setattr(helpers, "load_customer_history", history_loader)
    app.button[0].click().run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 3
    assert any("SHA-256 verified" in caption.value for caption in app.caption)
    assert all(file_fingerprint(path) == fingerprint for path, fingerprint in before.items())


def test_missing_context_workspace_does_not_break_legacy(context_files, monkeypatch):
    app = app_for(context_files, monkeypatch)
    app.text_input(key="context_dataset_dir").set_value("missing-context").run()
    assert not app.exception
    assert any("No valid saved operational context" in item.value for item in app.info)
    open_workspace(app, "Overview")
    assert not app.exception
    assert any("Select a local experiment" in item.value for item in app.info)


def test_partial_workspace_renders_unknown_evidence(context_files, monkeypatch):
    (context_files["context_dir"] / "outages.csv").unlink()
    app = app_for(context_files, monkeypatch)
    assert not app.exception
    assert any("Missing SYNTHETIC context" in item.value for item in app.warning)
    assert (
        "unknown / unresolved evidence" in app.dataframe[0].value.matched_operational_events.iloc[0]
    )
