"""No real customer identifiers or historical artifacts are used in these tests."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gridguard.cli import main
from gridguard.operational import (
    PROVENANCE,
    ContextBundle,
    GenerationConfig,
    compatibility_report,
    event_bounds,
    generate_context,
    load_context,
    record_issues,
    records,
    save_context,
    source_population,
    time_value,
)
from gridguard.provenance import file_fingerprint


@pytest.fixture
def source(tmp_path):
    dates = pd.date_range("2014-01-01", periods=120)
    frame = pd.DataFrame(
        np.full((60, len(dates)), 10.0),
        columns=[f"{day.year}/{day.month}/{day.day}" for day in dates],
    )
    frame.insert(0, "CONS_NO", [f"{i:032d}" for i in range(60)])
    frame["FLAG"] = [0, 1] * 30
    path = tmp_path / "sgcc.csv"
    frame.to_csv(path, index=False)
    return path


def test_generation_is_reproducible_label_and_reading_independent(source, tmp_path):
    before = source.read_bytes()
    first = generate_context(source)
    second = generate_context(source)
    for name in first.tables:
        pd.testing.assert_frame_equal(first.tables[name], second.tables[name])
    assert first.metadata == second.metadata
    changed = pd.read_csv(source, dtype={"CONS_NO": "string"})
    changed["FLAG"] = "labels-must-not-be-read"
    changed.iloc[:, 1:-1] = 0.0
    changed = changed.iloc[::-1]
    other_path = tmp_path / "changed.csv"
    changed.to_csv(other_path, index=False)
    independent = generate_context(other_path)
    for name in first.tables:
        pd.testing.assert_frame_equal(first.tables[name], independent.tables[name])
    assert source.read_bytes() == before
    third = generate_context(source, config=GenerationConfig(seed=19))
    assert not third.tables["customer_regions"].equals(first.tables["customer_regions"])


def test_exact_ids_dates_regions_and_subscription_consistency(source):
    ids, _, dates = source_population(source)
    bundle = generate_context(
        source,
        config=GenerationConfig(
            communication_failures_per_customer_year=0.5,
            meter_faults_per_customer_year=0.5,
            replacements_per_customer_year=0.5,
        ),
    )
    mapping = bundle.tables["customer_regions"]
    assert mapping.customer_id.dtype == ids.dtype
    assert set(mapping.customer_id) == set(ids)
    assert mapping.customer_id.is_unique
    assert mapping.customer_id.str.len().eq(32).all()
    assert mapping.region_id.value_counts().min() > 1
    assert set(bundle.tables["outages"].outage_type) == {"planned", "unplanned"}
    assert set(bundle.tables["meter_events"].event_type) == {
        "communication_failure",
        "meter_fault",
        "meter_replacement",
    }
    assert bundle.metadata["date_start"] == dates[0][0].date().isoformat()
    for table, frame in bundle.tables.items():
        for row in records(frame):
            assert not record_issues(table, row, bundle.metadata, set(ids), set(mapping.region_id))
    periods = bundle.tables["subscriptions"].sort_values(["customer_id", "start_date"])
    for _, group in periods.groupby("customer_id"):
        assert all(a < b for a, b in zip(group.end_date.iloc[:-1], group.start_date.iloc[1:]))
    events = bundle.tables["meter_events"]
    replacements = events.loc[events.event_type == "meter_replacement"]
    assert replacements.start_date.eq(replacements.end_date).all()
    assert bundle.tables["subscriptions"].customer_id.nunique() < len(ids)
    verifications = pd.concat(
        [
            bundle.tables[name].verification_status
            for name in ("outages", "meter_events", "subscriptions")
        ]
    )
    assert verifications.isna().any() and verifications.eq("unverified").any()


def test_na_like_and_leading_zero_identifiers_are_preserved(source):
    frame = pd.read_csv(source, dtype={"CONS_NO": "string"})
    frame.loc[0:2, "CONS_NO"] = ["NA", "000012", "null"]
    frame.to_csv(source, index=False)
    ids, _, _ = source_population(source)
    assert set(ids.iloc[:3]) == {"NA", "000012", "null"}
    assert set(generate_context(source).tables["customer_regions"].customer_id) == set(ids)


def test_reference_fingerprint_and_safe_outputs(source, tmp_path):
    reference = tmp_path / "metrics.json"
    reference.write_text(json.dumps({"dataset": {"fingerprint_sha256": file_fingerprint(source)}}))
    output = tmp_path / "context"
    report = save_context(source, output_dir=output, reference_metrics=reference)
    loaded = load_context(output)
    assert loaded.metadata["provenance"] == PROVENANCE
    assert report["customer_id_match_rate"] == 1
    assert report["events_within_observation_period_pct"] == 100
    assert report["orphan_customer_references"] == report["invalid_region_references"] == 0
    assert report["events_overlapping_zero_or_missing"] == 0
    assert report["events_without_observed_anomaly"] == report["generated_events"]
    with pytest.raises(ValueError, match="empty"):
        save_context(source, output_dir=output)
    with pytest.raises(ValueError, match="separate"):
        save_context(source, output_dir=source.parent)
    reference.write_text('{"dataset": {"fingerprint_sha256": "wrong"}}')
    with pytest.raises(ValueError, match="fingerprint"):
        generate_context(source, reference_metrics=reference)
    assert len(loaded.tables["customer_regions"]) == 60


def test_post_generation_overlap_is_observed_honestly(source):
    frame = pd.read_csv(source, dtype={"CONS_NO": "string"})
    frame.iloc[:, 1:-1] = 0.0
    frame.iloc[:, 2] = np.nan
    # A missing calendar header is also a missing day, independent of row values.
    frame = frame.drop(columns=[frame.columns[3]])
    frame.to_csv(source, index=False)
    bundle = generate_context(source)
    report = compatibility_report(source, bundle, chunk_size=7)
    assert report["consumption_date_coverage"]["missing_calendar_days"] == 1
    assert report["events_overlapping_zero_or_missing"] == report["generated_events"]
    assert report["events_without_observed_anomaly"] == 0
    assert report["events_not_assessable"] == 0


def test_invalid_foreign_keys_and_temporal_records_are_reported(source):
    bundle = generate_context(source)
    tables = {name: frame.copy() for name, frame in bundle.tables.items()}
    tables["meter_events"].loc[0, "customer_id"] = "ORPHAN-TEST-ID"
    tables["meter_events"].loc[1, "end_date"] = "2099-01-01"
    tables["outages"].loc[0, "region_id"] = "UNKNOWN-REGION"
    report = compatibility_report(source, ContextBundle(tables, bundle.metadata))
    assert report["orphan_customer_references"] == 1
    assert report["invalid_region_references"] == 1
    assert report["invalid_event_records"] == 3
    assert report["events_within_observation_period_pct"] < 100


def test_overlap_respects_exclusive_outage_end_and_inclusive_meter_date(source):
    bundle = generate_context(source)
    customer = bundle.tables["customer_regions"].customer_id.iloc[0]
    region = bundle.tables["customer_regions"].region_id.iloc[0]
    tables = {name: frame.iloc[:0].copy() for name, frame in bundle.tables.items()}
    tables["customer_regions"] = bundle.tables["customer_regions"]
    tables["outages"].loc[0] = [
        "SYN-TEST-OUT",
        region,
        "2014-01-01T23:00:00",
        "2014-01-02T00:00:00",
        "planned",
        "verified",
        "2014-01-03",
        PROVENANCE,
    ]
    tables["meter_events"].loc[0] = [
        "SYN-TEST-MTR",
        customer,
        "meter_replacement",
        "2014-01-02",
        "2014-01-02",
        "verified",
        "2014-01-03",
        PROVENANCE,
    ]
    frame = pd.read_csv(source, dtype={"CONS_NO": "string"})
    frame.iloc[:, 2] = 0.0  # Jan 2 only.
    frame.to_csv(source, index=False)
    metadata = {**bundle.metadata, "source_sha256": file_fingerprint(source)}
    report = compatibility_report(source, ContextBundle(tables, metadata))
    assert report["overlap_by_table"]["outages"]["overlapping_zero"] == 0
    assert report["overlap_by_table"]["meter_events"]["overlapping_zero"] == 1
    begin, end = event_bounds("meter_events", {"start_date": None, "end_date": None})
    assert pd.isna(begin) and pd.isna(end)
    assert pd.isna(time_value("invalid-date"))


@pytest.mark.parametrize(
    "changes",
    [
        {"regions": 0},
        {"seed": -1},
        {"unverified_rate": 0.9, "missing_verification_rate": 0.2},
        {"terminated_rate": float("nan")},
        {"meter_faults_per_customer_year": -1},
    ],
)
def test_invalid_generation_settings(changes):
    with pytest.raises(ValueError):
        replace(GenerationConfig(), **changes)


def test_context_cli(source, tmp_path, capsys):
    output = tmp_path / "cli-context"
    main(
        [
            "generate-context",
            "--input",
            str(source),
            "--output-dir",
            str(output),
            "--seed",
            "17",
            "--regions",
            "4",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["sgcc_customers"] == 60
    loaded = load_context(output)
    assert loaded.metadata["config"]["seed"] == 17
    assert loaded.tables["customer_regions"].region_id.nunique() == 4
