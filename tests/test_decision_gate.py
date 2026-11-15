import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gridguard.cli import main
from gridguard.decision_gate import GateConfig, decision_gate, save_decisions
from gridguard.operational import PROVENANCE, TABLE_COLUMNS, ContextBundle
from gridguard.provenance import partition_fingerprint

CUSTOMERS = ["TEST-0003", "TEST-0001", "TEST-0002"]
WINDOW = {"evaluation_date": "2014-02-15", "window_start": "2014-02-01", "window_end": "2014-02-10"}


@pytest.fixture
def context():
    tables = {name: pd.DataFrame(columns=columns) for name, columns in TABLE_COLUMNS.items()}
    for i, customer in enumerate(CUSTOMERS):
        tables["customer_regions"].loc[i] = [
            customer,
            "SYN-R1" if i == 0 else "SYN-R0",
            "2014-01-01",
            PROVENANCE,
        ]
        tables["subscriptions"].loc[i] = [
            f"SYN-SUB-{i}",
            customer,
            "2014-01-01",
            "2014-04-30",
            "active",
            "verified",
            "2014-01-01",
            PROVENANCE,
        ]
    return ContextBundle(
        tables,
        {
            "schema_version": 1,
            "provenance": PROVENANCE,
            "date_start": "2014-01-01",
            "date_end": "2014-04-30",
            "source_sha256": "test-source-not-real-data",
            "sgcc_customers": 3,
            "population_fingerprint": partition_fingerprint(sorted(CUSTOMERS)),
        },
    )


@pytest.fixture
def predictions():
    return pd.DataFrame(
        {
            "customer_id": pd.Series(CUSTOMERS, dtype="string"),
            "risk_score": [17, 95, 84],
            "model_score": [0.17, 0.95001, 0.84],
            "true_label": [1, 0, 1],  # Intentionally ignored.
        }
    )


def outage(context, **changes):
    row = dict(
        zip(
            TABLE_COLUMNS["outages"],
            [
                "SYN-OUT-TEST",
                "SYN-R0",
                "2014-02-03T02:00:00",
                "2014-02-03T08:00:00",
                "unplanned",
                "verified",
                "2014-02-04",
                PROVENANCE,
            ],
        )
    )
    context.tables["outages"].loc[len(context.tables["outages"])] = {**row, **changes}


def meter(context, **changes):
    row = dict(
        zip(
            TABLE_COLUMNS["meter_events"],
            [
                "SYN-MTR-TEST",
                "TEST-0001",
                "communication_failure",
                "2014-02-03",
                "2014-02-07",
                "verified",
                "2014-02-08",
                PROVENANCE,
            ],
        )
    )
    context.tables["meter_events"].loc[len(context.tables["meter_events"])] = {**row, **changes}


def test_preserves_scores_order_predictions_and_determinism(context, predictions):
    before = predictions.copy(deep=True)
    first, report = decision_gate(predictions, context, **WINDOW)
    second, other = decision_gate(predictions, context, **WINDOW)
    pd.testing.assert_frame_equal(predictions, before)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_series_equal(
        first.original_risk_score, predictions.risk_score, check_names=False
    )
    assert first.customer_id.tolist() == CUSTOMERS
    assert first.decision_status.eq("eligible_for_review").all()
    assert first.evidence_status.eq("synthetic_partial_unknown").all()
    assert "no_matching_outages_unknown" in first.reason_codes.iloc[0]
    assert report == other
    assert report["context_coverage"]["missing_outage_or_meter_match_rate"] == 1
    changed = predictions.copy()
    changed["true_label"] = 1 - changed.true_label
    changed["model_score"] = 0
    pd.testing.assert_frame_equal(first, decision_gate(changed, context, **WINDOW)[0])
    strict, _ = decision_gate(
        predictions, context, config=GateConfig(missing_event_policy="verify"), **WINDOW
    )
    assert strict.decision_status.eq("needs_context_verification").all()


def test_regional_outage_affects_multiple_mapped_customers(context, predictions):
    outage(context)
    decisions, report = decision_gate(predictions, context, **WINDOW)
    assert decisions.decision_status.tolist() == [
        "eligible_for_review",
        "needs_context_verification",
        "needs_context_verification",
    ]
    assert "SYN-OUT-TEST" not in decisions.matched_event_ids.iloc[0]
    assert all("SYN-OUT-TEST" in ids for ids in decisions.matched_event_ids.iloc[1:])
    assert report["customers_affected_by_event_type"]["unplanned_outage"] == 2
    assert report["matched_event_verification_counts"]["verified"] == 4  # Not per customer.
    assert report["high_risk_customers_requiring_verification"] == 2
    assert "synthetic_context_only" in decisions.reason_codes.iloc[1]


@pytest.mark.parametrize(
    "verification,reason",
    [
        ("unverified", "unverified_evidence"),
        (None, "missing_evidence"),
    ],
)
def test_unverified_and_missing_evidence(context, predictions, verification, reason):
    meter(context, verification_status=verification)
    decisions, report = decision_gate(predictions, context, **WINDOW)
    assert decisions.decision_status.iloc[1] == "needs_context_verification"
    assert reason in decisions.reason_codes.iloc[1]
    assert "overlapping_communication_failure" in decisions.reason_codes.iloc[1]
    key = "missing" if verification is None else verification
    assert report["matched_event_verification_counts"][key] == 1


def test_future_rows_and_future_verification_do_not_change_decisions(context, predictions):
    expected, _ = decision_gate(predictions, context, **WINDOW)
    meter(context, available_at="2014-02-16")  # Past event, not yet reported/verified.
    outage(context, available_at="2014-02-16")
    # Delayed contradictory subscription state and duplicate mapping must also be ignored.
    context.tables["subscriptions"].loc[3] = [
        "LATE-SUB",
        "TEST-0001",
        "2014-02-01",
        "2014-02-10",
        "terminated",
        "verified",
        "2014-02-16",
        PROVENANCE,
    ]
    context.tables["customer_regions"].loc[3] = ["TEST-0001", "SYN-R1", "2014-02-16", PROVENANCE]
    actual, report = decision_gate(predictions, context, **WINDOW)
    pd.testing.assert_frame_equal(expected, actual)
    assert report["ignored_future_records"] == {
        "customer_regions": 1,
        "outages": 1,
        "meter_events": 1,
        "subscriptions": 1,
    }
    after, _ = decision_gate(predictions, context, **{**WINDOW, "evaluation_date": "2014-02-16"})
    assert after.decision_status.iloc[1] == "insufficient_data"


@pytest.mark.parametrize(
    "changes",
    [
        {"start_date": "2014-03-01", "end_date": "2014-03-02", "available_at": "2014-01-01"},
        {"start_date": "2014-01-01", "end_date": "2014-01-31", "available_at": "2014-02-01"},
    ],
)
def test_nonoverlapping_events_are_irrelevant(context, predictions, changes):
    expected, _ = decision_gate(predictions, context, **WINDOW)
    meter(context, **changes)
    actual, _ = decision_gate(predictions, context, **WINDOW)
    pd.testing.assert_frame_equal(expected, actual)


def test_temporal_boundaries_and_cutoff(context, predictions):
    # An outage ending exactly at the window start is excluded.
    outage(context, start_time="2014-01-31T23:00:00", end_time="2014-02-01T00:00:00")
    meter(
        context, start_date="2014-02-10", end_date="2014-02-10", available_at="2014-02-15T23:59:59Z"
    )
    decisions, report = decision_gate(predictions, context, **WINDOW)
    assert report["customers_affected_by_event_type"]["unplanned_outage"] == 0
    assert decisions.decision_status.iloc[1] == "needs_context_verification"
    # An exact timestamp cutoff includes that instant, not the remainder of the day.
    exact, _ = decision_gate(
        predictions, context, **{**WINDOW, "evaluation_date": "2014-02-15T12:00:00Z"}
    )
    assert exact.decision_status.iloc[1] == "eligible_for_review"


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"end_date": None}, "invalid_event_interval"),
        ({"end_date": "2014-01-01"}, "invalid_event_interval"),
        ({"end_date": "2099-01-01"}, "outside_observation_period"),
        ({"available_at": None}, "missing_or_invalid_availability"),
        ({"verification_status": "maybe"}, "invalid_verification_status"),
        ({"provenance": "REAL"}, "invalid_provenance"),
        ({"event_type": "invented"}, "invalid_event_type"),
        ({"event_id": None}, "missing_event_id"),
    ],
)
def test_invalid_records_are_quarantined(context, predictions, changes, reason):
    meter(context, **changes)
    decisions, report = decision_gate(predictions, context, **WINDOW)
    assert decisions.decision_status.iloc[1] == "insufficient_data"
    assert reason in decisions.reason_codes.iloc[1]
    assert report["invalid_record_issue_counts"][reason] == 1
    assert "SYN-MTR-TEST" not in decisions.matched_event_ids.iloc[1]


def test_duplicate_events_and_unknown_foreign_keys(context, predictions):
    meter(context)
    meter(context)
    outage(context, region_id="ORPHAN-REGION")
    meter(context, customer_id="ORPHAN-CUSTOMER", event_id="ORPHAN-ID")
    decisions, report = decision_gate(predictions, context, **WINDOW)
    assert decisions.decision_status.iloc[1] == "insufficient_data"
    assert report["invalid_record_issue_counts"]["duplicate_event_id"] == 2
    assert report["invalid_record_issue_counts"]["invalid_region_reference"] == 1
    assert report["invalid_record_issue_counts"]["orphan_customer_reference"] == 1
    assert report["unlinked_invalid_records"] == 2


def test_missing_mapping_subscription_and_subscription_gap(context, predictions):
    context.tables["customer_regions"] = context.tables["customer_regions"].iloc[1:]
    context.tables["subscriptions"].loc[1, "end_date"] = "2014-02-04"
    context.tables["subscriptions"] = context.tables["subscriptions"].iloc[:2]
    decisions, report = decision_gate(predictions, context, **WINDOW)
    assert decisions.decision_status.eq("insufficient_data").all()
    assert "missing_or_invalid_region_mapping" in decisions.reason_codes.iloc[0]
    assert "subscription_coverage_unknown" in decisions.reason_codes.iloc[1]
    assert report["context_coverage"]["missing_subscription_record_rate"] == pytest.approx(2 / 3)


def test_conflicting_subscription_states_and_valid_transition(context, predictions):
    context.tables["subscriptions"].loc[3] = [
        "SYN-TERMINATED",
        "TEST-0001",
        "2014-02-05",
        "2014-04-30",
        "terminated",
        "verified",
        "2014-02-05",
        PROVENANCE,
    ]
    decisions, _ = decision_gate(predictions, context, **WINDOW)
    assert decisions.decision_status.iloc[1] == "insufficient_data"
    assert "conflicting_subscription_intervals" in decisions.reason_codes.iloc[1]
    context.tables["subscriptions"].loc[1, "end_date"] = "2014-02-04"
    decisions, _ = decision_gate(predictions, context, **WINDOW)
    assert decisions.decision_status.iloc[1] == "needs_context_verification"
    assert "overlapping_terminated" in decisions.reason_codes.iloc[1]
    assert "conflicting_subscription_intervals" not in decisions.reason_codes.iloc[1]


def test_configurable_relevant_meter_types(context, predictions):
    meter(context, event_type="meter_replacement", end_date="2014-02-03")
    decisions, _ = decision_gate(
        predictions, context, config=GateConfig(relevant_meter_types=()), **WINDOW
    )
    assert decisions.decision_status.iloc[1] == "eligible_for_review"
    assert "configured_irrelevant_meter_event" in decisions.reason_codes.iloc[1]
    assert "SYN-MTR-TEST" in decisions.matched_event_ids.iloc[1]


@pytest.mark.parametrize("score", [np.nan, np.inf, -1, 101, "invalid"])
def test_invalid_scores_cannot_become_eligible(context, predictions, score):
    predictions["risk_score"] = predictions.risk_score.astype(object)
    predictions.loc[1, "risk_score"] = score
    decisions, _ = decision_gate(predictions, context, **WINDOW)
    assert decisions.decision_status.iloc[1] == "insufficient_data"
    assert "invalid_or_missing_risk_score" in decisions.reason_codes.iloc[1]


@pytest.mark.parametrize(
    "window",
    [
        {"evaluation_date": "2014-02-09"},
        {"window_start": "2014-02-11"},
        {"window_start": "2013-01-01"},
        {"window_end": "2015-01-01", "evaluation_date": "2015-01-02"},
        {"evaluation_date": "not-a-date"},
        {"evaluation_date": "now"},
        {"evaluation_date": "today"},
    ],
)
def test_invalid_evaluation_windows(context, predictions, window):
    with pytest.raises(ValueError):
        decision_gate(predictions, context, **{**WINDOW, **window})


def test_invalid_ids_empty_cohort_and_config(context, predictions):
    duplicate = pd.concat([predictions, predictions.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        decision_gate(duplicate, context, **WINDOW)
    empty, report = decision_gate(predictions.iloc[:0], context, **WINDOW)
    assert empty.empty and report["total_customers_evaluated"] == 0
    assert report["context_coverage"]["mapped_customer_rate"] is None
    for kwargs in (
        {"high_risk_threshold": np.nan},
        {"missing_event_policy": "clear"},
        {"relevant_meter_types": ["invented"]},
    ):
        with pytest.raises(ValueError):
            replace(GateConfig(), **kwargs)


def test_saved_predictions_unchanged_and_cli_integration(context, predictions, tmp_path, capsys):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    for name, frame in context.tables.items():
        frame.to_csv(context_dir / f"{name}.csv", index=False)
    (context_dir / "synthetic_context.json").write_text(json.dumps(context.metadata))
    source = tmp_path / "test_predictions.csv"
    predictions.to_csv(source, index=False)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    output = tmp_path / "decisions"
    config = tmp_path / "strict.json"
    config.write_text('{"missing_event_policy": "verify"}')
    main(
        [
            "context-gate",
            "--predictions",
            str(source),
            "--context-dir",
            str(context_dir),
            "--output-dir",
            str(output),
            "--evaluation-date",
            WINDOW["evaluation_date"],
            "--window-start",
            WINDOW["window_start"],
            "--window-end",
            WINDOW["window_end"],
            "--config",
            str(config),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["customers_per_decision_status"]["needs_context_verification"] == 3
    saved = pd.read_csv(output / "context_decisions.csv")
    assert saved.customer_id.tolist() == CUSTOMERS
    assert saved.original_risk_score.tolist() == predictions.risk_score.tolist()
    assert json.loads(saved.matched_event_ids.iloc[0]) == ["SYN-SUB-0"]
    for path, payload in before.items():
        assert path.read_bytes() == payload
    with pytest.raises(ValueError, match="empty"):
        save_decisions(source, context_dir=context_dir, output_dir=output, **WINDOW)
    # A missing source file is treated as unknown, not proof of no events.
    (context_dir / "subscriptions.csv").unlink()
    report = save_decisions(
        source, context_dir=context_dir, output_dir=tmp_path / "missing", **WINDOW
    )
    assert report["customers_per_decision_status"]["insufficient_data"] == 3
