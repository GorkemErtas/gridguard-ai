"""Deterministic post-model review gate; never fits, scores or reorders a model."""

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gridguard.operational import (
    METER_TYPES,
    NOTICE,
    PROVENANCE,
    empty_output,
    event_bounds,
    load_context,
    record_issues,
    records,
    time_value,
)
from gridguard.provenance import file_fingerprint, write_json

STATUSES = ("eligible_for_review", "needs_context_verification", "insufficient_data")
EVENT_TYPES = ("planned_outage", "unplanned_outage", *METER_TYPES, "active", "terminated")


@dataclass(frozen=True)
class GateConfig:
    high_risk_threshold: float = 80.0
    # Annotation permits model review with explicit unknown outage/meter coverage.
    # Strict policy requires verification whenever either source has no match.
    missing_event_policy: str = "annotate"
    relevant_meter_types: tuple[str, ...] = METER_TYPES

    def __post_init__(self):
        if not np.isfinite(self.high_risk_threshold) or not 0 <= self.high_risk_threshold <= 100:
            raise ValueError("high_risk_threshold must be finite and in [0, 100].")
        if self.missing_event_policy not in ("annotate", "verify"):
            raise ValueError("missing_event_policy must be annotate or verify.")
        if not set(self.relevant_meter_types) <= set(METER_TYPES):
            raise ValueError("Unknown relevant meter event type.")


def _day_end(value):
    result = time_value(value)
    if pd.notna(result) and len(str(value)) == 10:
        result += pd.Timedelta(days=1)
    else:
        result += pd.Timedelta(nanoseconds=1)
    return result


def _covered(intervals, start, end):
    cursor = start
    for begin, finish in sorted(intervals):
        if begin > cursor:
            return False
        cursor = max(cursor, finish)
    return cursor >= end


def decision_gate(predictions, bundle, *, evaluation_date, window_start, window_end, config=None):
    """Return aligned decisions and aggregate diagnostics without changing predictions.

    Date-only cutoffs include the entire UTC day. Event/window intersections are
    half-open; verification describes the row snapshot known at available_at.
    Existing predictions can be retrospective: this is not prospective ML validation.
    """
    config = config or GateConfig()
    if bundle.metadata.get("provenance") != PROVENANCE:
        raise ValueError("Decision gate requires explicitly SYNTHETIC context.")
    cutoff, start, end = _day_end(evaluation_date), time_value(window_start), _day_end(window_end)
    if any(pd.isna(value) for value in (cutoff, start, end)) or not start < end <= cutoff:
        raise ValueError("Require a valid window_start <= window_end <= evaluation_date.")
    lower = time_value(bundle.metadata["date_start"])
    upper = time_value(bundle.metadata["date_end"]) + pd.Timedelta(days=1)
    if start < lower or end > upper:
        raise ValueError("Evaluation window must lie within the SGCC observation period.")
    if not {"customer_id", "risk_score"} <= set(predictions):
        raise ValueError("Saved predictions require customer_id and risk_score.")
    ids = predictions.customer_id
    if (
        not all(isinstance(value, str) for value in ids)
        or ids.isna().any()
        or ids.eq("").any()
        or ids.str.strip().ne(ids).any()
        or ids.duplicated().any()
    ):
        raise ValueError("Prediction customer IDs must be unique, nonempty strings.")
    numeric = pd.to_numeric(predictions.risk_score, errors="coerce").astype(float)
    valid_scores = np.isfinite(numeric) & numeric.between(0, 100)
    mapping = bundle.tables["customer_regions"]
    customer_rows, regional_rows = defaultdict(list), defaultdict(list)
    invalid_customers, invalid_regions = defaultdict(set), defaultdict(set)
    mapping_for, bad_mapping = {}, set()
    quality = Counter()
    available_counts = Counter()
    verification_counts = Counter({"verified": 0, "unverified": 0, "missing": 0})
    ignored_future = Counter()
    unlinked_invalid = 0

    available_mapping = []
    for row in records(mapping):
        available = time_value(row.get("available_at"))
        if pd.notna(available) and available >= cutoff:
            ignored_future["customer_regions"] += 1
            continue
        available_mapping.append(row)
    customers = {row["customer_id"] for row in available_mapping if row.get("customer_id")}
    regions = {row["region_id"] for row in available_mapping if row.get("region_id")}
    for row in available_mapping:
        errors = record_issues("customer_regions", row, bundle.metadata, customers, regions)
        if errors:
            quality.update(errors)
            bad_mapping.add(row["customer_id"])
            continue
        customer = row["customer_id"]
        if customer in mapping_for:
            bad_mapping.add(customer)
            quality["duplicate_customer_mapping"] += 1
        mapping_for[customer] = row["region_id"]

    for table in ("outages", "meter_events", "subscriptions"):
        # Future rows are removed before duplicate/conflict checks and summaries.
        candidates = []
        for row in records(bundle.tables[table]):
            available = time_value(row.get("available_at"))
            if pd.notna(available) and available >= cutoff:
                ignored_future[table] += 1
                continue
            candidates.append(row)
        id_col = {
            "outages": "outage_id",
            "meter_events": "event_id",
            "subscriptions": "subscription_id",
        }[table]
        duplicates = Counter(row.get(id_col) for row in candidates)
        for row in candidates:
            errors = record_issues(table, row, bundle.metadata, customers, regions)
            if duplicates[row.get(id_col)] > 1:
                errors.append("duplicate_event_id")
            if errors:
                quality.update(errors)
                target = row.get("region_id" if table == "outages" else "customer_id")
                lookup = invalid_regions if table == "outages" else invalid_customers
                if target in (regions if table == "outages" else customers):
                    lookup[target].update(errors)
                else:
                    unlinked_invalid += 1
                continue
            begin, finish = event_bounds(table, row)
            if not (begin < end and finish > start):
                continue
            verification = row.get("verification_status") or "missing"
            available_counts[table] += 1
            verification_counts[verification] += 1
            item = {
                "id": row[id_col],
                "table": table,
                "start": begin,
                "end": finish,
                "verification": verification,
                "type": f"{row['outage_type']}_outage"
                if table == "outages"
                else (row["event_type"] if table == "meter_events" else row["subscription_status"]),
            }
            lookup, target = (
                (regional_rows, row["region_id"])
                if table == "outages"
                else (customer_rows, row["customer_id"])
            )
            lookup[target].append(item)

    rows = []
    affected = {kind: set() for kind in EVENT_TYPES}
    matched_records = {}
    no_event_customers = 0
    no_subscription_customers = 0
    coverage_customers = 0
    for position, customer in enumerate(ids):
        reasons = {"synthetic_context_only"}
        insufficient = not bool(valid_scores.iloc[position])
        needs_verification = False
        if insufficient:
            reasons.add("invalid_or_missing_risk_score")
        region = mapping_for.get(customer)
        if region is None or customer in bad_mapping:
            insufficient = True
            reasons.add("missing_or_invalid_region_mapping")
        invalid = invalid_customers[customer] | invalid_regions[region]
        if invalid:
            insufficient = True
            reasons.update(invalid)
            reasons.add("invalid_context_records")
        if unlinked_invalid:
            needs_verification = True
            reasons.add("unlinked_invalid_context")
        matched = customer_rows[customer] + regional_rows[region]
        subscriptions = [item for item in matched if item["table"] == "subscriptions"]
        if not subscriptions:
            no_subscription_customers += 1
        periods = sorted((item["start"], item["end"]) for item in subscriptions)
        # Detect any overlap, including a long interval containing shorter ones.
        latest_end = None
        conflict = False
        for begin, finish in periods:
            if latest_end is not None and begin < latest_end:
                conflict = True
            latest_end = finish if latest_end is None else max(latest_end, finish)
        if conflict:
            insufficient = True
            reasons.add("conflicting_subscription_intervals")
        coverage = _covered(periods, start, end)
        if coverage:
            coverage_customers += 1
        else:
            insufficient = True
            reasons.add("subscription_coverage_unknown")
        source_missing = False
        for table in ("outages", "meter_events"):
            if not any(item["table"] == table for item in matched):
                reasons.add(f"no_matching_{table}_unknown")
                source_missing = True
                if config.missing_event_policy == "verify":
                    needs_verification = True
        if source_missing:
            no_event_customers += 1
        for item in matched:
            affected[item["type"]].add(customer)
            matched_records[(item["table"], item["id"])] = item
            verification = item["verification"]
            if verification != "verified":
                reasons.add(f"{verification}_evidence")
                needs_verification = True
            disruptive = (
                item["table"] == "outages"
                or item["type"] == "terminated"
                or (item["type"] in config.relevant_meter_types)
            )
            if disruptive:
                reasons.add(f"overlapping_{item['type']}")
                # Even 'verified' synthetic records need real-world confirmation.
                needs_verification = True
            elif item["type"] == "active":
                reasons.add("active_subscription_context")
            else:
                reasons.add("configured_irrelevant_meter_event")
        status = (
            "insufficient_data"
            if insufficient
            else ("needs_context_verification" if needs_verification else "eligible_for_review")
        )
        verifications = {item["verification"] for item in matched}
        evidence = "synthetic_only"
        if insufficient or source_missing or "missing" in verifications:
            evidence = "synthetic_partial_unknown"
        elif "unverified" in verifications:
            evidence = "synthetic_unverified"
        rows.append(
            {
                "customer_id": customer,
                "original_risk_score": predictions.risk_score.iloc[position],
                "decision_status": status,
                "reason_codes": sorted(reasons),
                "matched_event_ids": sorted({item["id"] for item in matched}),
                "evidence_status": evidence,
                "provenance": PROVENANCE,
            }
        )
    decisions = pd.DataFrame(
        rows,
        columns=[
            "customer_id",
            "original_risk_score",
            "decision_status",
            "reason_codes",
            "matched_event_ids",
            "evidence_status",
            "provenance",
        ],
    )
    # Keep the original dtype, including integer scores, and never sort this output.
    decisions["original_risk_score"] = predictions.risk_score.to_numpy(copy=True)
    status_counts = decisions.decision_status.value_counts()
    matched_verification = Counter(item["verification"] for item in matched_records.values())
    n = len(decisions)
    report = {
        "schema_version": 1,
        "provenance": PROVENANCE,
        "notice": NOTICE,
        "source_sha256": bundle.metadata["source_sha256"],
        "evaluation_date": str(evaluation_date),
        "window_start": str(window_start),
        "window_end": str(window_end),
        "rules": asdict(config),
        "total_customers_evaluated": n,
        "customers_per_decision_status": {
            status: int(status_counts.get(status, 0)) for status in STATUSES
        },
        "customers_affected_by_event_type": {
            kind: len(customers) for kind, customers in affected.items()
        },
        "matched_event_verification_counts": {
            key: matched_verification[key] for key in ("verified", "unverified", "missing")
        },
        "available_window_event_counts": dict(available_counts),
        "available_window_verification_counts": dict(verification_counts),
        "high_risk_customers_requiring_verification": int(
            (
                valid_scores.to_numpy()
                & (numeric.to_numpy() >= config.high_risk_threshold)
                & decisions.decision_status.ne("eligible_for_review").to_numpy()
            ).sum()
        ),
        "context_coverage": {
            "mapped_customer_rate": sum(
                cid in mapping_for and cid not in bad_mapping for cid in ids
            )
            / n
            if n
            else None,
            "subscription_window_coverage_rate": coverage_customers / n if n else None,
            "missing_subscription_record_rate": no_subscription_customers / n if n else None,
            "missing_outage_or_meter_match_rate": no_event_customers / n if n else None,
            "unknown_evidence_rate": float(
                decisions.evidence_status.eq("synthetic_partial_unknown").mean()
            )
            if n
            else None,
        },
        "invalid_record_issue_counts": dict(sorted(quality.items())),
        "unlinked_invalid_records": unlinked_invalid,
        "ignored_future_records": dict(ignored_future),
        "ranking_policy": "Original input order and risk_score preserved; no reranking.",
        "prediction_scope": "Saved historical scores; as-of filtering applies to context, not model training time.",
    }
    return decisions, report


def save_decisions(
    predictions_path,
    *,
    context_dir,
    output_dir,
    evaluation_date,
    window_start,
    window_end,
    config=None,
):
    output = empty_output(output_dir, protected=(predictions_path, context_dir))
    predictions = pd.read_csv(
        predictions_path, dtype={"customer_id": "string"}, keep_default_na=False
    )
    decisions, report = decision_gate(
        predictions,
        load_context(context_dir),
        evaluation_date=evaluation_date,
        window_start=window_start,
        window_end=window_end,
        config=config,
    )
    report["predictions_sha256"] = file_fingerprint(predictions_path)
    serialized = decisions.copy()
    for col in ("reason_codes", "matched_event_ids"):
        serialized[col] = serialized[col].map(json.dumps)
    output.mkdir(parents=True, exist_ok=True)
    serialized.to_csv(output / "context_decisions.csv", index=False)
    write_json(output / "context_evaluation.json", report)
    return report
