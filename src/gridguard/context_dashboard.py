"""Read-only operational-context parsing, integrity checks and display transforms."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridguard.dashboard import file_signature
from gridguard.data import load_customer_history
from gridguard.decision_gate import EVENT_TYPES, STATUSES, GateConfig, _day_end, decision_gate
from gridguard.operational import (
    PROVENANCE,
    TABLE_COLUMNS,
    event_bounds,
    load_context,
    record_issues,
    records,
    time_value,
)
from gridguard.provenance import file_fingerprint, partition_fingerprint
from gridguard.reliability import summaries_match

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECISION_FILES = ("context_decisions.csv", "context_evaluation.json")
CONTEXT_FILES = ("synthetic_context.json", *(f"{name}.csv" for name in TABLE_COLUMNS))
DECISION_COLUMNS = (
    "customer_id",
    "original_risk_score",
    "decision_status",
    "reason_codes",
    "matched_event_ids",
    "evidence_status",
    "provenance",
)
EVENT_COLUMNS = (
    "customer_id",
    "event_id",
    "event_type",
    "start",
    "end",
    "verification_status",
    "provenance",
)
REASON_CODES = {
    "synthetic_context_only",
    "invalid_or_missing_risk_score",
    "missing_or_invalid_region_mapping",
    "invalid_provenance",
    "orphan_customer_reference",
    "invalid_region_reference",
    "missing_or_invalid_availability",
    "missing_event_id",
    "invalid_event_interval",
    "outside_observation_period",
    "invalid_event_type",
    "invalid_subscription_state",
    "invalid_verification_status",
    "duplicate_event_id",
    "invalid_context_records",
    "unlinked_invalid_context",
    "conflicting_subscription_intervals",
    "subscription_coverage_unknown",
    "no_matching_outages_unknown",
    "no_matching_meter_events_unknown",
    "missing_evidence",
    "unverified_evidence",
    "active_subscription_context",
    "configured_irrelevant_meter_event",
    *(f"overlapping_{kind}" for kind in EVENT_TYPES if kind != "active"),
}


@dataclass
class ContextView:
    report: dict
    decisions: pd.DataFrame
    events: pd.DataFrame
    warnings: list[str]


def local_path(value, root=PROJECT_ROOT, *, directory=False, suffix=None):
    """Confine local selections to a server-configured root, including symlink targets."""
    text = str(value)
    if (
        not text.strip()
        or text.startswith(("\\\\", "//"))
        or "://" in text
        or any(ord(char) < 32 for char in text)
    ):
        raise ValueError("Choose a local path inside the configured data root.")
    raw = Path(text)
    blocked = {".git", ".agents", ".codex", ".aws", ".ssh", ".streamlit"}
    parts = raw.parts[1:] if raw.drive else raw.parts
    if any(
        part == ".." or ":" in part or part.lower() in blocked or part.lower().startswith(".env")
        for part in parts
    ):
        raise ValueError("Unsafe local path.")
    root = Path(root).resolve(strict=True)
    path = (raw if raw.is_absolute() else root / raw).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Choose a path inside the configured data root.")
    if any(
        part.lower() in blocked or part.lower().startswith(".env")
        for part in path.relative_to(root).parts
    ):
        raise ValueError("Unsafe resolved local path.")
    if directory and not path.is_dir():
        raise ValueError("Choose a local directory.")
    if not directory and (not path.is_file() or (suffix and path.suffix.lower() != suffix)):
        raise ValueError("Choose a local file with the expected extension.")
    return path


def context_signature(decision_dir, context_dir, *, root=PROJECT_ROOT, predictions_path=None):
    """Validate every fixed input before consulting caches; include every consumed file."""
    selected = []
    for folder, names in ((decision_dir, DECISION_FILES), (context_dir, CONTEXT_FILES)):
        folder = local_path(folder, root, directory=True)
        for name in names:
            path = folder / name
            if path.exists() or path.is_symlink():
                path = local_path(path, root, suffix=path.suffix)
                limit = 2 * 1024**2 if path.suffix == ".json" else 64 * 1024**2
                if path.stat().st_size > limit:
                    raise ValueError("Context file exceeds the local dashboard size limit.")
        selected.append((str(folder), file_signature(folder, names=names)))
    if predictions_path:
        path = local_path(predictions_path, root, suffix=".csv")
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError("Prediction CSV exceeds the local dashboard size limit.")
        selected.append((str(path), file_signature(path.parent, names=(path.name,))))
    return tuple(selected)


def _json_object(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON report object.")
    return result


def _string_list(value):
    result = json.loads(value)
    if (
        not isinstance(result, list)
        or len(result) > 1000
        or any(not isinstance(item, str) or not item or len(item) > 160 for item in result)
        or len(result) != len(set(result))
    ):
        raise ValueError("Expected a bounded JSON list of unique strings.")
    return result


def load_context_view(decision_dir, context_dir, *, root=PROJECT_ROOT, predictions_path=None):
    """Read saved outputs; replay only the pure gate, never a model or training workflow.

    Incomplete bundles retain the saved decisions but explicitly lose replay
    verification. Missing referenced evidence is never manufactured.
    """
    context_signature(decision_dir, context_dir, root=root, predictions_path=predictions_path)
    decision_dir = local_path(decision_dir, root, directory=True)
    context_dir = local_path(context_dir, root, directory=True)
    report = _json_object(decision_dir / "context_evaluation.json")
    if report.get("schema_version") != 1 or report.get("provenance") != PROVENANCE:
        raise ValueError("Expected schema-v1 SYNTHETIC gate outputs.")
    for key in ("source_sha256", "predictions_sha256"):
        if not isinstance(report.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", report[key]):
            raise ValueError("Missing or invalid source/prediction fingerprint.")
    decisions = pd.read_csv(
        decision_dir / "context_decisions.csv",
        dtype={"customer_id": "string"},
        keep_default_na=False,
        float_precision="round_trip",
    )
    if not set(DECISION_COLUMNS) <= set(decisions):
        raise ValueError("Missing saved decision columns.")
    # Never carry arbitrary source columns (labels, private notes, etc.) into the view.
    decisions = decisions.loc[:, list(DECISION_COLUMNS)].copy()
    ids = decisions.customer_id
    if (
        ids.eq("").any()
        or ids.str.strip().ne(ids).any()
        or ids.duplicated().any()
        or not decisions.provenance.eq(PROVENANCE).all()
        or not decisions.decision_status.isin(STATUSES).all()
        or not decisions.evidence_status.isin(
            ("synthetic_only", "synthetic_unverified", "synthetic_partial_unknown")
        ).all()
    ):
        raise ValueError("Invalid IDs, decision/evidence states or SYNTHETIC provenance.")
    for name in ("reason_codes", "matched_event_ids"):
        decisions[name] = decisions[name].map(_string_list)
    if any("synthetic_context_only" not in codes for codes in decisions.reason_codes):
        raise ValueError("Every decision must declare synthetic context.")
    if any(not set(codes) <= REASON_CODES for codes in decisions.reason_codes):
        raise ValueError(
            "Unknown reason codes; free-form customer details are not displayable evidence codes."
        )
    settings = GateConfig(**report["rules"])
    start, end = time_value(report["window_start"]), _day_end(report["window_end"])
    cutoff = _day_end(report["evaluation_date"])
    if any(pd.isna(value) for value in (start, end, cutoff)) or not start < end <= cutoff:
        raise ValueError("Invalid saved context evaluation window.")
    counts = {status: int(decisions.decision_status.eq(status).sum()) for status in STATUSES}
    scores = pd.to_numeric(decisions.original_risk_score, errors="coerce").astype(float)
    high = int(
        (
            np.isfinite(scores)
            & scores.between(settings.high_risk_threshold, 100)
            & decisions.decision_status.ne("eligible_for_review")
        ).sum()
    )
    invalid_scores = ~np.isfinite(scores) | ~scores.between(0, 100)
    if (
        report["total_customers_evaluated"] != len(decisions)
        or report["customers_per_decision_status"] != counts
        or report["high_risk_customers_requiring_verification"] != high
        or (invalid_scores & decisions.decision_status.ne("insufficient_data")).any()
    ):
        raise ValueError("Saved decision totals or scores disagree with the evaluation report.")
    for value in report["context_coverage"].values():
        if value is not None and (
            not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1
        ):
            raise ValueError("Invalid context coverage rate.")
    warnings = []
    if predictions_path:
        path = local_path(predictions_path, root, suffix=".csv")
        if file_fingerprint(path) != report["predictions_sha256"]:
            raise ValueError("Original prediction fingerprint mismatch.")
        original = pd.read_csv(
            path,
            usecols=["customer_id", "risk_score"],
            dtype={"customer_id": "string"},
            keep_default_na=False,
            float_precision="round_trip",
        )
        if (
            original.customer_id.tolist() != ids.tolist()
            or original.risk_score.tolist() != decisions.original_risk_score.tolist()
        ):
            raise ValueError("Original prediction order or scores differ from saved decisions.")
    else:
        warnings.append(
            "Original prediction file not supplied: saved ML score provenance is unverified."
        )
    missing = [name for name in CONTEXT_FILES if not (context_dir / name).is_file()]
    events = pd.DataFrame(columns=EVENT_COLUMNS)
    if "synthetic_context.json" in missing:
        warnings.append(
            "SYNTHETIC metadata is missing: operational events and replay validation are unavailable."
        )
        return ContextView(report, decisions, events, warnings)
    bundle = load_context(context_dir)
    if bundle.metadata["source_sha256"] != report["source_sha256"]:
        raise ValueError("Context and decisions use different SGCC sources.")
    if start < time_value(bundle.metadata["date_start"]) or end > _day_end(
        bundle.metadata["date_end"]
    ):
        raise ValueError("Saved window lies outside the source observation period.")
    if missing:
        warnings.append(
            "Missing SYNTHETIC context files: "
            + ", ".join(missing)
            + ". Saved decisions retained; replay integrity is unverified and unresolved evidence is unknown."
        )
    else:
        population = bundle.tables["customer_regions"].customer_id
        if (
            len(population) != bundle.metadata["sgcc_customers"]
            or partition_fingerprint(sorted(population.dropna()))
            != bundle.metadata["population_fingerprint"]
        ):
            raise ValueError("Source customer population fingerprint mismatch.")
        replay, summary = decision_gate(
            decisions[["customer_id", "original_risk_score"]].rename(
                columns={"original_risk_score": "risk_score"}
            ),
            bundle,
            config=settings,
            evaluation_date=report["evaluation_date"],
            window_start=report["window_start"],
            window_end=report["window_end"],
        )
        for name in ("decision_status", "evidence_status", "reason_codes", "matched_event_ids"):
            saved = (
                decisions[name].map(sorted)
                if name in ("reason_codes", "matched_event_ids")
                else decisions[name]
            )
            if saved.tolist() != replay[name].tolist():
                raise ValueError("Saved decisions disagree with SYNTHETIC context gate replay.")
        if not summaries_match(summary, {key: report.get(key) for key in summary}):
            raise ValueError("Saved context evaluation disagrees with gate replay.")
    mapping = bundle.tables["customer_regions"]
    customers, regions = set(mapping.customer_id.dropna()), set(mapping.region_id.dropna())
    region_for = {
        row["customer_id"]: row["region_id"]
        for row in records(mapping)
        if pd.notna(time_value(row["available_at"]))
        and time_value(row["available_at"]) < cutoff
        and not record_issues("customer_regions", row, bundle.metadata, customers, regions)
    }
    wanted = {event_id for values in decisions.matched_event_ids for event_id in values}
    catalog = {}
    for name, id_column in (
        ("outages", "outage_id"),
        ("meter_events", "event_id"),
        ("subscriptions", "subscription_id"),
    ):
        for row in records(bundle.tables[name]):
            event_id = row[id_column]
            if event_id not in wanted:
                continue
            # With a missing mapping, direct customer evidence can still be validated locally.
            known_customers = customers if "customer_regions.csv" not in missing else set(ids)
            if record_issues(name, row, bundle.metadata, known_customers, regions):
                continue
            begin, finish = event_bounds(name, row)
            available = time_value(row["available_at"])
            if available >= cutoff or not (begin < end and finish > start):
                continue
            if event_id in catalog:
                raise ValueError("Ambiguous matched event identifiers.")
            kind = (
                f"{row['outage_type']}_outage"
                if name == "outages"
                else (row["event_type"] if name == "meter_events" else row["subscription_status"])
            )
            catalog[event_id] = (row, kind, begin, finish)
    linked = []
    for decision in decisions.itertuples():
        for event_id in decision.matched_event_ids:
            if event_id not in catalog:
                continue
            row, kind, begin, finish = catalog[event_id]
            if row.get("customer_id", decision.customer_id) != decision.customer_id or (
                "region_id" in row and row["region_id"] != region_for.get(decision.customer_id)
            ):
                raise ValueError("Matched event customer or region differs from the decision.")
            linked.append(
                [
                    decision.customer_id,
                    event_id,
                    kind,
                    begin,
                    finish,
                    row.get("verification_status") or "missing",
                    PROVENANCE,
                ]
            )
    events = pd.DataFrame(linked, columns=EVENT_COLUMNS)
    return ContextView(report, decisions, events, warnings)


def filter_context(
    view,
    *,
    risk_range=(0, 100),
    statuses=STATUSES,
    event_types=(),
    verifications=(),
    search="",
    include_missing_scores=True,
):
    """Preserve saved order. Type/verification predicates must match the same event."""
    frame = view.decisions
    scores = pd.to_numeric(frame.original_risk_score, errors="coerce")
    risk = scores.between(*risk_range)
    if include_missing_scores:
        risk |= ~np.isfinite(scores) | ~scores.between(0, 100)
    selected = frame.loc[risk & frame.decision_status.isin(statuses)]
    if search:
        selected = selected.loc[selected.customer_id.str.contains(search, regex=False, na=False)]
    if event_types or verifications:
        events = view.events
        if event_types:
            events = events.loc[events.event_type.isin(event_types)]
        if verifications:
            events = events.loc[events.verification_status.isin(verifications)]
        selected = selected.loc[selected.customer_id.isin(events.customer_id)]
    return selected.copy()


def inspection_frame(decisions, events, *, reveal_ids=False):
    """Allowlist before browser serialization; never hide sensitive columns via CSS/config."""
    result = decisions.loc[
        :, ["original_risk_score", "decision_status", "evidence_status", "reason_codes"]
    ].copy()
    result.insert(
        0,
        "customer_id",
        decisions.customer_id
        if reveal_ids
        else [f"Customer {index + 1}" for index in decisions.index],
    )
    result["original_risk_score"] = pd.to_numeric(result.original_risk_score, errors="coerce")
    matched = events.groupby("customer_id").event_type.agg(lambda values: sorted(set(values)))
    result["matched_operational_events"] = decisions.customer_id.map(matched).map(
        lambda value: value if isinstance(value, list) else []
    )
    available = events.groupby("customer_id").size()
    for index, row in decisions.iterrows():
        if available.get(row.customer_id, 0) < len(row.matched_event_ids):
            result.at[index, "matched_operational_events"] = [
                *result.at[index, "matched_operational_events"],
                "unknown / unresolved evidence",
            ]
    result["provenance"] = PROVENANCE
    return result


def customer_events(view, customer_id, *, reveal_ids=False):
    result = view.events.loc[view.events.customer_id.eq(customer_id)].drop(columns="customer_id")
    if not reveal_ids:
        result = result.drop(columns="event_id")
    return result.copy()


def verified_context_history(path, customer_id, expected_sha256):
    """Call only after path confinement; read one customer using the existing bounded loader."""
    if file_fingerprint(path) != expected_sha256:
        raise ValueError(
            "Consumption source fingerprint differs from the SYNTHETIC context source."
        )
    return load_customer_history(path, customer_id)
