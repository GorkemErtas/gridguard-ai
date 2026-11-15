"""Independent SYNTHETIC operational context aligned to a local SGCC CSV.

Generation reads only identifiers and date headers. Reading-pattern analysis is
a separate, subsequent pass; labels and predictions are never loaded here.
"""

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from gridguard.data import _parse_date
from gridguard.provenance import file_fingerprint, partition_fingerprint, write_json

PROVENANCE = "SYNTHETIC"
NOTICE = (
    "SYNTHETIC context only; not real operational evidence. No inference of innocence, "
    "fraud or improved ML performance. Saved model scores and rankings are unchanged."
)
METER_TYPES = ("communication_failure", "meter_fault", "meter_replacement")
TABLE_COLUMNS = {
    "customer_regions": ["customer_id", "region_id", "available_at", "provenance"],
    "outages": [
        "outage_id",
        "region_id",
        "start_time",
        "end_time",
        "outage_type",
        "verification_status",
        "available_at",
        "provenance",
    ],
    "meter_events": [
        "event_id",
        "customer_id",
        "event_type",
        "start_date",
        "end_date",
        "verification_status",
        "available_at",
        "provenance",
    ],
    "subscriptions": [
        "subscription_id",
        "customer_id",
        "start_date",
        "end_date",
        "subscription_status",
        "verification_status",
        "available_at",
        "provenance",
    ],
}


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 42
    regions: int = 20
    outages_per_region_year: float = 6.0
    communication_failures_per_customer_year: float = 0.12
    meter_faults_per_customer_year: float = 0.05
    replacements_per_customer_year: float = 0.07
    missing_subscription_rate: float = 0.04
    terminated_rate: float = 0.08
    unverified_rate: float = 0.18
    missing_verification_rate: float = 0.08

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if type(self.regions) is not int or self.regions < 1:
            raise ValueError("regions must be a positive integer.")
        for key, value in asdict(self).items():
            if key in ("seed", "regions"):
                continue
            if not np.isfinite(value) or value < 0 or (key.endswith("rate") and value > 1):
                raise ValueError(f"Invalid generation setting: {key}")
        if self.unverified_rate + self.missing_verification_rate > 1:
            raise ValueError("Verification rates must sum to at most one.")


@dataclass(frozen=True)
class ContextBundle:
    tables: dict[str, pd.DataFrame]
    metadata: dict


def source_population(path):
    """Read exact lexical IDs (including leading zeroes/NA-like strings), no FLAG."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    id_column = next((col for col in ("CONS_NO", "customer_id") if col in header), None)
    if id_column is None:
        raise ValueError("Expected CONS_NO or customer_id.")
    dates = []
    for col in header:
        if col in (id_column, "FLAG", "flag"):
            continue
        date = _parse_date(col)
        if date is None:
            raise ValueError(f"Unexpected consumption column: {col}")
        dates.append((pd.Timestamp(date), col))
    if len(dates) < 60 or len({date for date, _ in dates}) != len(dates):
        raise ValueError("Expected at least 60 unique daily reading columns.")
    ids = pd.read_csv(
        path, usecols=[id_column], dtype={id_column: "string"}, keep_default_na=False
    )[id_column]
    if ids.eq("").any() or ids.str.strip().ne(ids).any() or ids.duplicated().any() or ids.empty:
        raise ValueError("Expected nonempty, unique IDs without surrounding whitespace.")
    return ids, id_column, sorted(dates)


def _frame(name, rows):
    frame = pd.DataFrame(rows, columns=TABLE_COLUMNS[name])
    for col in frame:
        frame[col] = frame[col].astype("string")
    return frame


def generate_context(input_path, *, config=None, reference_metrics=None) -> ContextBundle:
    """Seeded generation independent of readings, labels, splits and predictions."""
    config = config or GenerationConfig()
    ids, _, dates = source_population(input_path)
    fingerprint = file_fingerprint(input_path)
    if reference_metrics is not None:
        reference = json.loads(Path(reference_metrics).read_text(encoding="utf-8"))
        if reference["dataset"]["fingerprint_sha256"] != fingerprint:
            raise ValueError("SGCC fingerprint differs from the reference experiment.")
    first, last = dates[0][0], dates[-1][0]
    days = (last - first).days + 1
    years = days / 365.25
    rng = np.random.default_rng(config.seed)
    rows = {name: [] for name in TABLE_COLUMNS}

    def verification():
        value = rng.random()
        if value < config.missing_verification_rate:
            return None
        return (
            "unverified"
            if value < (config.missing_verification_rate + config.unverified_rate)
            else "verified"
        )

    def interval(low, high):
        length = min(int(rng.integers(low, high + 1)), days)
        start = first + pd.Timedelta(days=int(rng.integers(0, days - length + 1)))
        return start, start + pd.Timedelta(days=length - 1)

    def iso(value):
        return value.isoformat()

    regions = [f"SYN-REG-{i + 1:03d}" for i in range(min(config.regions, len(ids)))]
    # Sorted IDs + shuffled balanced assignments: stable under source row reordering.
    assignments = np.resize(np.array(regions), len(ids))
    rng.shuffle(assignments)
    for customer, region in zip(sorted(ids), assignments):
        rows["customer_regions"].append([customer, region, iso(first), PROVENANCE])

    for region in regions:
        for _ in range(rng.poisson(config.outages_per_region_year * years)):
            day = first + pd.Timedelta(days=int(rng.integers(days)))
            start = day + pd.Timedelta(hours=int(rng.integers(0, 12)))
            planned = rng.random() < 0.4
            hours = int(rng.integers(2, 9) if planned else rng.integers(1, 25))
            end = min(start + pd.Timedelta(hours=hours), last + pd.Timedelta(days=1))
            # Whole row, including final verification, is known only when recorded.
            available = end + pd.Timedelta(days=int(rng.integers(0, 4)))
            rows["outages"].append(
                [
                    f"SYN-OUT-{len(rows['outages']) + 1:07d}",
                    region,
                    iso(start),
                    iso(end),
                    "planned" if planned else "unplanned",
                    verification(),
                    iso(available),
                    PROVENANCE,
                ]
            )

    rates = (
        config.communication_failures_per_customer_year,
        config.meter_faults_per_customer_year,
        config.replacements_per_customer_year,
    )
    for customer in sorted(ids):
        for event_type, rate, duration in zip(METER_TYPES, rates, ((2, 10), (3, 21), (1, 1))):
            for _ in range(rng.poisson(rate * years)):
                start, end = interval(*duration)
                available = end + pd.Timedelta(days=1 + int(rng.integers(0, 4)))
                rows["meter_events"].append(
                    [
                        f"SYN-MTR-{len(rows['meter_events']) + 1:07d}",
                        customer,
                        event_type,
                        start.date().isoformat(),
                        end.date().isoformat(),
                        verification(),
                        iso(available),
                        PROVENANCE,
                    ]
                )
        if rng.random() < config.missing_subscription_rate:
            continue  # Absence deliberately means unknown coverage.
        start = first + pd.Timedelta(days=int(rng.integers(0, min(31, days))))
        terminated = rng.random() < config.terminated_rate
        boundary = first + pd.Timedelta(days=int(rng.integers(max(31, days // 2), days)))
        periods = (
            [(start, boundary - pd.Timedelta(days=1), "active"), (boundary, last, "terminated")]
            if terminated
            else [(start, last, "active")]
        )
        for begin, end, status in periods:
            rows["subscriptions"].append(
                [
                    f"SYN-SUB-{len(rows['subscriptions']) + 1:07d}",
                    customer,
                    begin.date().isoformat(),
                    end.date().isoformat(),
                    status,
                    verification(),
                    iso(begin),
                    PROVENANCE,
                ]
            )
    tables = {name: _frame(name, values) for name, values in rows.items()}
    metadata = {
        "schema_version": 1,
        "provenance": PROVENANCE,
        "notice": NOTICE,
        "source_sha256": fingerprint,
        "customer_id_dtype": "string",
        "population_fingerprint": partition_fingerprint(sorted(ids)),
        "sgcc_customers": len(ids),
        "date_start": first.date().isoformat(),
        "date_end": last.date().isoformat(),
        "reading_columns": len(dates),
        "missing_calendar_days": days - len(dates),
        "config": asdict(config),
        "availability_policy": "available_at dates the entire row, including verification_status",
    }
    return ContextBundle(tables, metadata)


@lru_cache(maxsize=32768)
def time_value(value):
    """All synthetic timestamps use UTC; missing/malformed timestamps stay unknown."""
    if isinstance(value, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ].+)?", value):
        return pd.NaT
    parsed = pd.to_datetime(value, format="ISO8601", utc=True, errors="coerce")
    return pd.NaT if pd.isna(parsed) else parsed


def event_bounds(table, row):
    if table == "outages":
        return time_value(row.get("start_time")), time_value(row.get("end_time"))
    start, end = time_value(row.get("start_date")), time_value(row.get("end_date"))
    return start, end + pd.Timedelta(days=1)  # Inclusive source end date -> half-open interval.


def records(frame):
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def record_issues(table, row, metadata, customers, regions):
    """Validate one row; the gate quarantines invalid evidence instead of trusting it."""
    issues = []
    if row.get("provenance") != PROVENANCE:
        issues.append("invalid_provenance")
    if "customer_id" in row and row["customer_id"] not in customers:
        issues.append("orphan_customer_reference")
    if "region_id" in row and row["region_id"] not in regions:
        issues.append("invalid_region_reference")
    if pd.isna(time_value(row.get("available_at"))):
        issues.append("missing_or_invalid_availability")
    if table == "customer_regions":
        return issues
    id_col = {
        "outages": "outage_id",
        "meter_events": "event_id",
        "subscriptions": "subscription_id",
    }[table]
    if not row.get(id_col) or pd.isna(row.get(id_col)):
        issues.append("missing_event_id")
    start, end = event_bounds(table, row)
    lower = time_value(metadata["date_start"])
    upper = time_value(metadata["date_end"]) + pd.Timedelta(days=1)
    if pd.isna(start) or pd.isna(end) or start >= end:
        issues.append("invalid_event_interval")
    elif start < lower or end > upper:
        issues.append("outside_observation_period")
    if table == "outages" and row.get("outage_type") not in ("planned", "unplanned"):
        issues.append("invalid_event_type")
    if table == "meter_events" and row.get("event_type") not in METER_TYPES:
        issues.append("invalid_event_type")
    if table == "subscriptions" and row.get("subscription_status") not in ("active", "terminated"):
        issues.append("invalid_subscription_state")
    verification = row.get("verification_status")
    if pd.notna(verification) and verification not in ("verified", "unverified"):
        issues.append("invalid_verification_status")
    return issues


def load_context(directory) -> ContextBundle:
    directory = Path(directory)
    metadata = json.loads((directory / "synthetic_context.json").read_text(encoding="utf-8"))
    if metadata.get("provenance") != PROVENANCE or metadata.get("schema_version") != 1:
        raise ValueError("Expected schema-v1 SYNTHETIC context.")
    tables = {}
    for name, columns in TABLE_COLUMNS.items():
        path = directory / f"{name}.csv"
        # A missing table is unknown, not evidence that no event happened.
        frame = (
            pd.read_csv(path, dtype="string", keep_default_na=False)
            if path.exists()
            else (_frame(name, []))
        )
        if set(columns) - set(frame.columns):
            raise ValueError(f"Missing schema columns in {name}.")
        tables[name] = frame.replace("", pd.NA)
    return ContextBundle(tables, metadata)


def empty_output(directory, *, protected=()):
    """Refuse source collisions and nonempty destinations, including historical artifacts."""
    directory = Path(directory)
    for path in protected:
        if path is not None and Path(path).resolve().is_relative_to(directory.resolve()):
            raise ValueError("Output directory must be separate from input artifacts.")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a new or empty output directory.")
    return directory


def compatibility_report(input_path, bundle, *, chunk_size=500):
    """Post-generation, label-free observation of zero/missing/drop overlap.

    Event-level counts mean at least one affected customer's calendar day.
    Drops are <50% of the preceding positive observed calendar day's reading.
    Active subscriptions are reported separately from disruptive events.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    ids, id_column, dates = source_population(input_path)
    if file_fingerprint(input_path) != bundle.metadata["source_sha256"]:
        raise ValueError("SGCC source fingerprint changed.")
    customers = set(ids)
    mapping = bundle.tables["customer_regions"]
    regions = set(mapping.region_id.dropna())
    mapped = set(mapping.customer_id) & customers
    issues = {}
    intervals = {}
    by_customer, by_region = {}, {}
    observation = pd.date_range(dates[0][0], dates[-1][0])
    for table in ("outages", "meter_events", "subscriptions"):
        for index, row in enumerate(records(bundle.tables[table])):
            key = (table, index)
            errors = record_issues(table, row, bundle.metadata, customers, regions)
            issues[key] = errors
            if errors:
                continue
            start, end = event_bounds(table, row)
            # A midnight outage end excludes the following day.
            lo = observation.searchsorted(start.tz_localize(None).normalize())
            hi = observation.searchsorted(
                (end - pd.Timedelta(nanoseconds=1)).tz_localize(None).normalize(), side="right"
            )
            intervals[key] = (lo, hi)
            lookup, foreign = (
                (by_region, row["region_id"])
                if table == "outages"
                else (by_customer, row["customer_id"])
            )
            lookup.setdefault(foreign, []).append(key)
    region_for = dict(zip(mapping.customer_id, mapping.region_id))
    overlaps = {name: set() for name in ("zero", "missing", "drop")}
    observed = set()
    usecols = [id_column] + [col for _, col in dates]
    for chunk in pd.read_csv(
        input_path,
        usecols=usecols,
        chunksize=chunk_size,
        dtype={id_column: "string"},
        keep_default_na=False,
    ):
        frame = chunk[[col for _, col in dates]].apply(pd.to_numeric, errors="coerce")
        frame.columns = [date for date, _ in dates]
        values = frame.reindex(columns=observation).to_numpy(dtype=float, copy=True)
        values[~np.isfinite(values) | (values < 0)] = np.nan
        drops = np.zeros_like(values, dtype=bool)
        drops[:, 1:] = (values[:, :-1] > 0) & (values[:, 1:] < 0.5 * values[:, :-1])
        masks = {"zero": values == 0, "missing": np.isnan(values), "drop": drops}
        sums = {name: np.pad(mask.cumsum(axis=1), ((0, 0), (1, 0))) for name, mask in masks.items()}
        for position, customer in enumerate(chunk[id_column]):
            matched = by_customer.get(customer, []) + by_region.get(region_for.get(customer), [])
            for key in matched:
                lo, hi = intervals[key]
                observed.add(key)
                for name, cumulative in sums.items():
                    if cumulative[position, hi] > cumulative[position, lo]:
                        overlaps[name].add(key)
    any_anomaly = set.union(*overlaps.values())
    zero_missing = overlaps["zero"] | overlaps["missing"]
    total = len(issues)
    within = sum(
        not any(
            issue in errors for issue in ("invalid_event_interval", "outside_observation_period")
        )
        for errors in issues.values()
    )
    return {
        "provenance": PROVENANCE,
        "notice": NOTICE,
        "source_sha256": bundle.metadata["source_sha256"],
        "sgcc_customers": len(ids),
        "mapped_customers": len(mapped),
        "customer_id_match_rate": len(mapped) / len(ids),
        "mapping_rows": len(mapping),
        "duplicate_mapping_rows": int(mapping.customer_id.duplicated().sum()),
        "consumption_date_coverage": {
            key: bundle.metadata[key]
            for key in ("date_start", "date_end", "reading_columns", "missing_calendar_days")
        },
        "generated_events": total,
        "events_by_table": {
            name: len(bundle.tables[name]) for name in ("outages", "meter_events", "subscriptions")
        },
        "events_within_observation_period": within,
        "events_within_observation_period_pct": 100 * within / total if total else None,
        "orphan_customer_references": sum(
            int((~frame.customer_id.isin(customers)).sum())
            for frame in bundle.tables.values()
            if "customer_id" in frame
        ),
        "invalid_region_references": int((~bundle.tables["outages"].region_id.isin(regions)).sum()),
        "invalid_event_records": sum(bool(errors) for errors in issues.values()),
        "events_overlapping_zero_or_missing": len(zero_missing),
        "events_without_observed_anomaly": len(observed - any_anomaly),
        "events_not_assessable": total - len(observed),
        "assessed_event_anomaly_overlap_rate": len(any_anomaly) / len(observed)
        if observed
        else None,
        "overlap_by_table": {
            table: {
                "events": len(bundle.tables[table]),
                **{
                    f"overlapping_{name}": sum(key[0] == table for key in keys)
                    for name, keys in overlaps.items()
                },
                "overlapping_zero_or_missing": sum(key[0] == table for key in zero_missing),
                "overlapping_any_anomaly": sum(key[0] == table for key in any_anomaly),
                "assessed_events": sum(key[0] == table for key in observed),
                "observed_anomaly_overlap_rate": (
                    sum(key[0] == table for key in any_anomaly)
                    / sum(key[0] == table for key in observed)
                )
                if any(key[0] == table for key in observed)
                else None,
                "without_observed_anomaly": sum(key[0] == table for key in observed - any_anomaly),
            }
            for table in ("outages", "meter_events", "subscriptions")
        },
        "overlap_definition": "At least one affected customer-day; missing includes absent calendar columns; "
        "drop <50% of preceding positive calendar day. Descriptive, not causal. No event tuning.",
    }


def save_context(input_path, *, output_dir, config=None, reference_metrics=None):
    output = empty_output(output_dir, protected=(input_path, reference_metrics))
    bundle = generate_context(input_path, config=config, reference_metrics=reference_metrics)
    report = compatibility_report(input_path, bundle)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in bundle.tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    write_json(output / "synthetic_context.json", bundle.metadata)
    write_json(output / "context_compatibility.json", report)
    return report
