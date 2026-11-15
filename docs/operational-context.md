# Synthetic operational context and decision gate

This phase demonstrates SGCC-compatible data integration and transparent human-review routing. **Every operational record is SYNTHETIC, including records marked `verified`.** None establishes an actual outage, meter problem, subscription, innocence or fraud. No precision, recall, accuracy or financial benefit is inferred from this context.

## Existing project audit

The supplied `data/consumption.csv` has 42,372 unique `CONS_NO` identifiers, each a 32-character string, and 1,034 daily columns spanning **2014-01-01 through 2016-10-31**, with one missing calendar date. `FLAG` is the historical customer-level label. The saved pipeline experiment uses a stratified 10,000-customer sample from this same source file, identified by SHA-256. Context generation covers the **entire source population**; the gate evaluates exactly the customers in the supplied prediction file.

The existing loader accepts `CONS_NO` or `customer_id`, parses SGCC date headers, validates IDs and labels, and treats negative/nonnumeric/nonfinite readings as missing. Model ranking uses raw probability descending, then customer ID; `risk_score` is rounded `100 * model_score`. Saved `test_predictions.csv` also contains labels and descriptive features. The gate uses only identifiers and `risk_score` from those predictions and preserves their original order. It never calls training, feature extraction, inference, ranking or model evaluation.

The additions follow existing pandas/NumPy, argparse, local CSV/JSON and pytest conventions. They do not change the loader, training, splits, models, historical reports or dashboard, and add no dependencies.

## CLI usage

Use new/empty output directories. These commands do not retrain a model:

```bash
gridguard generate-context \
  --input data/consumption.csv \
  --reference-metrics artifacts/pipeline-final-20261111/metrics.json \
  --output-dir artifacts/context-sgcc-phase1 \
  --seed 42 --regions 20

gridguard context-gate \
  --predictions artifacts/pipeline-final-20261111/test_predictions.csv \
  --context-dir artifacts/context-sgcc-phase1 \
  --evaluation-date 2016-11-07 \
  --window-start 2016-10-01 --window-end 2016-10-31 \
  --output-dir reports/context-sgcc-phase1
```

PowerShell users can put each command on one line. `--reference-metrics` is optional for generic fixtures but recommended for the actual experiment: a source fingerprint mismatch fails before output is written. `--input` is always explicit; this workflow never substitutes synthetic consumption for SGCC.

IDs are read as pandas strings, preserving leading zeros, case and NA-like identifiers. Empty/duplicate/whitespace-padded IDs are rejected rather than changed. Dates come exclusively from actual CSV headers using the existing date parser. Every customer-specific record references this source population; every outage region references the mapping. Source and population fingerprints are recorded locally.

**Do not commit generated files.** Synthetic events still contain actual SGCC identifiers. Standard output directories and customer-level output filenames are ignored by Git. Only code, invented test fixtures and documentation belong in the PR.

## Schemas

All four CSVs carry `provenance=SYNTHETIC`. IDs are strings. Timestamps use UTC. `available_at` is the first instant the **entire current row snapshot, including verification status**, is available. Empty verification means missing evidence.

| File | Columns beyond `available_at`, `provenance` | Meaning |
| --- | --- | --- |
| `customer_regions.csv` | `customer_id`, `region_id` | One deterministic synthetic region per source customer; no actual geography. |
| `outages.csv` | `outage_id`, `region_id`, `start_time`, `end_time`, `outage_type`, `verification_status` | Types `planned`, `unplanned`; half-open timestamp interval `[start_time, end_time)`. |
| `meter_events.csv` | `event_id`, `customer_id`, `event_type`, `start_date`, `end_date`, `verification_status` | Types `communication_failure`, `meter_fault`, `meter_replacement`; endpoint dates inclusive. |
| `subscriptions.csv` | `subscription_id`, `customer_id`, `start_date`, `end_date`, `subscription_status`, `verification_status` | States `active`, `terminated`; endpoint dates inclusive. |

`verification_status` is `verified`, `unverified` or empty. Subscription IDs make matched subscription evidence auditable. `synthetic_context.json` records schema/provenance, config, SHA-256, population fingerprint, lexical ID dtype, dates and availability semantics. `context_compatibility.json` holds aggregate validation and consumption overlap observations.

## Independent generation assumptions

Generation in `operational.py` reads **only IDs and date headers**, using a configurable NumPy seed and sorted IDs. It never reads label values, readings, predictions, anomaly rankings, split membership or holdout outcomes. Source hashing does not influence random draws. Row reordering does not change the generated tables.

| Assumption | Default |
| --- | --- |
| Regions | 20, capped at customer count; shuffled balanced assignments |
| Outage frequency | Poisson, 6/region/year |
| Planned share and duration | 40%; 2–8 hours |
| Unplanned duration | 1–24 hours |
| Communication failures | Poisson, 0.12/customer/year; 2–10 days |
| Meter faults | Poisson, 0.05/customer/year; 3–21 days |
| Replacements | Poisson, 0.07/customer/year; exactly one calendar day |
| Missing subscriptions | 4% of customers |
| Termination | 8% of customers with a subscription |
| Unverified / missing verification | 18% / 8% of rows, independently sampled |
| Availability | Completed outage/meter record plus 0–3 days reporting lag |

Year exposure is inclusive calendar span / 365.25. Event placement is uniform inside source observation and durations are bounded to that span. No orphan, out-of-period or contradictory records are deliberately generated. Small samples may contain no rare events. Missing verification/subscriptions, nonoverlapping events and events with no observed consumption anomaly demonstrate incomplete/irrelevant context without corrupting the source.

Subscription starts fall in the first 31 observation days. Most remain active through the last day. Some transition in the second half: active ends the day before terminated begins. A terminated interval describes inactivity after termination. Fixed subscription schedules, including end dates, are assumed known at interval start; terminated rows become available at termination. This is an artificial schedule assumption, not a reconstructed lifecycle. Early uncovered days remain unknown.

Use `--config generation.json` with `GenerationConfig` fields; explicit `--seed` and `--regions` override the file:

```json
{
  "seed": 42,
  "regions": 20,
  "outages_per_region_year": 6.0,
  "communication_failures_per_customer_year": 0.12,
  "meter_faults_per_customer_year": 0.05,
  "replacements_per_customer_year": 0.07,
  "missing_subscription_rate": 0.04,
  "terminated_rate": 0.08,
  "unverified_rate": 0.18,
  "missing_verification_rate": 0.08
}
```

These are predefined demonstration assumptions, not estimated SGCC operational rates. Do not tune them to improve anomaly overlap or apparent model performance.

## Post-generation consumption overlap

Only **after** generating all tables, `compatibility_report` streams source readings in customer chunks without loading label values. It observes affected customer-days and, for regional outages, **all mapped customers**. Nothing feeds back into generation.

Patterns are zero readings, missing/invalid readings (including absent calendar columns), and drops below 50% of the preceding positive calendar-day reading. Missing prior days do not establish drops. Negative/nonnumeric/nonfinite readings follow the existing loader's missing policy locally; source readings are unchanged.

An event overlaps if at least one affected customer-day matches. Outages ending at midnight exclude the next day. This day-level comparison is not proof that an intraday outage caused an anomaly. Large regions and long subscription periods have more opportunity for chance overlap; each table is reported separately.

The compatibility report includes customer/mapping counts, ID match rate, date coverage, event/table counts, percentage of intervals inside observation, orphan/invalid-region counts, zero/missing/drop overlap counts and rates, events with no observed anomaly, and unassessable events. Rates use assessed events as denominator. Counts are event-level, not customer-event pairs. Pattern counts can overlap; union counts deduplicate. Low overlap remains low; neither events nor readings are adjusted.

## Gate rules

`decision_gate.py` requires saved predictions, a context bundle, window endpoints and an evaluation cutoff. Date-only endpoints/cutoffs include the entire UTC day; explicit timestamps include that exact instant. The ordered window must be inside source observation and end no later than evaluation. Evaluation can occur after observation to accommodate reporting lag.

Availability filtering occurs **before** matching, duplicate/conflict checks and evidence summaries. A past event with future `available_at` is ignored. Missing/malformed availability is quarantined. Verification applies to the complete row snapshot at `available_at`; no later verification is backdated. A future integration with changing statuses needs auditable versioned snapshots.

Customer events join by exact ID, outages through an available region mapping. Matches require temporal intersection. Inclusive date periods become half-open intervals. Unknown types, invalid provenance, missing IDs/timestamps, reversed/out-of-period intervals, invalid verification/states, duplicate IDs and foreign-key failures are quarantined and counted. Linked invalid rows cause insufficient data for affected customers; unlinked invalid rows request verification. Missing tables mean unknown context; malformed schemas fail explicitly.

Rules apply in priority order:

| Status | Rule |
| --- | --- |
| `insufficient_data` | Invalid/missing score, missing/invalid/duplicate mapping, linked invalid context, conflicting subscription intervals or incomplete subscription window coverage. |
| `needs_context_verification` | Otherwise, overlapping outage/relevant meter event/terminated subscription; missing/unverified evidence; unlinked invalid context; or strict missing-event policy. Even synthetic `verified` disruptive events require confirmation. |
| `eligible_for_review` | Otherwise, model review can proceed with explicit context limitations. This does not establish operational clearance or fraud. |

Any overlapping subscription intervals, including two active intervals, are ambiguous. Adjacent active/terminated intervals are valid; termination requests verification. Coverage uses the union of matched periods; gaps are insufficient. High risk is illustrative `risk_score >= 80` and affects aggregate reporting only.

No matching outage/meter row means **unknown**, never proof of no outage/issue. Default `missing_event_policy="annotate"` permits model review with `no_matching_outages_unknown` / `no_matching_meter_events_unknown` and `synthetic_partial_unknown`. For conservative routing pass `--config gate.json`:

```json
{
  "missing_event_policy": "verify",
  "high_risk_threshold": 80,
  "relevant_meter_types": ["communication_failure", "meter_fault", "meter_replacement"]
}
```

Removing a meter type makes a verified matching record informational (`configured_irrelevant_meter_event`); missing/unverified evidence still requests verification. The aggregate report records all rule settings.

## Gate outputs

`context_decisions.csv` contains `customer_id`, unchanged `original_risk_score`, `decision_status`, sorted JSON arrays `reason_codes` / `matched_event_ids`, `evidence_status` and `provenance`. Rows retain input order, including ties. The source prediction file, including `model_score`, labels and features, is never overwritten.

Every decision includes `synthetic_context_only`. Other reasons include `overlapping_unplanned_outage`, `overlapping_communication_failure`, `overlapping_meter_replacement`, `overlapping_terminated`, `active_subscription_context`, `unverified_evidence`, `missing_evidence`, `subscription_coverage_unknown` and `conflicting_subscription_intervals`. Validation reasons identify failed checks. Evidence statuses are `synthetic_only`, `synthetic_unverified` and `synthetic_partial_unknown`; none is real operational confirmation.

`context_evaluation.json` reports customer/status totals, affected customers by event type, unique matched event verification counts, all available window-event counts, high-risk customers needing verification or more data, coverage/missing rates, invalid-record issue counts and ignored future records. Matched counts deduplicate regional events across customers. Available-window counts can include events outside the prediction cohort. Source/prediction SHA-256 values permit audit. No ML quality metric is calculated.

## Example aggregate output

With seed 42, 20 regions, the supplied source and the CLI window above:

| Compatibility measure | Result |
| --- | --- |
| SGCC customers / mapped customers | 42,372 / 42,372 |
| Customer ID match rate | 100% |
| Outage / meter / subscription records | 356 / 28,986 / 43,988 |
| Event intervals inside observation | 100% |
| Orphan customer / invalid region references | 0 / 0 |
| Events overlapping zero or missing consumption | 56,379 |
| Events with no defined observed consumption anomaly | 14,529 |

Meter events overlap zero, missing or drop patterns in 14,594/28,986 cases (50.35%); 14,392 have no defined anomaly. All 356 regional outages overlap a pattern somewhere in their region. That regional union statistic is expected to be large when thousands of customers share a region and does not demonstrate causal explanation. Long subscription intervals overlap patterns in 43,851/43,988 cases and are likewise not evidence of an explanation.

```json
{
  "provenance": "SYNTHETIC",
  "total_customers_evaluated": 2000,
  "customers_per_decision_status": {
    "eligible_for_review": 681,
    "needs_context_verification": 1242,
    "insufficient_data": 77
  },
  "high_risk_customers_requiring_verification": 4
}
```

These aggregate examples contain no customer identifiers. Full customer-level outputs remain local. Counts describe demonstration routing only.

## Integrity and limitations

As-of filtering governs operational context only. Existing SGCC predictions summarize historical customer readings/labels; they are **not prospective scores at the evaluation date**. An earlier context cutoff does not make a full-history model temporally valid. This workflow never changes or re-evaluates the historical holdout.

Tests use invented customers and cover deterministic generation/outcomes, exact IDs/foreign keys, date bounds, regions, overlap endpoints, future/delayed records, missing/invalid evidence, subscription conflicts, optional rules, input immutability and both CLI paths. CI runs the existing suite, Ruff and context smoke commands. Real SGCC outputs remain local.

Geography, frequencies, durations, subscriptions and verification are assumptions. There are no dated inspection outcomes, causal counterfactuals, operational probabilities or demonstrated performance improvements. A reviewer still needs independent operational evidence and a legitimate investigation process.
