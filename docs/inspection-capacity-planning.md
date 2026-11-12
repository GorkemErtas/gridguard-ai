# Inspection Capacity Planning

This read-only workspace compares inspection capacities using a selected experiment's saved customer rankings and retrospective holdout labels. Open **Inspection Capacity Planning** in the dashboard sidebar after selecting a final experiment. See the [dashboard launch guide](dashboard.md) for local setup. No estimator is loaded, fitted, tuned or scored again, and no artifact or historical metric is rewritten.

## Workflow and scope

1. Select an existing final experiment and check its source and provenance. Synthetic experiments remain prominently labeled. Legacy and missing-manifest warnings follow the existing dashboard validation rules.
2. Move the capacity slider from zero to the full cohort size. The displayed metrics and candidate CSV use this exact K.
3. Enter comma-separated comparison capacities, such as `0, 50, 100, 200`. The selected slider capacity is always included. Duplicate requests are removed, requests are sorted, and actual capacity is capped by cohort size. Oversized requests remain visible so unused capacity is apparent. An empty comparison field compares only the slider capacity.
4. Compare retrospective precision, recall, lift, observed positives, random expected positives and marginal yield. These are unfiltered top-K selections from the same fixed cohort; the separate Inspection queue's band/search filters do not apply.
5. Optionally enable the illustrative financial calculator and enter both assumptions in the same currency units. It starts disabled; inputs default to zero, not estimated business values.
6. Download candidates for the slider capacity. The existing export allowlist always excludes ground-truth labels, reading indicators and arbitrary extra fields. Customer IDs are excluded by default; opt in only for private local use. Spreadsheet formula prefixes in IDs are escaped. Ranks and scores can still link records when IDs are excluded.

Ordering uses the existing `ranking_order` contract: **raw `model_score` descending, then stripped customer ID lexicographically ascending**. Rounded `risk_score`, saved display ranks and labels never break ties. Unique, nonempty IDs and finite raw scores in `[0, 1]` are required. This score range does not imply calibration.

## Metric definitions

Let N be the selected cohort size, P its number of historical positive labels, k = min(requested K, N), and F(k) the observed positives among the top k customers.

| Quantity | Definition |
| --- | --- |
| Precision@K | F(k) / k |
| Recall@K | F(k) / P |
| Lift@K | Precision@K / (P / N) |
| Observed positives | F(k), a count of historical positive labels |
| Random expected positives | k × P / N |
| Random expected precision | P / N for nonzero k |
| Random expected recall | k / N when positives exist |
| Additional inspections | k minus the preceding scenario's actual capacity |
| Additional observed positives | F(k) minus the preceding scenario's observed positives |
| Marginal yield | Additional observed positives / additional inspections |

The random baseline means uniform selection **without replacement from the same cohort**. Its expected count is analytical, potentially fractional, and does not require a seed or Monte Carlo simulation. It is not a measured inspection outcome. The first scenario's marginal interval starts at zero inspections. Marginal intervals depend on the chosen scenario list; no cumulative precision value is substituted for marginal yield.

The charts sample at most 101 evenly spaced integer capacities to bound rendering work, with a vertical marker at the selected K. Lines connect sampled points; scenario rows and selected metrics use exact capacities. The capacity curve cache has an eight-entry bound and a five-minute TTL; its key includes the selected prediction frame, so a different cohort invalidates it.

### Boundary behavior

- **Zero actual capacity:** observed positives and random expected positives are zero when labels are available. Precision and lift are unavailable, because no customers were selected. Recall is zero if cohort positives exist.
- **Zero-positive cohort:** observed positives, random expected positives and nonempty-selection precision are zero. Recall and lift remain unavailable. A labeled empty cohort follows these conventions, with zero actual capacity and undefined rates.
- **Missing labels:** all outcome-based metrics, random baselines and financial recovery values are unavailable. Ranking and private CSV export still work. Stored summary metrics and raw scores cannot fill in missing labels.
- **Saturated capacity:** subsequent larger requests can have zero additional inspections and zero additional positives; marginal yield remains unavailable for that zero-width interval.
- **Validation-only run:** no customer scenarios or exports are offered. Invalid primary artifacts remain blocked by `load_run`, including checksum and partition mismatches under its existing validation rules.

## Illustrative financial arithmetic

**All financial results are illustrative scenarios, not validated financial forecasts.** For user-supplied inspection cost c and hypothetical recovery v per observed positive:

| Quantity | Definition |
| --- | --- |
| Total inspection cost | k × c |
| Hypothetical model recovery | F(k) × v |
| Hypothetical random recovery | (k × P / N) × v |
| Illustrative model net | Hypothetical model recovery minus total inspection cost |
| Illustrative random net | Hypothetical random recovery minus total inspection cost |

Both strategies incur the same cost at the same actual capacity. Costs use actual k, never an oversized requested K. Inputs must be finite and nonnegative; zero costs and zero recovery are valid, and net values may be negative. Nonfinite arithmetic results are rejected. Missing outcomes stay unavailable even if v is zero.

For example, with four customers, two positive labels, k=3 and two positives observed in the top three, the random expected count is 1.5. At c=20 and v=100, total cost is 60, hypothetical recoveries are 200 and 150, and illustrative net values are 140 and 90. This is arithmetic on assumptions, not evidence of collectable revenue.

The calculator assumes an identical hypothetical recovery per historical positive and constant inspection cost. It excludes collection rates, timing, other operating costs and uncertainty. Labels do not establish recoverable losses. Raw scores are never multiplied by recovery values or treated as calibrated probabilities.

## Reusable module

```python
from gridguard.capacity_planning import capacity_plan, financial_scenarios
from gridguard.dashboard import load_run, private_export

# Existing artifact validation and provenance checks happen here.
run = load_run("artifacts/pipeline-final-20261111")
plan = capacity_plan(run.predictions, capacities=[0, 50, 100, 200])
illustrative = financial_scenarios(
    plan, inspection_cost=20.0, recovery_per_positive=100.0
)
private_csv = private_export(run.predictions.head(50), include_ids=False)
```

`capacity_plan` and `financial_scenarios` are pure in-memory transforms and leave their inputs unchanged. `capacity_plan` reuses `modeling.ranking_metrics` and its deterministic validation/order contract. `financial_scenarios` consumes a frame produced by `capacity_plan` and includes an illustrative-use notice. File discovery, cache signatures, prediction checksum/partition validation, and export privacy stay in the existing dashboard helpers; the new module does not duplicate artifact validation.

## Evaluation integrity and verification

Capacity comparisons describe a previously inspected final holdout. Choose inspection capacity using staffing and operating constraints. Do not select or tune a model, score threshold or feature configuration using these holdout scenarios. Prospective performance requires separately collected inspection outcomes and an appropriate future evaluation design. A customer must never be classified as a confirmed electricity thief from model output alone; independent human verification is required.

Run the existing checks:

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m pytest -q
# Focused Streamlit interaction smoke tests (also included in the full suite):
.\.venv\Scripts\python.exe -m pytest -q tests/test_dashboard.py tests/test_app.py
```

Unit tests cover canonical metric equivalence, exact random/marginal calculations, deterministic ties under shuffled inputs, capacity limits, invalid inputs, empty/unlabeled/zero-positive/singleton cohorts, financial assumptions and overflow. Headless Streamlit `AppTest` checks navigate all six sections, change K and financial inputs, recover from invalid capacities, verify private exports, and assert that model loading/fitting and artifact writes do not occur during planning. Existing training tests use separate temporary synthetic fixtures.

Verification on 2026-11-12: Ruff and all 110 pytest tests passed. A separate read-only AppTest rendered all six sections against the existing final experiment, exercised zero/single/full capacity and financial controls, and blocked estimator loading/fitting. Checksums of all 65 existing local artifact/report files were unchanged. No real-data model was retrained, tuned or scored again.
