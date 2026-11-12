import numpy as np
import pandas as pd
import pytest

from gridguard.capacity_planning import (
    FINANCIAL_NOTICE,
    PLAN_COLUMNS,
    capacity_plan,
    financial_scenarios,
)
from gridguard.modeling import ranking_metrics


@pytest.fixture
def cohort():
    return pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d"],
            "model_score": [0.9, 0.8, 0.5, 0.1],
            "true_label": [1, 0, 1, 0],
        }
    )


def test_exact_metrics_random_baseline_and_marginal_intervals(cohort):
    before = cohort.copy(deep=True)
    plan = capacity_plan(cohort, [99, 3, 1, 0, 4, 3])
    assert plan.requested_k.tolist() == [0, 1, 3, 4, 99]
    assert plan.evaluated_k.tolist() == [0, 1, 3, 4, 4]
    assert plan.positives_found.tolist() == [0, 1, 2, 2, 2]
    assert plan.random_expected_positives.tolist() == [0, 0.5, 1.5, 2, 2]
    assert plan.marginal_inspections.tolist() == [0, 1, 2, 1, 0]
    assert plan.marginal_positives.tolist() == [0, 1, 1, 0, 0]
    assert plan.marginal_yield.iloc[1:4].tolist() == [1, 0.5, 0]
    assert pd.isna(plan.marginal_yield.iloc[0])
    assert pd.isna(plan.marginal_yield.iloc[-1])
    assert pd.isna(plan.precision.iloc[0])
    assert pd.isna(plan.lift.iloc[0])
    assert plan.recall.iloc[0] == 0
    for row in plan.iloc[1:].to_dict("records"):
        canonical = ranking_metrics(
            cohort.true_label,
            cohort.model_score,
            row["requested_k"],
            customer_ids=cohort.customer_id,
        )
        for key, value in canonical.items():
            assert row[key] == pytest.approx(value)
        assert row["random_precision"] == 0.5
        assert row["random_recall"] == row["evaluated_k"] / 4
    assert plan.precision.iloc[2] == pytest.approx(2 / 3)
    assert plan.recall.iloc[2] == 1
    assert plan.lift.iloc[2] == pytest.approx(4 / 3)
    pd.testing.assert_frame_equal(cohort, before)


def test_first_marginal_interval_starts_at_zero(cohort):
    row = capacity_plan(cohort, [3]).iloc[0]
    assert row.marginal_inspections == 3
    assert row.marginal_positives == 2
    assert row.marginal_yield == pytest.approx(2 / 3)


def test_ties_raw_precision_and_permutation_invariance():
    cohort = pd.DataFrame(
        {
            "customer_id": ["z", " a ", "m", "00"],
            "model_score": [0.80001, 0.80001, 0.79999, 0.79998],
            "risk_score": [80, 80, 80, 100],  # Display scores cannot reorder the queue.
            "rank": [4, 3, 2, 1],  # Nor can stale display ranks.
            "true_label": [0, 1, 1, 0],
        },
        index=[40, 30, 20, 10],
    )
    expected = capacity_plan(cohort, [1, 2, 3, 4])
    assert expected.positives_found.tolist() == [1, 1, 2, 2]
    for seed in range(5):
        actual = capacity_plan(cohort.sample(frac=1, random_state=seed), [4, 2, 1, 3])
        pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("capacity", [-1, 1.5, "3", None, True, False, np.bool_(True), np.nan])
def test_reject_invalid_capacity(cohort, capacity):
    with pytest.raises(ValueError, match="nonnegative integers"):
        capacity_plan(cohort, [capacity])


def test_numpy_integers_generator_and_capacity_limits(cohort):
    plan = capacity_plan(cohort, (k for k in [np.int64(0), np.int32(1), 10**20]))
    assert plan.requested_k.tolist() == [0, 1, 10**20]
    assert plan.evaluated_k.tolist() == [0, 1, 4]
    assert plan.iloc[-1].recall == 1
    assert capacity_plan(cohort).evaluated_k.tolist() == [4, 4, 4]


@pytest.mark.parametrize("labeled", [True, False])
def test_empty_cohort_and_empty_capacities(cohort, labeled):
    if not labeled:
        cohort = cohort.drop(columns="true_label")
    plan = capacity_plan(cohort.iloc[:0], [0, 1, 99])
    assert plan.evaluated_k.tolist() == [0, 0, 0]
    assert plan.marginal_inspections.eq(0).all()
    assert plan[["precision", "recall", "lift", "marginal_yield"]].isna().all().all()
    if labeled:
        assert plan.positives_found.eq(0).all()
        assert plan.random_expected_positives.eq(0).all()
        assert plan.marginal_positives.eq(0).all()
    else:
        assert plan[["positives_found", "random_expected_positives"]].isna().all().all()
    empty = capacity_plan(cohort, [])
    assert empty.empty
    assert empty.columns.tolist() == PLAN_COLUMNS


def test_unlabeled_cohort_does_not_use_scores_as_probabilities(cohort):
    plan = capacity_plan(cohort.drop(columns="true_label"), [0, 1, 3])
    assert plan.evaluated_k.tolist() == [0, 1, 3]
    assert plan.marginal_inspections.tolist() == [0, 1, 2]
    outcomes = set(PLAN_COLUMNS) - {"requested_k", "evaluated_k", "marginal_inspections"}
    assert plan[sorted(outcomes)].isna().all().all()


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_and_single_customer(cohort, label):
    cohort["true_label"] = label
    plan = capacity_plan(cohort, [0, 1, 4, 5])
    assert plan.positives_found.tolist() == [0, label, 4 * label, 4 * label]
    assert plan.precision.iloc[1:].eq(label).all()
    assert plan.random_expected_positives.tolist() == [0, label, 4 * label, 4 * label]
    if label:
        assert plan.recall.tolist() == [0, 0.25, 1, 1]
        assert plan.lift.iloc[1:].eq(1).all()
    else:
        assert plan[["recall", "lift", "random_recall"]].isna().all().all()
    single = capacity_plan(cohort.head(1), [0, 1, 2])
    assert single.evaluated_k.tolist() == [0, 1, 1]
    assert single.positives_found.tolist() == [0, label, label]


@pytest.mark.parametrize("score", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_reject_invalid_scores_even_without_labels(cohort, score):
    cohort.loc[0, "model_score"] = score
    with pytest.raises(ValueError, match="probabilities"):
        capacity_plan(cohort.drop(columns="true_label"), [0])


@pytest.mark.parametrize("label", [None, np.nan, 2, -1, 0.5, "1"])
def test_reject_invalid_labels_even_with_no_capacity_rows(cohort, label):
    cohort["true_label"] = cohort.true_label.astype(object)
    cohort.loc[0, "true_label"] = label
    with pytest.raises(ValueError, match="binary"):
        capacity_plan(cohort, [])


@pytest.mark.parametrize("customer_id", [None, "", "  ", "b", " b "])
def test_reject_invalid_or_normalized_duplicate_ids(cohort, customer_id):
    cohort.loc[0, "customer_id"] = customer_id
    with pytest.raises(ValueError, match="customer IDs"):
        capacity_plan(cohort, [0])


@pytest.mark.parametrize("column", ["customer_id", "model_score"])
def test_required_columns(cohort, column):
    with pytest.raises(ValueError, match="Missing prediction columns"):
        capacity_plan(cohort.drop(columns=column), [1])


def test_financial_calculation_uses_actual_capacity_and_equal_baseline_cost(cohort):
    plan = capacity_plan(cohort, [0, 1, 3, 99])
    before = plan.copy(deep=True)
    result = financial_scenarios(plan, inspection_cost=20, recovery_per_positive=100)
    assert result.inspection_cost_total.tolist() == [0, 20, 60, 80]
    assert result.hypothetical_model_recovery.tolist() == [0, 100, 200, 200]
    assert result.hypothetical_random_recovery.tolist() == [0, 50, 150, 200]
    assert result.illustrative_model_net.tolist() == [0, 80, 140, 120]
    assert result.illustrative_random_net.tolist() == [0, 30, 90, 120]
    assert result.notice.eq(FINANCIAL_NOTICE).all()
    assert "not validated financial forecasts" in FINANCIAL_NOTICE
    pd.testing.assert_frame_equal(plan, before)


@pytest.mark.parametrize("cost,recovery", [(0, 100), (20, 0), (0, 0), (2.5, 10.25)])
def test_zero_and_fractional_financial_inputs(cohort, cost, recovery):
    result = financial_scenarios(
        capacity_plan(cohort, [3]), inspection_cost=cost, recovery_per_positive=recovery
    ).iloc[0]
    assert result.inspection_cost_total == 3 * cost
    assert result.hypothetical_model_recovery == 2 * recovery
    assert result.illustrative_model_net == pytest.approx(2 * recovery - 3 * cost)
    assert result.illustrative_random_net == pytest.approx(1.5 * recovery - 3 * cost)


@pytest.mark.parametrize("parameter", ["inspection_cost", "recovery_per_positive"])
@pytest.mark.parametrize("value", [-1, np.nan, np.inf, -np.inf, True, "20", None])
def test_reject_invalid_financial_inputs(cohort, parameter, value):
    inputs = {"inspection_cost": 10, "recovery_per_positive": 100, parameter: value}
    with pytest.raises(ValueError, match="finite nonnegative"):
        financial_scenarios(capacity_plan(cohort, [1]), **inputs)


def test_finances_for_zero_positive_empty_and_unlabeled_cohorts(cohort):
    cohort["true_label"] = 0
    zero = financial_scenarios(
        capacity_plan(cohort, [2]), inspection_cost=5, recovery_per_positive=100
    ).iloc[0]
    assert zero.illustrative_model_net == zero.illustrative_random_net == -10
    for empty in (cohort.iloc[:0], cohort.drop(columns="true_label")):
        result = financial_scenarios(
            capacity_plan(empty, [0, 2]), inspection_cost=5, recovery_per_positive=100
        )
        assert result.inspection_cost_total.tolist() == ([0, 0] if empty.empty else [0, 10])
        if empty.empty:
            assert result.illustrative_model_net.eq(0).all()
        else:
            assert result.hypothetical_model_recovery.isna().all()
            assert result.illustrative_model_net.isna().all()
            assert result.illustrative_random_net.isna().all()
    assert financial_scenarios(
        capacity_plan(cohort, []), inspection_cost=0, recovery_per_positive=0
    ).empty


def test_financial_overflow_and_missing_plan_columns(cohort):
    plan = capacity_plan(cohort, [4])
    with pytest.raises(ValueError, match="finite numeric range"):
        financial_scenarios(plan, inspection_cost=1e308, recovery_per_positive=100)
    with pytest.raises(ValueError, match="finite numeric range"):
        financial_scenarios(plan, inspection_cost=100, recovery_per_positive=1e308)
    with pytest.raises(ValueError, match="capacity_plan result"):
        financial_scenarios(pd.DataFrame(), inspection_cost=1, recovery_per_positive=1)
