"""Statistics tests. Every estimator is checked against a brute force or closed form."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from step0 import assign, stats


def brute_force_covariance(errors: list[float], groups: list[int]) -> float:
    mean = sum(errors) / len(errors)
    total = 0.0
    pairs = 0
    for g in set(groups):
        members = [errors[i] for i in range(len(errors)) if groups[i] == g]
        for i in range(len(members)):
            for j in range(len(members)):
                if i != j:
                    total += (members[i] - mean) * (members[j] - mean)
            pairs += len(members) - 1
    return total / pairs


def test_pair_weighted_covariance_matches_brute_force() -> None:
    rng = np.random.default_rng(1)
    errors = (rng.random(60) < 0.3).astype(float)
    groups = np.repeat(np.arange(12), 5)
    assert stats.pair_weighted_covariance(errors, groups) == pytest.approx(
        brute_force_covariance(errors.tolist(), groups.tolist())
    )


def test_pair_weighted_covariance_handles_unequal_groups() -> None:
    errors = np.array([1.0, 0, 1, 0, 0, 1, 1])
    groups = np.array([0, 0, 0, 1, 1, 2, 2])
    assert stats.pair_weighted_covariance(errors, groups) == pytest.approx(
        brute_force_covariance(errors.tolist(), groups.tolist())
    )


def test_perfectly_correlated_calls_give_a_large_positive_covariance() -> None:
    errors = np.repeat([1.0, 0.0] * 5, 4)
    groups = np.repeat(np.arange(10), 4)
    assert stats.pair_weighted_covariance(errors, groups) == pytest.approx(0.25)


def test_independent_errors_give_about_zero() -> None:
    rng = np.random.default_rng(7)
    errors = (rng.random(16 * 400) < 0.2).astype(float)
    groups = np.repeat(np.arange(400), 16)
    assert abs(stats.pair_weighted_covariance(errors, groups)) < 0.005


def test_nan_rows_are_dropped_not_imputed() -> None:
    errors = np.array([1.0, np.nan, 1.0, 0.0])
    groups = np.array([0, 0, 1, 1])
    assert not math.isnan(stats.pair_weighted_covariance(errors, groups))
    assert math.isnan(stats.pair_weighted_covariance(np.array([np.nan, np.nan]), np.array([0, 0])))


def test_fast_path_agrees_with_the_general_path() -> None:
    rng = np.random.default_rng(3)
    matrix = (rng.random((30, 16)) < 0.25).astype(float)
    groups = np.repeat(np.arange(30), 16)
    assert stats.covariance_from_stratified(matrix) == pytest.approx(
        stats.pair_weighted_covariance(matrix.ravel(), groups)
    )


def test_delta_exact_is_signed_and_zero_for_identical_arms() -> None:
    rng = np.random.default_rng(5)
    errors = (rng.random(160) < 0.3).astype(float)
    groups = np.repeat(np.arange(10), 16)
    assert stats.delta_exact(errors, errors, groups) == pytest.approx(0.0)
    clustered = np.repeat((rng.random(10) < 0.5).astype(float), 16)
    assert stats.delta_exact(clustered, errors, groups) > 0
    assert stats.delta_exact(errors, clustered, groups) < 0


def test_cell_delta_averages_assignments_without_pooling() -> None:
    groups = [np.repeat(np.arange(10), 16)] * 3
    joint = [
        np.repeat((np.random.default_rng(i).random(10) < 0.4).astype(float), 16)
        for i in range(3)
    ]
    exact = [(np.random.default_rng(100 + i).random(160) < 0.4).astype(float) for i in range(3)]
    result = stats.cell_delta(joint, exact, groups)
    assert len(result["per_assignment"]) == 3
    assert result["point"] == pytest.approx(float(np.mean(result["per_assignment"])))


def test_fake_batch_contrast_is_near_zero_for_independent_labels() -> None:
    rng = np.random.default_rng(11)
    independent = (rng.random(16 * 200) < 0.2).astype(float)
    real = np.repeat(np.arange(200), 16)
    fake = rng.permutation(real)
    result = stats.fake_batch_contrast(independent, independent, real, fake)
    assert abs(result["delta_perm"]) < 0.01


def _frame(n_items: int = 384, n_strata: int = 16) -> assign.Frame:
    per = n_items // n_strata
    return assign.Frame(
        doc_ids=tuple(f"d{i:05d}" for i in range(n_items)),
        strata=tuple(i // per for i in range(n_items)),
    )


def _replay_for(
    frame: assign.Frame, cell_id: str, n_assignments: int
) -> Callable[[int], list[np.ndarray]]:
    def replay(draw: int) -> list[np.ndarray]:
        return assign.replay_stratum_matrices(frame, cell_id, n_assignments, draw)

    return replay


def test_randomization_test_returns_a_valid_p_value_under_the_null() -> None:
    rng = np.random.default_rng(2)
    frame = _frame()
    matrices = assign.observed_stratum_matrices(frame, "cell", 3)
    joint = [(rng.random(frame.n_items) < 0.25).astype(float) for _ in range(3)]
    exact = [(rng.random(frame.n_items) < 0.25).astype(float) for _ in range(3)]
    result = stats.randomization_test(
        joint, exact, matrices, 199, _replay_for(frame, "cell", 3)
    )
    assert result["min_reachable_p"] == pytest.approx(1 / 200)
    assert result["min_reachable_p"] <= result["p_value_one_sided_upper"] <= 1.0
    assert result["p_value_one_sided_upper"] > 0.01
    interval = result["effect_interval"]
    assert interval["low"] <= result["observed"] <= interval["high"]


def test_randomization_test_detects_strong_within_call_dependence() -> None:
    rng = np.random.default_rng(4)
    frame = _frame()
    matrices = assign.observed_stratum_matrices(frame, "cell", 3)
    joint: list[np.ndarray] = []
    exact: list[np.ndarray] = []
    for matrix in matrices:
        n_strata, n_calls = matrix.shape
        call_rate = np.asarray(rng.random(n_calls)) * 0.6
        cube = (rng.random((n_calls, n_strata)) < call_rate[:, None]).astype(float)
        flat = np.empty(frame.n_items)
        flat[matrix.T.ravel()] = cube.ravel()
        joint.append(flat)
        exact.append((rng.random(frame.n_items) < 0.3).astype(float))
    result = stats.randomization_test(
        joint, exact, matrices, 199, _replay_for(frame, "cell", 3)
    )
    assert result["observed"] > 0.01
    assert result["p_value_one_sided_upper"] == pytest.approx(1 / 200)


def test_randomization_test_replays_the_frozen_generator_reproducibly() -> None:
    rng = np.random.default_rng(9)
    frame = _frame()
    matrices = assign.observed_stratum_matrices(frame, "cell", 2)
    joint = [(rng.random(frame.n_items) < 0.3).astype(float) for _ in range(2)]
    exact = [(rng.random(frame.n_items) < 0.3).astype(float) for _ in range(2)]
    a = stats.randomization_test(joint, exact, matrices, 99, _replay_for(frame, "cell", 2))
    b = stats.randomization_test(joint, exact, matrices, 99, _replay_for(frame, "cell", 2))
    c = stats.randomization_test(joint, exact, matrices, 99, _replay_for(frame, "other", 2))
    assert a["p_value_one_sided_upper"] == b["p_value_one_sided_upper"]
    assert a["null_sd"] == b["null_sd"]
    assert a["null_sd"] != c["null_sd"]
    assert "execution block" in a["replay_mechanism"]


def test_the_statistic_ignores_the_slot_order_inside_a_call() -> None:
    """Slot balance cannot change the value, so replaying partitions is enough."""
    rng = np.random.default_rng(21)
    frame = _frame()
    matrix = assign.observed_stratum_matrices(frame, "cell", 1)[0]
    values = (rng.random(frame.n_items) < 0.3).astype(float)
    shuffled = matrix.copy()
    for call in range(shuffled.shape[1]):
        shuffled[:, call] = rng.permutation(shuffled[:, call])
    assert stats.covariance_from_stratified(
        stats._reshape(values, matrix)
    ) == pytest.approx(stats.covariance_from_stratified(stats._reshape(values, shuffled)))


def test_the_randomization_test_refuses_unresolved_labels() -> None:
    frame = _frame()
    matrices = assign.observed_stratum_matrices(frame, "cell", 1)
    values = np.zeros(frame.n_items)
    values[7] = np.nan
    with pytest.raises(ValueError, match="resolved value"):
        stats.randomization_test(
            [values], [np.zeros(frame.n_items)], matrices, 9, _replay_for(frame, "cell", 1)
        )


def test_recurrence_statistic_uses_equal_cell_weights() -> None:
    result = stats.recurrence_statistic({"a": 0.004, "b": 0.0, "c": 0.002})
    assert result["equal_weight_mean"] == pytest.approx(0.002)
    assert result["n_cells"] == 3


def test_holm_matches_a_worked_example() -> None:
    result = stats.holm({"a": 0.01, "b": 0.02, "c": 0.04})
    assert result["adjusted_p"]["a"] == pytest.approx(0.03)
    assert result["adjusted_p"]["b"] == pytest.approx(0.04)
    assert result["adjusted_p"]["c"] == pytest.approx(0.04)
    assert result["rejected"] == {"a": True, "b": True, "c": True}


def test_holm_is_monotone_and_never_exceeds_one() -> None:
    result = stats.holm({"a": 0.5, "b": 0.6, "c": 0.9})
    values = [result["adjusted_p"][k] for k in result["order"]]
    assert values == sorted(values)
    assert max(values) <= 1.0


def test_malformed_sensitivity_brackets_the_dropped_estimate() -> None:
    rng = np.random.default_rng(6)
    groups = np.repeat(np.arange(20), 16)
    joint = np.repeat((rng.random(20) < 0.5).astype(float), 16)
    exact = (rng.random(320) < 0.3).astype(float)
    j_bad = np.zeros(320, dtype=bool)
    e_bad = np.zeros(320, dtype=bool)
    j_bad[[3, 40, 41]] = True
    e_bad[[100]] = True
    joint_nan = np.where(j_bad, np.nan, joint)
    exact_nan = np.where(e_bad, np.nan, exact)
    out = stats.malformed_sensitivity(joint_nan, exact_nan, groups, j_bad, e_bad)
    assert set(out) >= {
        "delta_unresolved_as_correct",
        "delta_unresolved_as_error",
        "delta_unresolved_dropped",
        "delta_whole_call_discarded",
    }
    assert out["n_calls_discarded"] == 3  # calls 0 and 2 from joint, call 6 from exact


def test_failure_clustering_detects_call_level_failures() -> None:
    groups = np.repeat(np.arange(10), 16)
    clustered = np.repeat([True, False] * 5, 16)
    assert stats.failure_clustering(clustered, groups) == pytest.approx(0.25)


def test_wald_interval_matches_the_closed_form() -> None:
    errors = np.concatenate([np.ones(64), np.zeros(448)])
    interval = stats.wald_interval(errors)
    p = 64 / 512
    half = stats.Z975 * math.sqrt(p * (1 - p) / 512)
    assert interval.point == pytest.approx(p)
    assert interval.low == pytest.approx(p - half)
    assert interval.high == pytest.approx(p + half)


def test_call_t_interval_uses_31_degrees_of_freedom() -> None:
    rng = np.random.default_rng(8)
    call_means = rng.random(32) * 0.2
    interval = stats.call_t_interval(call_means)
    se = call_means.std(ddof=1) / math.sqrt(32)
    assert interval.high - interval.point == pytest.approx(2.0395134463964077 * se, rel=1e-6)


def test_call_aware_interval_is_wider_when_errors_cluster_by_call() -> None:
    clustered = np.repeat([1.0, 0.0] * 16, 16)
    call_means = clustered.reshape(32, 16).mean(axis=1)
    wald = stats.wald_interval(clustered)
    t_interval = stats.call_t_interval(call_means)
    assert (t_interval.high - t_interval.low) > (wald.high - wald.low)


def test_fpc_is_the_documented_factor() -> None:
    assert stats.fpc(512, 2048) == pytest.approx(math.sqrt(0.75))
    assert stats.fpc(2048, 2048) == pytest.approx(0.0)


def test_whole_call_bootstrap_brackets_the_mean() -> None:
    rng = np.random.default_rng(12)
    call_means = rng.random(32) * 0.3
    interval = stats.whole_call_bootstrap(call_means, 500, rng)
    assert interval.low < interval.point < interval.high


def test_variance_ratio_recovers_a_known_inflation() -> None:
    rng = np.random.default_rng(13)
    nominal = np.full(500, 0.0002)
    means = 0.2 + rng.normal(0, math.sqrt(0.0002 * 4.0), 500)
    result = stats.variance_ratio(means, nominal, n_boot=400, rng=rng)
    assert 3.0 < result["point"] < 5.0
    assert result["low"] < result["point"] < result["high"]
    assert result["floor"] == 1.25


def test_clopper_pearson_matches_known_bounds() -> None:
    interval = stats.clopper_pearson(0, 20)
    assert interval.low == 0.0
    assert interval.high == pytest.approx(0.1684, abs=1e-3)
    full = stats.clopper_pearson(20, 20)
    assert full.high == 1.0
    assert full.low == pytest.approx(0.8316, abs=1e-3)


def test_coverage_counts_hits_exactly() -> None:
    lows = np.array([0.0, 0.3, 0.1])
    highs = np.array([0.2, 0.5, 0.4])
    result = stats.coverage(lows, highs, 0.15)
    assert result["n_covered"] == 2
    assert result["point"] == pytest.approx(2 / 3)


def test_undercoverage_needs_the_least_favourable_reference_variant() -> None:
    rng = np.random.default_rng(14)
    centres = 0.2 + rng.normal(0, 0.02, 800)
    lows, highs = centres - 0.005, centres + 0.005
    strict = stats.coverage_with_reference_uncertainty(lows, highs, 0.2, 0.0)
    assert strict["undercoverage_supported"]
    generous = stats.coverage_with_reference_uncertainty(lows, highs, 0.2, 0.05)
    assert generous["least_favourable_upper_bound"] >= strict["least_favourable_upper_bound"]


def test_reference_precision_check_applies_the_frozen_threshold() -> None:
    rng = np.random.default_rng(15)
    means = 0.2 + rng.normal(0, 0.02, 250)
    tight = stats.reference_precision_check(means, 0.03)
    assert tight["precise_enough_for_a_coverage_claim"]
    loose = stats.reference_precision_check(means, 0.001)
    assert not loose["precise_enough_for_a_coverage_claim"]
    assert loose["threshold"] == 0.25
