"""Estimator behaviour: planted effects, nulls, and degenerate input.

A single simulation cannot show that an estimator is unbiased. These tests check
that the bias over several planted values stays small next to the sampling error
that the same simulation reports, and that the null does not produce a positive
value. They are a guard against coding errors, not a proof.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from batchgate.stats import (
    bootstrap_mean,
    cluster_bootstrap_icc,
    design_effect,
    effective_n,
    icc_oneway,
    percentile_interval,
    permutation_null_icc,
    slot_effect,
    upper_p_value,
)

K = 16
N_BATCHES = 125


def beta_binomial(
    rho: float, mean: float, m: int, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Batches of Bernoulli draws whose true intra-batch correlation is rho.

    For a beta-binomial with shape a + b = 1/rho - 1, the ICC is exactly rho.
    """
    if rho <= 0:
        probabilities = np.full(m, mean)
    else:
        total = 1.0 / rho - 1.0
        probabilities = rng.beta(mean * total, (1 - mean) * total, size=m)
    values = (rng.random((m, k)) < probabilities[:, None]).astype(float)
    batches = np.repeat(np.arange(m), k)
    return values.reshape(-1), batches


@pytest.mark.parametrize("rho", [0.0, 0.02, 0.05, 0.15])
def test_icc_recovers_planted_correlation(rho: float) -> None:
    rng = np.random.default_rng(1234)
    estimates = np.array(
        [icc_oneway(*beta_binomial(rho, 0.15, N_BATCHES, K, rng)) for _ in range(200)]
    )
    bias = float(estimates.mean()) - rho
    standard_error = float(estimates.std(ddof=1)) / math.sqrt(estimates.size)
    assert abs(bias) < 0.01
    assert abs(bias) < 5 * standard_error + 0.002


def test_null_does_not_manufacture_correlation() -> None:
    rng = np.random.default_rng(99)
    estimates = np.array(
        [icc_oneway(*beta_binomial(0.0, 0.2, N_BATCHES, K, rng)) for _ in range(200)]
    )
    assert abs(float(estimates.mean())) < 0.005
    assert float((estimates < 0).mean()) > 0.3  # negative estimates are normal under the null


def test_fake_batch_permutation_collapses_a_planted_effect() -> None:
    rng = np.random.default_rng(7)
    values, batches = beta_binomial(0.15, 0.15, N_BATCHES, K, rng)
    observed = icc_oneway(values, batches)
    null = permutation_null_icc(values, batches, 200, rng)
    assert observed > 0.05
    assert abs(float(np.nanmean(null))) < 0.01
    assert upper_p_value(observed, null) < 0.01


def test_permutation_false_positive_rate_is_near_alpha() -> None:
    rng = np.random.default_rng(21)
    p_values = []
    for _ in range(40):
        values, batches = beta_binomial(0.0, 0.15, 40, K, rng)
        p_values.append(
            upper_p_value(
                icc_oneway(values, batches), permutation_null_icc(values, batches, 200, rng)
            )
        )
    assert float(np.mean(np.array(p_values) < 0.05)) < 0.20


def test_permutation_preserves_batch_sizes() -> None:
    rng = np.random.default_rng(3)
    values, batches = beta_binomial(0.1, 0.3, 10, K, rng)
    sizes_before = np.unique(batches, return_counts=True)[1]
    permutation_null_icc(values, batches, 5, rng)
    sizes_after = np.unique(batches, return_counts=True)[1]
    assert np.array_equal(sizes_before, sizes_after)


def test_cluster_bootstrap_covers_the_planted_value() -> None:
    rng = np.random.default_rng(5)
    covered = 0
    for _ in range(40):
        values, batches = beta_binomial(0.1, 0.2, N_BATCHES, K, rng)
        samples = cluster_bootstrap_icc(values, batches, 300, rng)
        interval = percentile_interval(samples, icc_oneway(values, batches), 0.05)
        covered += int(interval.low <= 0.1 <= interval.high)
    assert covered >= 32  # nominal 95% on 40 replicates


def test_unequal_batch_sizes_do_not_crash() -> None:
    values = np.array([0, 1, 1, 0, 1, 0, 0], dtype=float)
    batches = np.array(["a", "a", "a", "b", "b", "c", "c"])
    assert not math.isnan(icc_oneway(values, batches))


def test_degenerate_input_returns_nan_not_a_number() -> None:
    ones = np.ones(32)
    batches = np.repeat(np.arange(2), 16)
    assert math.isnan(icc_oneway(ones, batches))  # no variance anywhere
    assert math.isnan(icc_oneway(np.array([1.0, 0.0]), np.array(["a", "a"])))  # one batch
    assert math.isnan(icc_oneway(np.array([]), np.array([])))
    assert math.isnan(icc_oneway(np.array([1.0, 0.0]), np.array(["a", "b"])))  # every batch size 1


def test_negative_estimates_survive_and_do_not_inflate_deff() -> None:
    values = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    batches = np.repeat(np.arange(4), 2)
    rho = icc_oneway(values, batches)
    assert rho < 0
    assert design_effect(rho, 16) == 1.0
    assert effective_n(100, rho, 16) == 100.0


def test_design_effect_matches_the_published_numbers() -> None:
    assert design_effect(0.02, 16) == pytest.approx(1.30)
    assert design_effect(0.085, 16) == pytest.approx(2.275)
    assert effective_n(6400, 0.085, 16) == pytest.approx(2813.2, rel=1e-3)
    assert math.isnan(design_effect(math.nan, 16))


def test_bootstrap_mean_resamples_the_given_unit() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(0.05, 0.2, 500)
    interval = bootstrap_mean(values, 500, rng, 0.05)
    assert interval.low < interval.point < interval.high
    assert interval.low < 0.05 < interval.high


def test_slot_effect_finds_a_planted_position_effect() -> None:
    rng = np.random.default_rng(13)
    slots = np.tile(np.arange(K), 200)
    rates = 0.05 + 0.02 * slots
    values = (rng.random(slots.size) < rates).astype(float)
    result = slot_effect(values, slots, 200, rng)
    assert cast(float, result["p_any_slot_difference"]) < 0.05
    assert cast(float, result["p_monotone_trend"]) < 0.05

    flat = (rng.random(slots.size) < 0.2).astype(float)
    null_result = slot_effect(flat, slots, 200, rng)
    assert cast(float, null_result["p_any_slot_difference"]) > 0.05
