"""The concentration direction must never flip. These tests pin it."""

from __future__ import annotations

import numpy as np

from gatek.metrics import (
    CONCENTRATED,
    DECONCENTRATED,
    coefficient_of_variation,
    concentration_label,
    cosine_distances,
    distance_stats,
    euclidean_distances,
    nearest_farthest_ratio,
    relative_contrast,
    spearman_rho,
)

CONCENTRATED_SET = np.array([1.00, 1.01, 1.02, 0.99])
SPREAD_SET = np.array([0.20, 1.00, 3.00, 9.00])


def test_relative_contrast_is_lower_when_distances_are_alike() -> None:
    assert relative_contrast(CONCENTRATED_SET) < relative_contrast(SPREAD_SET)


def test_concentration_label_direction() -> None:
    rc_conc = relative_contrast(CONCENTRATED_SET)
    rc_spread = relative_contrast(SPREAD_SET)
    assert concentration_label(rc_conc, rc_spread) == CONCENTRATED
    assert concentration_label(rc_spread, rc_conc) == DECONCENTRATED


def test_label_is_reflexive_at_equal_contrast() -> None:
    assert concentration_label(2.0, 2.0) == DECONCENTRATED


def test_nearest_farthest_ratio_is_higher_when_concentrated() -> None:
    assert nearest_farthest_ratio(CONCENTRATED_SET) > nearest_farthest_ratio(SPREAD_SET)


def test_coefficient_of_variation_is_lower_when_concentrated() -> None:
    assert coefficient_of_variation(CONCENTRATED_SET) < coefficient_of_variation(SPREAD_SET)


def test_distance_stats_round_trip() -> None:
    stats = distance_stats(SPREAD_SET)
    assert stats.n == 4
    assert stats.d_min == 0.2
    assert stats.d_max == 9.0
    assert abs(stats.relative_contrast - (9.0 - 0.2) / 0.2) < 1e-6


def test_euclidean_and_cosine_agree_on_the_nearest_row_for_scaled_copies() -> None:
    rows = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    query = np.array([2.0, 0.1])
    assert int(np.argmin(euclidean_distances(query, rows))) == 0
    assert int(np.argmin(cosine_distances(query, rows))) == 0


def test_spearman_returns_none_on_constant_input() -> None:
    assert spearman_rho(np.ones(5), np.arange(5.0)) is None
    assert spearman_rho(np.arange(5.0), np.arange(5.0)) == 1.0
