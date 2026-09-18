"""CPU checks for the frozen B5 estimands, thresholds, outcome readings and power study."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from freezeb import b5, b5_analysis, b5_power
from step0 import stats

CELL = b5.CELLS[0]
AMPLITUDE = 10.0
BASE_DEPENDENCE = 0.02


def blank_quality() -> dict[str, int]:
    return {"calls": 0, "resolved_calls": 0, "terminal_failures": 0, "parse_failures": 0}


def synthetic_cell(
    cell_key: str, curve: dict[int, float], seed: int, marginal: float = 0.2
) -> b5_analysis.CellData:
    """One cell whose joint arm carries a known excess correlation at every size."""
    rates = b5_power.item_rates(marginal, b5.N_ITEMS, seed)
    scale = b5_power.attenuation(rates, seed + 1)
    cell = b5_analysis.CellData(cell_key, [f"d{index:05d}" for index in range(b5.N_ITEMS)], {})
    for k in b5.SIZES:
        n_groups = b5.N_ITEMS // k
        for assignment in range(1, b5.N_ASSIGNMENTS + 1):
            ordered = rates[cell.order[assignment]]
            rng = np.random.default_rng(seed * 100_003 + 1_000 * k + assignment)
            cell.values[(k, assignment, "exact_context")] = b5_power.draw_errors(
                ordered, n_groups, k, BASE_DEPENDENCE / scale, rng
            )
            cell.values[(k, assignment, "joint")] = b5_power.draw_errors(
                ordered, n_groups, k, max((BASE_DEPENDENCE + curve[k]) / scale, 0.0), rng
            )
            for condition in b5.CONDITIONS:
                cell.quality[(k, assignment, condition)] = blank_quality()
    return cell


def family(shape: str) -> list[b5_analysis.CellData]:
    curve = {k: AMPLITUDE * v for k, v in b5_power.scenario_contrasts()[shape].items()}
    return [synthetic_cell(key, curve, index + 1) for index, key in enumerate(b5.CELLS)]


def contrast_by_size(tables: list[dict[str, Any]]) -> list[float]:
    return [
        float(np.mean([table["by_size"][position]["standardized_contrast"] for table in tables]))
        for position in range(len(b5.SIZES))
    ]


def cross_slope(tables: list[dict[str, Any]]) -> float:
    return float(np.mean([float(table["slope_standardized"]) for table in tables]))


# --- the estimator on known cases -------------------------------------------


def test_covariance_matches_the_sealed_pair_weighted_estimator() -> None:
    rng = np.random.default_rng(11)
    values = rng.integers(0, 2, 256).astype(float)
    values[3] = math.nan
    values[200] = math.nan
    groups = np.repeat(np.arange(32), 8)
    assert b5_analysis.covariance(values, 32, 8) == pytest.approx(
        stats.pair_weighted_covariance(values, groups)
    )


def test_perfect_within_call_agreement_standardizes_to_one() -> None:
    k = 8
    n_groups = 64
    values = np.repeat(np.arange(n_groups) % 2, k).astype(float)
    row = b5_analysis.condition_summary(values, n_groups, k)
    assert row["marginal_error"] == pytest.approx(0.5)
    assert row["covariance"] == pytest.approx(0.25)
    assert row["standardized"] == pytest.approx(1.0)
    assert row["variance_inflation"] == pytest.approx(1.0 + (k - 1) * 1.0)


def test_balanced_within_call_split_gives_the_exact_negative_floor() -> None:
    k = 4
    n_groups = 64
    values = np.tile(np.array([0.0, 1.0, 0.0, 1.0]), n_groups)
    row = b5_analysis.condition_summary(values, n_groups, k)
    assert row["marginal_error"] == pytest.approx(0.5)
    assert row["standardized"] == pytest.approx(-1.0 / (k - 1))


def test_primary_contrast_is_joint_minus_exact_on_a_known_pair() -> None:
    k = 4
    n_groups = 64
    joint = np.repeat(np.arange(n_groups) % 2, k).astype(float)
    exact = np.tile(np.array([0.0, 1.0, 0.0, 1.0]), n_groups)
    standardized, raw = b5_analysis._contrast(joint, exact, n_groups, k)
    assert standardized == pytest.approx(1.0 + 1.0 / (k - 1))
    assert raw == pytest.approx(0.25 - (-0.25 / (k - 1)))


def test_unresolved_labels_shrink_the_call_and_never_become_zero() -> None:
    k = 8
    n_groups = 64
    values = np.repeat(np.arange(n_groups) % 2, k).astype(float)
    dropped = values.copy()
    dropped[:4] = math.nan
    assert b5_analysis.covariance(dropped, n_groups, k) == pytest.approx(
        b5_analysis.covariance(values, n_groups, k), abs=5e-3
    )
    assert not math.isnan(b5_analysis.covariance(dropped, n_groups, k))


def test_slope_is_ordinary_least_squares_on_log2_size() -> None:
    assert b5_analysis._slope([2, 4, 8, 16, 32], [1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(1.0)
    assert b5_analysis._slope([2, 4, 8, 16, 32], [1.0] * 5) == pytest.approx(0.0)
    assert math.isnan(b5_analysis._slope([2, 4, 8, 16, 32], [1.0, math.nan, 3.0, 4.0, 5.0]))


# --- the five curves ---------------------------------------------------------


def test_increasing_curve_recovers_a_positive_ordered_slope() -> None:
    tables = [b5_analysis.cell_table(cell) for cell in family("positive_monotonic")]
    values = contrast_by_size(tables)
    true_slope = AMPLITUDE * b5_analysis.PRACTICAL_SLOPE
    assert cross_slope(tables) == pytest.approx(true_slope, rel=0.2)
    assert values == sorted(values)


def test_flat_curve_recovers_a_level_and_no_slope() -> None:
    tables = [b5_analysis.cell_table(cell) for cell in family("flat_positive")]
    values = contrast_by_size(tables)
    level = AMPLITUDE * b5_analysis.PRACTICAL_CONTRAST
    assert cross_slope(tables) == pytest.approx(0.0, abs=0.1 * level)
    assert all(value == pytest.approx(level, rel=0.2) for value in values)


def test_decreasing_curve_recovers_a_negative_slope() -> None:
    tables = [b5_analysis.cell_table(cell) for cell in family("decreasing")]
    true_slope = -AMPLITUDE * b5_analysis.PRACTICAL_SLOPE
    assert cross_slope(tables) == pytest.approx(true_slope, rel=0.2)


def test_nonlinear_curve_peaks_in_the_middle_with_a_flat_slope() -> None:
    tables = [b5_analysis.cell_table(cell) for cell in family("non_monotonic")]
    values = contrast_by_size(tables)
    assert b5.SIZES[int(np.argmax(values))] == 8
    assert cross_slope(tables) == pytest.approx(0.0, abs=0.1 * max(values))
    assert values != sorted(values)


def test_null_curve_recovers_zero_at_every_size() -> None:
    tables = [b5_analysis.cell_table(cell) for cell in family("null")]
    values = contrast_by_size(tables)
    assert cross_slope(tables) == pytest.approx(0.0, abs=b5_analysis.PRACTICAL_SLOPE)
    assert all(abs(value) < b5_analysis.PRACTICAL_CONTRAST for value in values)


def test_secondary_tables_report_every_required_quantity() -> None:
    table = b5_analysis.cell_table(family("positive_monotonic")[0])
    row = table["by_size"][-1]
    for key in (
        "joint_covariance",
        "exact_covariance",
        "raw_contrast",
        "joint_standardized",
        "exact_standardized",
        "marginal_error_joint",
        "marginal_error_exact",
        "approximate_variance_inflation",
        "assignment_range",
    ):
        assert key in row
    assert len(row["per_assignment"]) == b5.N_ASSIGNMENTS
    assert row["per_assignment"][0]["joint_label_homogeneity"] > 0.0
    assert row["approximate_variance_inflation"] == pytest.approx(
        1.0 + (32 - 1) * row["joint_standardized"]
    )
    assert table["slope_raw"] == b5_analysis._slope(
        list(b5.SIZES), [float(each["raw_contrast"]) for each in table["by_size"]]
    )


# --- resampling and regeneration --------------------------------------------


def test_bootstrap_resamples_size_32_super_groups_only() -> None:
    cell = family("null")[0]
    assert cell.blocks(1, "joint", 2).shape == (b5.N_ITEMS // 32, 32)
    draws = b5_analysis.bootstrap([cell], 200)
    assert draws["unit"] == "size-32 super-group within cell and assignment"
    assert draws["cell_slope"].shape == (200, 1)
    assert draws["cell_size_contrast"].shape == (200, 1, len(b5.SIZES))
    with pytest.raises(ValueError, match="at least 200"):
        b5_analysis.bootstrap([cell], 10)


def test_analysis_regenerates_byte_for_byte() -> None:
    first = b5_analysis.analyze(family("null"), n_bootstrap=200, n_replay=20)
    second = b5_analysis.analyze(family("null"), n_bootstrap=200, n_replay=20)
    assert first["result_sha256"] == second["result_sha256"]
    assert first == second


def test_analysis_reports_the_frozen_secondary_views() -> None:
    result = b5_analysis.analyze(family("positive_monotonic"), n_bootstrap=200, n_replay=20)
    cross = result["cross_family"]
    assert set(cross["model_slopes"]) == {b5.split_cell(key)[0] for key in b5.CELLS}
    assert set(cross["leave_one_family_out_slope"]) == set(cross["model_slopes"])
    assert set(cross["categorical_by_size"]) == {str(k) for k in b5.SIZES}
    assert set(cross["categorical_by_size"]["32"]["holm_adjusted_p"]) == set(b5.CELLS)
    assert result["historical_size_16_b3"]["values"] == {}
    assert "never pooled" in result["historical_size_16_b3"]["use"]
    assert result["primary"]["cross_family_slope_interval"]["low"] is not None


def test_replay_test_reaches_a_small_p_value_on_a_real_effect() -> None:
    cell = family("positive_monotonic")[0]
    strong = b5_analysis.replay_test(cell, 32, 99)
    assert strong["observed"] > 0.0
    assert strong["p_value_one_sided_upper"] == pytest.approx(strong["min_reachable_p"])
    flat = b5_analysis.replay_test(family("null")[0], 32, 99)
    assert flat["p_value_one_sided_upper"] > 0.05


# --- thresholds and the six readings -----------------------------------------


def test_practical_thresholds_derive_from_the_inflation_floor_not_from_0_002() -> None:
    assert b5_analysis.PRACTICAL_CONTRAST == pytest.approx(0.25 / 15.0)
    assert b5_analysis.PRACTICAL_SLOPE == pytest.approx(b5_analysis.PRACTICAL_CONTRAST / 4.0)
    assert b5_analysis.PRACTICAL_CONTRAST != 0.002
    assert 1.0 + (16 - 1) * b5_analysis.PRACTICAL_CONTRAST == pytest.approx(
        b5_analysis.PRACTICAL_INFLATION
    )


def test_all_six_interpretations_are_frozen() -> None:
    assert set(b5_analysis.OUTCOME_INTERPRETATIONS) == {
        "positive_monotonic",
        "flat_positive",
        "threshold",
        "non_monotonic",
        "null",
        "inconclusive",
    }


def result_for(
    slope: float, low: float, high: float, per_size: list[float], cell_width: float = 0.001
) -> dict[str, Any]:
    step = b5_analysis.PRACTICAL_SLOPE
    return {
        "primary": {
            "cross_family_slope": slope,
            "cross_family_slope_interval": {"low": low, "high": high, "width": high - low},
        },
        "cross_family": {
            "standardized_contrast_by_size": {
                str(k): per_size[index] for index, k in enumerate(b5.SIZES)
            },
            "standardized_contrast_interval_by_size": {
                str(k): {"low": per_size[index] - step, "high": per_size[index] + step,
                         "width": 2 * step}
                for index, k in enumerate(b5.SIZES)
            },
        },
        "cells": [
            {
                "slope_standardized": slope,
                "slope_interval": {"low": low, "high": high, "width": cell_width},
            }
            for _ in b5.CELLS
        ],
    }


def test_reading_1_positive_monotonic() -> None:
    step = b5_analysis.PRACTICAL_SLOPE
    curve = [step * index for index in range(1, 6)]
    matched = b5_analysis.classify(result_for(step, step / 2, step * 1.4, curve))["matched"]
    assert "positive_monotonic" in matched


def test_reading_2_flat_positive() -> None:
    step = b5_analysis.PRACTICAL_SLOPE
    level = b5_analysis.PRACTICAL_CONTRAST * 2
    matched = b5_analysis.classify(
        result_for(0.0, -step * 0.4, step * 0.4, [level] * 5)
    )["matched"]
    assert "flat_positive" in matched
    assert "positive_monotonic" not in matched


def test_reading_3_threshold_boundary() -> None:
    step = b5_analysis.PRACTICAL_SLOPE
    contrast = b5_analysis.PRACTICAL_CONTRAST
    curve = [0.0, 0.0, contrast * 0.5, contrast * 1.5, contrast * 2.0]
    assert "threshold" in b5_analysis.classify(
        result_for(step, step * 0.4, step * 1.2, curve)
    )["matched"]


def test_reading_4_non_monotonic() -> None:
    step = b5_analysis.PRACTICAL_SLOPE
    contrast = b5_analysis.PRACTICAL_CONTRAST
    curve = [0.0, contrast * 4, contrast * 8, contrast * 4, 0.0]
    matched = b5_analysis.classify(result_for(0.0, -step * 0.3, step * 0.3, curve))["matched"]
    assert "non_monotonic" in matched
    assert "positive_monotonic" not in matched


def test_reading_5_null_does_not_replicate() -> None:
    step = b5_analysis.PRACTICAL_SLOPE
    matched = b5_analysis.classify(
        result_for(0.0, -step * 0.2, step * 0.2, [0.0] * 5)
    )["matched"]
    assert "null" in matched
    assert "flat_positive" not in matched


def test_reading_6_wide_interval_is_inconclusive_not_equivalence() -> None:
    step = b5_analysis.PRACTICAL_SLOPE
    outcome = b5_analysis.classify(result_for(0.0, -step * 3, step * 3, [0.0] * 5))
    assert "inconclusive" in outcome["matched"]
    assert "not evidence of equivalence" in outcome["interpretations"]["inconclusive"]
    assert "equivalence" in outcome["note"]


def test_success_is_not_defined_as_a_positive_slope() -> None:
    step = b5_analysis.PRACTICAL_SLOPE
    for reading in ("null", "flat_positive", "non_monotonic", "inconclusive"):
        assert reading in b5_analysis.OUTCOME_INTERPRETATIONS
    outcome = b5_analysis.classify(result_for(0.0, -step * 0.2, step * 0.2, [0.0] * 5))
    assert "not defined as a positive slope" in outcome["note"]


# --- loading receipts --------------------------------------------------------


SMALL = 256


def small_cell() -> b5_analysis.CellData:
    ids = [f"s{index:04d}" for index in range(SMALL)]
    return b5_analysis.CellData(CELL, ids, dict.fromkeys(ids, 0))


def small_events(**override: Any) -> dict[str, dict[str, Any]]:
    cell = small_cell()
    events: dict[str, dict[str, Any]] = {}
    for k in b5.SIZES:
        for assignment in range(1, b5.N_ASSIGNMENTS + 1):
            permutation = b5.canonical_permutation(CELL, assignment, SMALL)
            for group in b5.group_positions(permutation, k):
                order = [cell.doc_ids[position] for position in group]
                for targets, condition in [(order, "joint")] + [
                    ([doc_id], "exact_context") for doc_id in order
                ]:
                    key = f"{k}:{assignment}:{condition}:{targets[0]}:{len(targets)}"
                    events[key] = {
                        "cell_key": CELL,
                        "batch_size": k,
                        "assignment": assignment,
                        "condition": condition,
                        "target_ids": targets,
                        "status": "ok",
                        "parse_status": "ok",
                        "predictions": dict.fromkeys(targets, 1),
                        **override,
                    }
    return events


def test_load_builds_permutation_ordered_error_vectors() -> None:
    cell = small_cell()
    cell.load(small_events())
    vector = cell.values[(8, 1, "joint")]
    assert vector.shape == (SMALL,)
    assert not np.isnan(vector).any()
    assert vector.sum() == SMALL
    assert cell.quality[(8, 1, "joint")]["resolved_calls"] == SMALL // 8


def test_terminal_failures_stay_in_the_ledger_as_unresolved_rows() -> None:
    events = small_events()
    for key, event in events.items():
        if event["batch_size"] == 8 and event["condition"] == "joint" and key.endswith(":8"):
            event.update({"status": "truncated", "parse_status": "failed", "predictions": None})
            break
    cell = small_cell()
    cell.load(events)
    counts = cell.quality[(8, 1, "joint")]
    other = cell.quality[(8, 2, "joint")]
    assert counts["terminal_failures"] + other["terminal_failures"] == 1
    assert counts["calls"] == SMALL // 8
    assert int(np.isnan(cell.values[(8, 1, "joint")]).sum()) + int(
        np.isnan(cell.values[(8, 2, "joint")]).sum()
    ) == 8


def test_load_rejects_a_duplicate_target() -> None:
    events = small_events()
    first = next(iter(events.values()))
    events["extra"] = dict(first)
    with pytest.raises(ValueError, match="duplicate"):
        small_cell().load(events)


def test_load_rejects_incomplete_coverage() -> None:
    events = small_events()
    events.pop(next(iter(events)))
    with pytest.raises(ValueError, match="covers"):
        small_cell().load(events)


def test_a_wrong_prediction_key_set_never_counts_as_resolved() -> None:
    events = small_events()
    for event in events.values():
        if event["condition"] == "joint" and event["batch_size"] == 2:
            event["predictions"] = {"nobody": 1}
    cell = small_cell()
    cell.load(events)
    assert cell.quality[(2, 1, "joint")]["resolved_calls"] == 0
    assert bool(np.isnan(cell.values[(2, 1, "joint")]).all())


# --- power and precision -----------------------------------------------------


def test_six_shapes_are_simulated_including_null_and_decreasing() -> None:
    shapes = b5_power.scenario_contrasts()
    assert set(shapes) == {
        "positive_monotonic",
        "flat_positive",
        "threshold",
        "non_monotonic",
        "decreasing",
        "null",
    }
    assert set(shapes["positive_monotonic"]) == set(b5.SIZES)
    assert all(value == 0.0 for value in shapes["null"].values())
    assert shapes["decreasing"][32] < shapes["decreasing"][2]


def test_shared_uniform_keeps_every_item_marginal_exactly() -> None:
    rates = b5_power.item_rates(0.25, 4096, 5)
    rng = np.random.default_rng(3)
    drawn = np.mean(
        [b5_power.draw_errors(rates, 4096 // 8, 8, 0.3, rng) for _ in range(200)], axis=0
    )
    assert float(np.mean(drawn)) == pytest.approx(float(np.mean(rates)), abs=0.004)
    assert float(np.corrcoef(drawn, rates)[0, 1]) > 0.9


def test_draw_errors_rejects_an_impossible_correlation() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="must lie in"):
        b5_power.draw_errors(b5_power.item_rates(0.2, 64, 1), 8, 8, 1.0, rng)


def test_attenuation_is_a_fraction_that_shrinks_with_heterogeneity() -> None:
    uniform = np.full(4096, 0.25)
    spread = b5_power.item_rates(0.25, 4096, 9)
    assert b5_power.attenuation(uniform, 2) == pytest.approx(1.0, abs=0.01)
    assert 0.0 < b5_power.attenuation(spread, 2) < 1.0


def test_simulation_reports_precision_for_every_shape() -> None:
    planning = [
        {"cell_key": key, "marginal_error": 0.2, "exact_standardized": 0.01} for key in b5.CELLS
    ]
    record = b5_power.finalize(b5_power.simulate(planning, n_replicates=50))
    assert set(record["scenarios"]) == set(b5_power.scenario_contrasts())
    assert set(record["half_width_below_practical_slope"]) == set(record["scenarios"])
    for row in record["scenarios"].values():
        assert row["slope_standard_error"] > 0.0
        assert set(row["per_size_standard_error"]) == {str(k) for k in b5.SIZES}
    assert record["scenarios"]["null"]["rejection_rate_against_zero"] < 0.15
    assert record["record_sha256"]
    with pytest.raises(ValueError, match="at least 50"):
        b5_power.simulate(planning, n_replicates=10)


def test_simulation_states_what_the_frame_cannot_resolve() -> None:
    planning = [
        {"cell_key": key, "marginal_error": 0.2, "exact_standardized": 0.01} for key in b5.CELLS
    ]
    record = b5_power.simulate(planning, n_replicates=50)
    assert "not resolvable by this design" in record["what_8192_items_resolve"]
    assert "no B5 output exists" in record["planning_source"]
