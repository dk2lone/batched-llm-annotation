"""Null and power simulations on the exact frozen design.

Every scenario uses the real 2,048-item frame, k = 16, 128 calls per assignment, 3
assignments, the real length strata, and the real assignment generator. The
simulations resolve five protocol fields: the Monte Carlo draw rationale, the solo-screen
error count and rate, the high-error-rate boundary, and the corresponding accuracy
ceiling. They also diagnose whether the design supports the 0.002 covariance floor.

The scenario grid comes only from the disclosed discovery inputs recorded in the
protocol. No pilot per-assignment value, interval bound, or verdict is used.

Speed note. The production randomization test in `step0.stats` calls the frozen
generator `step0.assign.replay_stratum_matrices` for every draw. A simulation needs
millions of draws, so it samples independent uniform permutations inside each
length-stratum-by-execution-block cell instead. That is the same conditional null
distribution as the frozen generator. `tests/test_simulate.py` checks the two samplers
against each other on the real frame.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any

import numpy as np

from .assign import N_BLOCKS, Frame, observed_partition
from .hashing import derived_seed, write_json, write_text
from .lock import NO_NEW_LABELS, SIMULATION_SPEC, TESTS
from .stats import covariance_from_stratified

COVARIANCE_FLOOR = float(TESTS["practical_floors"]["covariance_floor_absolute"])
POWER_TARGET = float(SIMULATION_SPEC["power_target"])
ALPHA = float(SIMULATION_SPEC["alpha"])
HOLM_ALPHA_3 = float(SIMULATION_SPEC["holm_adjusted_alpha_for_three_cells"])


def synthetic_frame(n_items: int = 2048, n_strata: int = 16) -> Frame:
    """A frame with the real shape. Only the shape enters a simulation."""
    per = n_items // n_strata
    doc_ids = tuple(f"sim{i:06d}" for i in range(n_items))
    strata = tuple(i // per for i in range(n_items))
    return Frame(doc_ids=doc_ids, strata=strata)


def stratum_matrices(frame: Frame, cell_id: str, n_assignments: int) -> list[np.ndarray]:
    """Observed (n_strata, n_calls) item-index matrices, one per assignment."""
    return [
        observed_partition(frame, cell_id, a).T.copy() for a in range(1, n_assignments + 1)
    ]


def draw_joint_errors(
    rng: np.random.Generator, n_calls: int, k: int, p: float, delta: float
) -> np.ndarray:
    """Beta-binomial exchangeable errors, shaped (n_calls, k).

    E[C] = rho * p * (1 - p) for this model, so rho = delta / (p * (1 - p)) puts the
    expected pair-weighted covariance at `delta`.
    """
    if delta <= 0:
        return (rng.random((n_calls, k)) < p).astype(float)
    rho = delta / (p * (1.0 - p))
    if rho >= 1.0:
        raise ValueError(f"delta {delta} is unreachable at marginal rate {p}: rho would be {rho}")
    total = 1.0 / rho - 1.0
    call_rates = rng.beta(p * total, (1.0 - p) * total, size=n_calls)
    return (rng.random((n_calls, k)) < call_rates[:, None]).astype(float)


def _cube_covariance(cube: np.ndarray) -> np.ndarray:
    n_draws, n_calls, k = cube.shape
    means = cube.reshape(n_draws, -1).mean(axis=1)[:, None, None]
    centred = cube - means
    sums = centred.sum(axis=2)
    values: np.ndarray = (
        (sums**2).sum(axis=1) - (centred**2).sum(axis=(1, 2))
    ) / (n_calls * k * (k - 1))
    return values


def _regroup(values: np.ndarray, matrix: np.ndarray, perms: np.ndarray) -> np.ndarray:
    n_draws, n_strata, n_per = perms.shape
    gathered = np.take_along_axis(
        np.broadcast_to(matrix, (n_draws, n_strata, n_per)), perms, axis=2
    )
    cube: np.ndarray = values[gathered].transpose(0, 2, 1)
    return cube


def _block_restricted_permutations(
    rng: np.random.Generator, size: int, n_strata: int, n_calls: int
) -> np.ndarray:
    """Permutation indices that never move an item between execution blocks."""
    perms = np.broadcast_to(
        np.arange(n_calls, dtype=np.int64), (size, n_strata, n_calls)
    ).copy()
    for execution_block in range(N_BLOCKS):
        call_indices = np.arange(execution_block, n_calls, N_BLOCKS)
        local = np.argsort(rng.random((size, n_strata, call_indices.size)), axis=2)
        perms[:, :, call_indices] = call_indices[local]
    return perms


def one_trial(
    matrices: list[np.ndarray],
    p_joint: float,
    p_exact: float,
    delta: float,
    n_draws: int,
    rng: np.random.Generator,
    block: int = 250,
) -> dict[str, float]:
    """Simulate one cell, then run the frozen randomization test on it."""
    n_strata, n_calls = matrices[0].shape
    k = n_strata
    joint_items: list[np.ndarray] = []
    exact_items: list[np.ndarray] = []
    observed = 0.0
    for matrix in matrices:
        joint_calls = draw_joint_errors(rng, n_calls, k, p_joint, delta)
        exact_calls = (rng.random((n_calls, k)) < p_exact).astype(float)
        observed += covariance_from_stratified(joint_calls) - covariance_from_stratified(
            exact_calls
        )
        joint_flat = np.empty(n_calls * k)
        exact_flat = np.empty(n_calls * k)
        joint_flat[matrix.T.ravel()] = joint_calls.ravel()
        exact_flat[matrix.T.ravel()] = exact_calls.ravel()
        joint_items.append(joint_flat)
        exact_items.append(exact_flat)
    observed /= len(matrices)

    null = np.empty(n_draws)
    done = 0
    while done < n_draws:
        size = min(block, n_draws - done)
        totals = np.zeros(size)
        for a, matrix in enumerate(matrices):
            perms = _block_restricted_permutations(
                rng, size, matrix.shape[0], matrix.shape[1]
            )
            totals += _cube_covariance(_regroup(joint_items[a], matrix, perms))
            totals -= _cube_covariance(_regroup(exact_items[a], matrix, perms))
        null[done : done + size] = totals / len(matrices)
        done += size

    p_value = (1 + int((null >= observed).sum())) / (1 + n_draws)
    return {
        "observed": float(observed),
        "p_value": float(p_value),
        "null_sd": float(null.std(ddof=1)),
        "effect_low": float(observed - np.percentile(null - null.mean(), 100 * (1 - ALPHA / 2))),
        "effect_high": float(observed - np.percentile(null - null.mean(), 100 * (ALPHA / 2))),
    }


def run_scenario(
    matrices: list[np.ndarray],
    p_joint: float,
    p_exact: float,
    delta: float,
    n_sim: int,
    n_draws: int,
    seed_key: tuple[object, ...],
) -> dict[str, Any]:
    started = time.monotonic()
    p_values = np.empty(n_sim)
    estimates = np.empty(n_sim)
    lows = np.empty(n_sim)
    highs = np.empty(n_sim)
    for trial in range(n_sim):
        rng = np.random.default_rng(derived_seed(*seed_key, trial))
        result = one_trial(matrices, p_joint, p_exact, delta, n_draws, rng)
        p_values[trial] = result["p_value"]
        estimates[trial] = result["observed"]
        lows[trial] = result["effect_low"]
        highs[trial] = result["effect_high"]
    reject = p_values <= ALPHA
    reject_holm3 = p_values <= HOLM_ALPHA_3
    above_floor = estimates >= COVARIANCE_FLOOR
    holm_rate = float(reject_holm3.mean())
    return {
        "monte_carlo_se_of_holm_rate": float(np.sqrt(holm_rate * (1 - holm_rate) / n_sim)),
        "p_joint": p_joint,
        "p_exact": p_exact,
        "delta_true": delta,
        "n_sim": n_sim,
        "n_draws": n_draws,
        "rejection_rate_alpha_0_05": float(reject.mean()),
        "rejection_rate_holm3_alpha_0_0167": float(reject_holm3.mean()),
        "joint_rejection_and_above_floor": float((reject_holm3 & above_floor).mean()),
        "mean_estimate": float(estimates.mean()),
        "bias": float(estimates.mean() - delta),
        "sd_estimate": float(estimates.std(ddof=1)),
        "effect_interval_coverage_of_truth": float(
            ((lows <= delta) & (delta <= highs)).mean()
        ),
        "mean_p_value": float(p_values.mean()),
        "seconds": round(time.monotonic() - started, 2),
    }


# --------------------------------------------------------------------------------------
# The three frozen questions
# --------------------------------------------------------------------------------------


def monte_carlo_study(
    matrices: list[np.ndarray], p: float, n_sim: int, counts: list[int]
) -> list[dict[str, Any]]:
    """Type-I rate and p-value resolution at several Monte Carlo draw counts."""
    out = []
    for count in counts:
        row = run_scenario(matrices, p, p, 0.0, n_sim, count, ("mc", p, count))
        row["min_reachable_p"] = 1.0 / (1 + count)
        row["resolution_ok_for_holm3"] = bool(row["min_reachable_p"] < HOLM_ALPHA_3)
        row["monte_carlo_se_at_holm_threshold"] = float(
            np.sqrt(HOLM_ALPHA_3 * (1 - HOLM_ALPHA_3) / count)
        )
        out.append(row)
    return out


def information_threshold(
    matrices: list[np.ndarray], rates: list[float], n_sim: int, n_draws: int, n_items: int
) -> dict[str, Any]:
    """Detection power at the frozen floor across the marginal error-rate grid.

    The threshold is the smallest prespecified grid rate whose detection power reaches
    the frozen target. The grid includes points below and immediately above the arithmetic
    reachability boundary. Detection power is the rejection rate at the three-cell
    Holm-adjusted alpha when the true excess covariance equals the floor. The joint claim gate,
    which also demands a point estimate at or above the floor, is reported next to it and is
    deliberately not the threshold: an unbiased estimator clears a boundary equal to the
    truth about half the time, so that gate cannot reach 0.90 at the boundary for any
    design. The effect is never redefined to make a target pass.
    """
    rows: list[dict[str, Any]] = []
    for p in rates:
        reachable = COVARIANCE_FLOOR < p * (1.0 - p)
        row: dict[str, Any] = {
            "p": p,
            "max_reachable_covariance": p * (1.0 - p),
            "floor_is_reachable_at_all": bool(reachable),
            "implied_rho_at_the_floor": (
                COVARIANCE_FLOOR / (p * (1.0 - p)) if reachable else None
            ),
        }
        if reachable:
            row.update(run_scenario(matrices, p, p, COVARIANCE_FLOOR, n_sim, n_draws, ("thr", p)))
        else:
            row.update(
                {
                    "rejection_rate_alpha_0_05": 0.0,
                    "rejection_rate_holm3_alpha_0_0167": 0.0,
                    "joint_rejection_and_above_floor": 0.0,
                    "monte_carlo_se_of_holm_rate": 0.0,
                    "note": (
                        "p(1-p) is below the 0.002 floor, so no within-call correlation can "
                        "reach it. The cell is information-poor by arithmetic, not by outcome."
                    ),
                }
            )
        rows.append(row)

    passing = [
        r for r in rows if float(r["rejection_rate_holm3_alpha_0_0167"]) >= POWER_TARGET
    ]
    chosen = min((float(r["p"]) for r in passing), default=None)
    highest = max((float(r["p"]) for r in passing), default=None)
    failing_above = [
        float(r["p"])
        for r in rows
        if highest is not None
        and float(r["p"]) > highest
        and float(r["rejection_rate_holm3_alpha_0_0167"]) < POWER_TARGET
    ]
    return {
        "sweep": rows,
        "power_target": POWER_TARGET,
        "criterion": (
            "detection power: the trial rejects at the three-cell Holm-adjusted alpha when the "
            "true excess covariance equals the 0.002 floor"
        ),
        "secondary_criterion_reported_not_used": (
            "the joint claim gate also requires the point estimate to reach the floor. At a true "
            "effect exactly equal to the floor an unbiased estimator meets it about half the "
            "time, so this rate is near 0.5 by construction and is not a power failure."
        ),
        "min_error_rate_meeting_the_target": chosen,
        "max_error_rate_meeting_the_target": highest,
        "error_rates_above_the_band_that_miss_the_target": failing_above,
        "min_solo_errors": None if chosen is None else int(np.ceil(chosen * n_items)),
        "n_items": n_items,
        "frozen_threshold_note": (
            "this smallest qualifying prespecified grid value becomes "
            "solo_screen.criteria.error_information_threshold. A cell below it is labelled "
            "information-poor and stays in the published panel."
        ),
        "high_error_rate_note": (
            "at a fixed absolute floor the implied within-call correlation falls as p(1-p) grows, "
            "so detection power falls at high marginal error rates. Any rate above the band is "
            "reported as a disclosed power limit of the fixed design, not as a screen failure."
        ),
    }


def run_all(
    out_dir: Path,
    n_sim: int = 300,
    n_draws_sim: int = 999,
    n_draws_production: int = 2000,
    n_sim_threshold: int = 200,
    n_sim_mc: int = 200,
) -> dict[str, Any]:
    """Run every frozen simulation and write results.json plus REPORT.md."""
    started = time.monotonic()
    frame = synthetic_frame()
    matrices = stratum_matrices(frame, "simulation_cell", 3)
    grid = SIMULATION_SPEC["scenario_grid"]

    scenarios: list[dict[str, Any]] = []
    for p in grid["marginal_error_rate_p"]:
        for delta in grid["true_excess_covariance_delta"]:
            if delta > 0 and delta >= p * (1 - p):
                scenarios.append(
                    {
                        "p_joint": p,
                        "p_exact": p,
                        "delta_true": delta,
                        "skipped": True,
                        "reason": f"delta {delta} exceeds p(1-p) = {p * (1 - p):.5f}",
                    }
                )
                continue
            scenarios.append(
                run_scenario(matrices, p, p, delta, n_sim, n_draws_sim, ("grid", p, delta))
            )

    shift = run_scenario(matrices, 0.113, 0.083, 0.0, n_sim, n_draws_sim, ("shift",))
    shift["label"] = "marginal shift under the true null: exact arm 3 points more accurate"

    mc = monte_carlo_study(matrices, 0.113, n_sim_mc, grid["monte_carlo_count_study"])
    threshold = information_threshold(
        matrices, grid["information_threshold_sweep_p"], n_sim_threshold, n_draws_sim,
        frame.n_items,
    )

    at_floor = [
        s
        for s in scenarios
        if not s.get("skipped") and s["delta_true"] == COVARIANCE_FLOOR
    ]
    at_pilot_like = [
        s for s in scenarios if not s.get("skipped") and s["delta_true"] == 0.0075
    ]
    nulls = [s for s in scenarios if not s.get("skipped") and s["delta_true"] == 0.0]

    floor_detectable = {
        str(s["p_joint"]): float(s["rejection_rate_holm3_alpha_0_0167"]) for s in at_floor
    }
    floor_supported = any(rate >= POWER_TARGET for rate in floor_detectable.values())
    results: dict[str, Any] = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "no_new_experimental_model_labels": NO_NEW_LABELS,
        "design": {
            "n_items": frame.n_items,
            "batch_size_k": 16,
            "n_calls_per_assignment": frame.n_calls,
            "n_assignments": 3,
            "n_strata": 16,
            "generator": "the frozen operational generator in step0.assign",
        },
        "settings": {
            "n_sim_grid": n_sim,
            "n_sim_threshold": n_sim_threshold,
            "n_sim_monte_carlo_study": n_sim_mc,
            "n_draws_in_simulation": n_draws_sim,
            "n_draws_frozen_for_production": n_draws_production,
            "alpha": ALPHA,
            "holm_alpha_three_cells": HOLM_ALPHA_3,
            "power_target": POWER_TARGET,
            "covariance_floor": COVARIANCE_FLOOR,
        },
        "scenarios": scenarios,
        "marginal_shift_null": shift,
        "monte_carlo_count_study": mc,
        "information_threshold": threshold,
        "diagnostics": {
            "typeI_rate_at_delta_zero": {
                str(s["p_joint"]): s["rejection_rate_alpha_0_05"] for s in nulls
            },
            "typeI_rate_with_marginal_shift": shift["rejection_rate_alpha_0_05"],
            "power_at_floor_0_002": floor_detectable,
            "power_at_floor_0_002_joint_claim_gate": {
                str(s["p_joint"]): s["joint_rejection_and_above_floor"] for s in at_floor
            },
            "power_at_pilot_like_0_0075": {
                str(s["p_joint"]): s["rejection_rate_holm3_alpha_0_0167"] for s in at_pilot_like
            },
            "power_at_pilot_like_0_0075_joint_claim_gate": {
                str(s["p_joint"]): s["joint_rejection_and_above_floor"] for s in at_pilot_like
            },
            "information_threshold_min_error_rate": threshold["min_error_rate_meeting_the_target"],
            "monte_carlo_resolution": {
                str(row["n_draws"]): row["min_reachable_p"] for row in mc
            },
            "estimator_bias": {
                f"p={s['p_joint']},delta={s['delta_true']}": s["bias"]
                for s in scenarios
                if not s.get("skipped")
            },
            "effect_interval_coverage": {
                f"p={s['p_joint']},delta={s['delta_true']}": s[
                    "effect_interval_coverage_of_truth"
                ]
                for s in scenarios
                if not s.get("skipped")
            },
        },
        "frozen_conclusions": {
            "monte_carlo_draws": n_draws_production,
            "monte_carlo_draws_justification": (
                f"the count study reports Type-I rates at "
                f"{grid['monte_carlo_count_study']} draws without using those rates to tune the "
                f"production count. At {n_draws_production} draws the smallest reachable p is "
                f"{1 / (1 + n_draws_production):.6f}, far below the three-cell Holm threshold of "
                f"{HOLM_ALPHA_3}."
            ),
            "error_information_threshold_error_rate": threshold[
                "min_error_rate_meeting_the_target"
            ],
            "error_information_threshold_min_solo_errors": threshold["min_solo_errors"],
            "error_information_threshold_upper_error_rate": threshold[
                "max_error_rate_meeting_the_target"
            ],
            "error_rates_above_the_band_that_miss_the_target": threshold[
                "error_rates_above_the_band_that_miss_the_target"
            ],
            "covariance_floor_0_002_supported_by_the_fixed_design": floor_supported,
            "covariance_floor_detection_power_by_error_rate": floor_detectable,
            "covariance_floor_decision": (
                "kept at 0.002"
                if floor_supported
                else "kept at 0.002 and reported as underpowered for the fixed design"
            ),
            "named_power_and_typeI_diagnostics": SIMULATION_SPEC["diagnostics"],
        },
        "boundary": SIMULATION_SPEC["tuning_boundary"],
        "total_seconds": round(time.monotonic() - started, 1),
    }
    write_json(out_dir / "results.json", results)
    write_text(out_dir / "REPORT.md", render_report(results))
    return results


def render_report(r: dict[str, Any]) -> str:
    lines = [
        "# Null and power simulation for the frozen design",
        "",
        f"Created {r['created_at_utc']}. Total runtime {r['total_seconds']} s.",
        "",
        "Every scenario uses the exact frozen design: "
        f"{r['design']['n_items']} items, k = {r['design']['batch_size_k']}, "
        f"{r['design']['n_calls_per_assignment']} calls per assignment, "
        f"{r['design']['n_assignments']} assignments, {r['design']['n_strata']} length strata, "
        "and the real assignment generator.",
        "",
        r["no_new_experimental_model_labels"],
        "",
        f"Every rejection rate below is a Monte Carlo estimate. The grid uses "
        f"{r['settings']['n_sim_grid']} trials per scenario, the threshold sweep uses "
        f"{r['settings']['n_sim_threshold']}, and each trial runs "
        f"{r['settings']['n_draws_in_simulation']} randomization draws. A rate near 0.9 from 200 "
        "trials carries a standard error near 0.021.",
        "",
        "## 1. Scenario grid",
        "",
        "| marginal error rate | true excess covariance | reject at 0.05 | reject at Holm 0.0167 "
        "| reject and above floor | mean estimate | bias | interval covers truth |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in r["scenarios"]:
        if s.get("skipped"):
            lines.append(
                f"| {s['p_joint']} | {s['delta_true']} | skipped | skipped | skipped | - | - | - |"
            )
            continue
        lines.append(
            f"| {s['p_joint']} | {s['delta_true']} | {s['rejection_rate_alpha_0_05']:.3f} | "
            f"{s['rejection_rate_holm3_alpha_0_0167']:.3f} | "
            f"{s['joint_rejection_and_above_floor']:.3f} | {s['mean_estimate']:.5f} | "
            f"{s['bias']:+.5f} | {s['effect_interval_coverage_of_truth']:.3f} |"
        )
    shift = r["marginal_shift_null"]
    lines += [
        "",
        "## 2. Type-I error under a marginal accuracy shift",
        "",
        shift["label"] + ".",
        "",
        f"Rejection rate at alpha 0.05: {shift['rejection_rate_alpha_0_05']:.3f}. "
        f"Mean estimate {shift['mean_estimate']:+.5f}.",
        "",
        "A rate near 0.05 shows the centred statistic is not driven by the arms having different "
        "marginal error rates.",
        "",
        "## 3. Monte Carlo draw count",
        "",
        (
            "| draws | Type-I at 0.05 | smallest reachable p | resolution adequate for Holm over 3 "
            "cells |"
        ),
        "|---:|---:|---:|---|",
    ]
    for row in r["monte_carlo_count_study"]:
        lines.append(
            f"| {row['n_draws']} | {row['rejection_rate_alpha_0_05']:.3f} | "
            f"{row['min_reachable_p']:.6f} | {row['resolution_ok_for_holm3']} |"
        )
    thr = r["information_threshold"]
    lines += [
        "",
        f"Frozen production count: **{r['frozen_conclusions']['monte_carlo_draws']}**.",
        "",
        r["frozen_conclusions"]["monte_carlo_draws_justification"],
        "",
        "## 4. Achieved power across the error-rate grid, and the solo-screen threshold",
        "",
        "Criterion: " + thr["criterion"] + ".",
        "",
        thr["secondary_criterion_reported_not_used"],
        "",
        "| marginal error rate | p(1-p) | implied rho at the floor | floor reachable "
        "| detection power at Holm 0.0167 | Monte Carlo SE | joint claim gate |",
        "|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in thr["sweep"]:
        rho = row["implied_rho_at_the_floor"]
        lines.append(
            f"| {row['p']} | {row['max_reachable_covariance']:.5f} | "
            f"{'-' if rho is None else f'{rho:.4f}'} | "
            f"{row['floor_is_reachable_at_all']} | "
            f"{float(row['rejection_rate_holm3_alpha_0_0167']):.3f} | "
            f"{float(row['monte_carlo_se_of_holm_rate']):.3f} | "
            f"{float(row['joint_rejection_and_above_floor']):.3f} |"
        )
    lines += [
        "",
        f"Frozen threshold: marginal error rate **{thr['min_error_rate_meeting_the_target']}**, "
        f"which is **{thr['min_solo_errors']}** solo errors out of {thr['n_items']} items.",
        "",
        f"Highest grid error rate that still reaches the target: "
        f"**{thr['max_error_rate_meeting_the_target']}**. Grid error rates above that band which "
        f"miss the target: {thr['error_rates_above_the_band_that_miss_the_target']}.",
        "",
        thr["high_error_rate_note"],
        "",
        thr["frozen_threshold_note"],
        "",
        "## 5. The 0.002 covariance floor",
        "",
        f"Detectable at the frozen power target somewhere on the grid: "
        f"**{r['frozen_conclusions']['covariance_floor_0_002_supported_by_the_fixed_design']}**. "
        f"Decision: {r['frozen_conclusions']['covariance_floor_decision']}.",
        "",
        "Detection power at the floor, by marginal error rate:",
        "",
        "| marginal error rate | detection power |",
        "|---:|---:|",
    ]
    for rate, power in r["frozen_conclusions"][
        "covariance_floor_detection_power_by_error_rate"
    ].items():
        lines.append(f"| {rate} | {float(power):.3f} |")
    lines += [
        "",
        "The floor is never lowered and the effect is never redefined. Where p(1-p) is below "
        "0.002 no within-call correlation can reach the floor, so such a cell is information-poor "
        "by arithmetic. Where power at the floor is below the target, the design is disclosed as "
        "underpowered for the smallest meaningful effect at that marginal error rate.",
        "",
        "## 6. Named diagnostics",
        "",
    ]
    for name in r["frozen_conclusions"]["named_power_and_typeI_diagnostics"]:
        lines.append(f"- `{name}`")
    lines += ["", "## 7. Tuning boundary", "", r["boundary"], ""]
    return "\n".join(lines)
