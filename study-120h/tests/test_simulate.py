"""Simulation tests: the error model, the sampler, and the threshold rule.

These run small. The frozen production run is `step0 simulate`, whose results are
copied into the freeze directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from step0 import assign, simulate, stats


def test_the_synthetic_frame_has_the_real_shape() -> None:
    frame = simulate.synthetic_frame()
    assert frame.n_items == 2048
    assert frame.n_calls == 128
    matrices = simulate.stratum_matrices(frame, "cell", 3)
    assert len(matrices) == 3
    assert matrices[0].shape == (16, 128)


def test_the_error_model_reaches_the_requested_covariance() -> None:
    rng = np.random.default_rng(3)
    values = [
        stats.covariance_from_stratified(
            simulate.draw_joint_errors(rng, 128, 16, 0.113, 0.0075)
        )
        for _ in range(200)
    ]
    assert float(np.mean(values)) == pytest.approx(0.0075, abs=0.0008)


def test_the_error_model_is_unbiased_at_the_null() -> None:
    rng = np.random.default_rng(5)
    values = [
        stats.covariance_from_stratified(simulate.draw_joint_errors(rng, 128, 16, 0.1, 0.0))
        for _ in range(200)
    ]
    assert abs(float(np.mean(values))) < 0.0004


def test_an_unreachable_covariance_raises_instead_of_being_clipped() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="unreachable"):
        simulate.draw_joint_errors(rng, 128, 16, 0.01, 0.02)


def test_the_fast_sampler_matches_the_frozen_generator_in_structure() -> None:
    """Both samplers permute within each stratum and observed execution block."""
    frame = simulate.synthetic_frame()
    matrix = simulate.stratum_matrices(frame, "cell", 1)[0]
    rng = np.random.default_rng(11)
    perms = simulate._block_restricted_permutations(
        rng, 4, matrix.shape[0], matrix.shape[1]
    )
    values = (rng.random(frame.n_items) < 0.2).astype(float)
    fast = simulate._regroup(values, matrix, perms)
    assert fast.shape == (4, 128, 16)

    frozen = assign.replay_stratum_matrices(frame, "cell", 1, 0)[0]
    for cube_slice in (fast[0], stats._reshape(values, frozen)):
        assert cube_slice.shape == (128, 16)
    # both samplers keep one item per stratum in every call and use every item once
    for draw in range(4):
        gathered = np.take_along_axis(
            np.broadcast_to(matrix, (4, *matrix.shape)), perms, axis=2
        )[draw]
        assert sorted(gathered.ravel().tolist()) == list(range(frame.n_items))
        for stratum in range(16):
            assert {frame.strata[i] for i in gathered[stratum].tolist()} == {stratum}
            for execution_block in range(assign.N_BLOCKS):
                calls = np.arange(execution_block, frame.n_calls, assign.N_BLOCKS)
                assert set(gathered[stratum, calls]) == set(matrix[stratum, calls])
    assert sorted(frozen.ravel().tolist()) == list(range(frame.n_items))
    for stratum in range(16):
        for execution_block in range(assign.N_BLOCKS):
            calls = np.arange(execution_block, frame.n_calls, assign.N_BLOCKS)
            assert set(frozen[stratum, calls]) == set(matrix[stratum, calls])


def test_one_trial_returns_a_usable_p_value_and_interval() -> None:
    frame = simulate.synthetic_frame(512)
    matrices = simulate.stratum_matrices(frame, "cell", 2)
    rng = np.random.default_rng(2)
    result = simulate.one_trial(matrices, 0.2, 0.2, 0.0, 99, rng)
    assert 1 / 100 <= result["p_value"] <= 1.0
    assert result["effect_low"] <= result["observed"] <= result["effect_high"]


def test_a_strong_true_effect_is_detected() -> None:
    frame = simulate.synthetic_frame(512)
    matrices = simulate.stratum_matrices(frame, "cell", 2)
    rng = np.random.default_rng(8)
    result = simulate.one_trial(matrices, 0.2, 0.2, 0.02, 99, rng)
    assert result["observed"] > 0.005
    assert result["p_value"] <= 0.05


def _canned(monkeypatch: pytest.MonkeyPatch, table: dict[float, tuple[float, float]]) -> None:
    def fake(
        matrices: list[np.ndarray],
        p_joint: float,
        p_exact: float,
        delta: float,
        n_sim: int,
        n_draws: int,
        seed_key: tuple[object, ...],
    ) -> dict[str, Any]:
        holm, gate = table[p_joint]
        return {
            "p_joint": p_joint,
            "delta_true": delta,
            "rejection_rate_alpha_0_05": holm,
            "rejection_rate_holm3_alpha_0_0167": holm,
            "joint_rejection_and_above_floor": gate,
            "monte_carlo_se_of_holm_rate": 0.02,
            "mean_estimate": delta,
            "bias": 0.0,
            "effect_interval_coverage_of_truth": 0.95,
        }

    monkeypatch.setattr(simulate, "run_scenario", fake)


def test_the_threshold_is_the_lowest_rate_reaching_the_detection_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _canned(monkeypatch, {0.02: (0.95, 0.42), 0.05: (0.93, 0.35), 0.2: (0.50, 0.50)})
    result = simulate.information_threshold([np.zeros((16, 4))], [0.02, 0.05, 0.2], 10, 99, 2048)
    assert result["min_error_rate_meeting_the_target"] == 0.02
    assert result["min_solo_errors"] == 41
    assert result["max_error_rate_meeting_the_target"] == 0.05
    assert result["error_rates_above_the_band_that_miss_the_target"] == [0.2]
    assert "detection power" in result["criterion"]


def test_the_claim_gate_at_the_boundary_never_sets_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unbiased estimate clears a boundary equal to the truth about half the time."""
    _canned(monkeypatch, {0.05: (0.99, 0.50)})
    result = simulate.information_threshold([np.zeros((16, 4))], [0.05], 10, 99, 2048)
    assert result["min_error_rate_meeting_the_target"] == 0.05
    assert result["sweep"][0]["joint_rejection_and_above_floor"] == 0.5
    assert "about half the time" in result["secondary_criterion_reported_not_used"]


def test_an_arithmetically_unreachable_floor_is_reported_not_tested() -> None:
    result = simulate.information_threshold([np.zeros((16, 4))], [0.001], 2, 9, 2048)
    row = result["sweep"][0]
    assert not row["floor_is_reachable_at_all"]
    assert row["rejection_rate_holm3_alpha_0_0167"] == 0.0
    assert "information-poor by arithmetic" in row["note"]
    assert result["min_error_rate_meeting_the_target"] is None


def test_run_all_writes_a_report_that_states_power_and_no_new_labels(tmp_path: Path) -> None:
    simulate.run_all(tmp_path, n_sim=2, n_draws_sim=19, n_sim_threshold=2, n_sim_mc=2)
    assert (tmp_path / "results.json").exists()
    report = (tmp_path / "REPORT.md").read_text()
    assert "No new experimental model labels exist" in report
    assert "Achieved power across the error-rate grid" in report
    assert "detection power" in report
    assert "0.002 covariance floor" in report
    stored = json.loads((tmp_path / "results.json").read_text())
    assert stored["settings"]["n_draws_frozen_for_production"] == 2000
    assert set(stored["diagnostics"]) >= {
        "typeI_rate_at_delta_zero",
        "typeI_rate_with_marginal_shift",
        "power_at_floor_0_002",
        "information_threshold_min_error_rate",
        "estimator_bias",
        "effect_interval_coverage",
    }
