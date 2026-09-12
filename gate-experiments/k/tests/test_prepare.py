"""Invariant checks reported by prepare-synthetic."""

from __future__ import annotations

from pathlib import Path

from gatek.config import Config, GateConfig, PlanConfig, RunConfig, SyntheticConfig
from gatek.prepare import prepare_synthetic

TINY = Config(
    name="tiny",
    synthetic=SyntheticConfig(dims=[2, 8], seeds=[0], n_demo=12, n_query=4),
    plan=PlanConfig(ablation_dims=[8], n_ablated_rows=2),
    run=RunConfig(backend="mock"),
    # Four held-out rows make accuracy move in 0.25 steps. This fixture tests
    # construction plumbing, not whether a fitted baseline generalizes.
    gate=GateConfig(min_classical_accuracy=0.2, min_nuisance_raw_contrast_ratio=1.2, high_dims=[8]),
)


def test_all_invariants_pass_on_the_tiny_config(tmp_path: Path) -> None:
    manifest = prepare_synthetic(TINY, tmp_path)
    assert manifest["construction_ok"], manifest["failed_invariants"]
    names = {c["check"] for c in manifest["invariants"]}
    assert {
        "latent_rows_identical",
        "labels_identical",
        "label_rule_holds",
        "class_balance_identical",
        "feature_names_identical",
        "numeric_format_fixed_width",
        "latent_concentration_identical",
        "observed_classical_diagnostic",
        "nuisance_raw_contrast",
        "token_length_recorded",
    } <= names


def test_nuisance_contrast_check_is_skipped_at_dimension_two(tmp_path: Path) -> None:
    manifest = prepare_synthetic(TINY, tmp_path)
    at_two = [
        c for c in manifest["invariants"] if c["check"] == "nuisance_raw_contrast" and c["dim"] == 2
    ]
    assert at_two and "skipped" in at_two[0]["detail"]


def test_an_impossible_threshold_is_reported_as_construction_failure(tmp_path: Path) -> None:
    strict = Config(
        name="strict",
        synthetic=TINY.synthetic,
        plan=TINY.plan,
        run=TINY.run,
        gate=GateConfig(min_nuisance_raw_contrast_ratio=1000.0, high_dims=[8]),
    )
    manifest = prepare_synthetic(strict, tmp_path)
    assert not manifest["construction_ok"]
    assert "nuisance_raw_contrast" in manifest["failed_invariants"]


def test_manifest_records_checksums_and_environment(tmp_path: Path) -> None:
    manifest = prepare_synthetic(TINY, tmp_path)
    assert len(manifest["artifacts"]["data_npz_sha256"]) == 64
    assert manifest["environment"]["numpy"]
