"""Freeze creation must bind resolved protocol values to simulation evidence and source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from step0 import freeze


def matched_protocol_and_results() -> tuple[dict[str, Any], dict[str, Any]]:
    rationale = "2,000 draws give sufficient p-value resolution"
    results = {
        "settings": {"covariance_floor": 0.002},
        "frozen_conclusions": {
            "error_information_threshold_min_solo_errors": 21,
            "error_information_threshold_error_rate": 0.01,
            "error_information_threshold_upper_error_rate": 0.15,
            "error_rates_above_the_band_that_miss_the_target": [0.2, 0.3],
            "monte_carlo_draws": 2_000,
            "monte_carlo_draws_justification": rationale,
        },
    }
    protocol = {
        "solo_screen": {
            "criteria": {
                "error_information_threshold": {
                    "min_solo_errors": 21,
                    "min_solo_error_rate": 0.01,
                    "high_error_rate_finding": {
                        "highest_grid_error_rate_meeting_target": 0.15,
                        "error_rates_above_the_band_that_miss_the_target": [0.2, 0.3],
                    },
                },
                "accuracy_ceiling": {"value": 0.99},
            }
        },
        "tests": {
            "primary_contrast": {
                "monte_carlo_draws": 2_000,
                "monte_carlo_draws_rationale": rationale,
            },
            "practical_floors": {"covariance_floor_absolute": 0.002},
        },
    }
    return protocol, results


def test_simulation_resolution_accepts_an_exact_match() -> None:
    protocol, results = matched_protocol_and_results()
    freeze.validate_simulation_resolution(protocol, results)


@pytest.mark.parametrize(
    "field",
    ["min_solo_errors", "min_solo_error_rate", "high_error_rate_finding"],
)
def test_simulation_resolution_rejects_a_threshold_mismatch(field: str) -> None:
    protocol, results = matched_protocol_and_results()
    protocol["solo_screen"]["criteria"]["error_information_threshold"][field] = "wrong"
    with pytest.raises(ValueError, match=field):
        freeze.validate_simulation_resolution(protocol, results)


def test_source_snapshot_copies_and_hashes_frozen_code(tmp_path: Path) -> None:
    package = tmp_path / "package"
    (package / "src" / "step0").mkdir(parents=True)
    (package / "tests").mkdir()
    for rel, content in {
        "README.md": "readme\n",
        "pyproject.toml": "[project]\nname='demo'\n",
        "src/step0/demo.py": "VALUE = 1\n",
        "tests/test_demo.py": "def test_demo(): assert True\n",
    }.items():
        (package / rel).write_text(content, encoding="utf-8")

    hashes = freeze._snapshot_source(package, tmp_path / "out")

    assert set(hashes) == {
        "README.md",
        "pyproject.toml",
        "src/step0/demo.py",
        "tests/test_demo.py",
    }
    for rel in hashes:
        assert (tmp_path / "out" / "source_snapshot" / rel).read_text() == (
            package / rel
        ).read_text()


def test_simulation_results_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run `step0 simulate"):
        freeze._simulation_results(tmp_path)

    (tmp_path / "results.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert freeze._simulation_results(tmp_path) == {"ok": True}


def test_runtime_versions_record_the_rng_and_statistics_packages() -> None:
    versions = freeze._runtime_versions()
    assert set(versions) == set(freeze.RUNTIME_PACKAGES)
    assert versions["numpy"] != "NOT_INSTALLED"
    assert versions["scipy"] != "NOT_INSTALLED"
