"""Gate A reproduction tests. The historical artifact is read, never written."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from step0 import repro_gatea as R

GATE_A = Path(__file__).resolve().parents[2] / "gate-experiments" / "a"


def test_dig_walks_objects_and_lists() -> None:
    payload = {"verdict": {"decision_ci": [0.01, 0.13], "label": "PROCEED"}}
    assert R._dig(payload, ("verdict", "label")) == "PROCEED"
    assert R._dig(payload, ("verdict", "decision_ci", "0")) == 0.01


def test_same_is_exact_for_booleans_and_tolerant_for_floats() -> None:
    assert R._same(True, True)
    assert not R._same(True, 1)
    assert R._same(0.1 + 0.2, 0.3)
    assert not R._same(0.0754, 0.0755)
    assert R._same([1, 2.0], [1.0, 2])
    assert not R._same([1], [1, 2])


def test_report_claims_parse_without_swallowing_the_sentence_stop() -> None:
    text = (
        "Aggregate ICC 0.0754, decision interval [0.0182, 0.1314], largest fake-batch p 0.0010."
    )
    claims = R._report_claims(text)
    assert claims["report_aggregate_icc"] == 0.0754
    assert claims["report_decision_ci_low"] == 0.0182
    assert claims["report_decision_ci_high"] == 0.1314
    assert claims["report_max_permutation_p"] == 0.0010


def test_report_claims_return_nothing_when_the_sentence_is_absent() -> None:
    assert R._report_claims("no numbers here") == {}


@pytest.mark.skipif(not GATE_A.exists(), reason="the Gate A artifact is not in this checkout")
def test_the_preserved_report_still_matches_the_expected_sentence() -> None:
    text = (GATE_A / "artifacts/full/analysis/report.md").read_text(encoding="utf-8")
    claims = R._report_claims(text)
    assert claims, "the preserved report no longer states its key metrics in the expected form"
    preserved = json.loads(
        (GATE_A / "artifacts/full/analysis/analysis.json").read_text(encoding="utf-8")
    )
    assert round(float(R._dig(preserved, R.KEY_METRICS["aggregate_icc"])), 4) == claims[
        "report_aggregate_icc"
    ]


def test_the_rerun_config_must_name_exactly_one_analyze_output(tmp_path: Path) -> None:
    root = tmp_path / "gate"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "full.toml").write_text(
        'out_dir = "artifacts/full/analysis"\n', encoding="utf-8"
    )
    replaced = R._config_with_new_out_dir(root, tmp_path / "rerun")
    assert str(tmp_path / "rerun") in replaced
    (root / "configs" / "full.toml").write_text("nothing\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to guess"):
        R._config_with_new_out_dir(root, tmp_path / "rerun")


def test_write_outputs_records_a_blocker_without_claiming_a_pass(tmp_path: Path) -> None:
    result = {
        "ok": False,
        "gate_a_root": "/nowhere",
        "rerun_dir": "/nowhere/rerun",
        "historical_artifact_unchanged": True,
        "historical_files_hashed": 10,
        "returncode": 1,
        "blocker": "the analyze stage did not run",
        "stderr_tail": "ImportError",
        "comparisons": {},
    }
    _, report = R.write_outputs(tmp_path, result)
    text = report.read_text(encoding="utf-8")
    assert "**FAIL**" in text
    assert "the analyze stage did not run" in text


def test_write_outputs_tabulates_every_compared_metric(tmp_path: Path) -> None:
    result = {
        "ok": True,
        "gate_a_root": "/gate",
        "rerun_dir": "/gate/rerun",
        "historical_artifact_unchanged": True,
        "historical_files_hashed": 10,
        "returncode": 0,
        "n_metrics": 2,
        "n_mismatched": 0,
        "comparisons": {
            "aggregate_icc": {"preserved": 0.0754, "recomputed": 0.0754, "match": True},
            "verdict_label": {"preserved": "PROCEED", "recomputed": "PROCEED", "match": True},
        },
    }
    _, report = R.write_outputs(tmp_path, result)
    text = report.read_text(encoding="utf-8")
    assert "**PASS**" in text
    assert "`aggregate_icc`" in text
    assert "2 of 2 match" in text
