"""Execution staging and stopping-rule plumbing."""

from __future__ import annotations

from pathlib import Path

from batchgate.config import load_config
from batchgate.data import prepare_data
from batchgate.plan import build_plan, write_plan
from batchgate.runner import run


def test_primary_and_control_phases_are_disjoint(tmp_path: Path) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "smoke.toml"
    cfg = load_config(config_path, root=tmp_path)
    prepare_data(cfg, allow_download=False, synthetic=512)
    jobs, meta = build_plan(cfg)
    write_plan(cfg, jobs, meta, force=False)

    primary = run(cfg, backend_name="mock", dry_run=True, phase="primary", progress=False)
    controls = run(cfg, backend_name="mock", dry_run=True, phase="controls", progress=False)

    assert primary["jobs_to_run"] == sum(j.condition in {"solo", "joint"} for j in jobs)
    assert controls["jobs_to_run"] == sum(j.condition == "filler" for j in jobs)
    assert primary["jobs_to_run"] + controls["jobs_to_run"] == len(jobs)
