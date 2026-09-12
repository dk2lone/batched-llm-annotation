"""Job planning: pairing, nearest-row selection and immutability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gatek.config import Config, GateConfig, PlanConfig, SyntheticConfig
from gatek.metrics import euclidean_distances
from gatek.planning import build_jobs, plan, verify_plan
from gatek.synth import build_arm, make_latent

LATENT = make_latent(
    seed=0, n_demo=8, n_query=4, cluster_offset=2.0, cluster_sigma=0.45, margin=0.75
)


def fake_data() -> tuple[dict[str, np.ndarray], list[str]]:
    data: dict[str, np.ndarray] = {
        "latent__s0__demo_z": LATENT.demo_z,
        "latent__s0__demo_y": LATENT.demo_y,
        "latent__s0__query_z": LATENT.query_z,
        "latent__s0__query_y": LATENT.query_y,
    }
    keys = []
    for arm in ("sparse_zero_pad", "dense_redundant_projection", "independent_nuisance"):
        a = build_arm(arm, 8, LATENT, 1.0, "unit_std")
        key = f"{arm}__d8__s0"
        data[f"{key}__demo_x"] = a.demo_x
        data[f"{key}__query_x"] = a.query_x
        keys.append(key)
    return data, keys


CFG = Config(plan=PlanConfig(ablation_dims=[8], n_ablated_rows=2, ordering_dims=[8]))


def test_paired_jobs_share_a_pair_id() -> None:
    data, keys = fake_data()
    jobs = build_jobs(CFG, data, keys)
    by_pair: dict[str, set[str]] = {}
    for job in jobs:
        by_pair.setdefault(job.pair_id, set()).add(job.job_type)
    for types in by_pair.values():
        assert types == {"order_random", "order_nearest_first", "ablate_nearest", "ablate_random"}


def test_nearest_first_order_is_sorted_by_latent_distance() -> None:
    data, keys = fake_data()
    jobs = [j for j in build_jobs(CFG, data, keys) if j.job_type == "order_nearest_first"]
    job = jobs[0]
    d = euclidean_distances(LATENT.query_z[job.query_index], LATENT.demo_z)
    assert job.demo_order == list(np.argsort(d, kind="stable").astype(int))


def test_ablate_nearest_removes_the_latent_nearest_rows() -> None:
    data, keys = fake_data()
    for job in build_jobs(CFG, data, keys):
        if job.job_type != "ablate_nearest":
            continue
        d = euclidean_distances(LATENT.query_z[job.query_index], LATENT.demo_z)
        expected = list(np.argsort(d, kind="stable").astype(int))[:2]
        assert sorted(job.removed_indices) == sorted(expected)
        assert not set(job.removed_indices) & set(job.demo_order)


def test_random_removal_is_matched_in_count_and_avoids_the_nearest_rows() -> None:
    data, keys = fake_data()
    for job in build_jobs(CFG, data, keys):
        if job.job_type != "ablate_random":
            continue
        assert len(job.removed_indices) == 2
        assert not set(job.removed_indices) & set(job.nearest_indices)


def test_ablation_and_base_jobs_keep_the_same_base_order() -> None:
    data, keys = fake_data()
    jobs = {(j.pair_id, j.job_type): j for j in build_jobs(CFG, data, keys)}
    for (pid, jt), job in jobs.items():
        if jt != "ablate_nearest":
            continue
        base = jobs[(pid, "order_random")]
        kept = [i for i in base.demo_order if i not in set(job.removed_indices)]
        assert job.demo_order == kept


def test_job_ids_are_unique_and_deterministic() -> None:
    data, keys = fake_data()
    first = build_jobs(CFG, data, keys)
    second = build_jobs(CFG, data, keys)
    assert [j.job_id for j in first] == [j.job_id for j in second]
    assert len({j.job_id for j in first}) == len(first)


def test_attention_jobs_appear_only_when_enabled() -> None:
    data, keys = fake_data()
    assert not any(j.capture_attention for j in build_jobs(CFG, data, keys))
    cfg = Config(
        plan=PlanConfig(
            ablation_dims=[8], n_ablated_rows=2, include_attention_job=True, attention_dims=[8]
        )
    )
    assert any(j.capture_attention for j in build_jobs(cfg, data, keys))


def test_frozen_plan_detects_tampering(tmp_path: Path) -> None:
    from gatek.prepare import prepare_synthetic

    cfg = Config(
        synthetic=SyntheticConfig(dims=[2, 8], seeds=[0], n_demo=8, n_query=4),
        plan=PlanConfig(ablation_dims=[8], ordering_dims=[8]),
        gate=GateConfig(high_dims=[8]),
    )
    prepare_synthetic(cfg, tmp_path)
    plan(cfg, tmp_path)
    verify_plan(tmp_path)
    jobs_path = tmp_path / "plan" / "jobs.jsonl"
    jobs_path.write_text(jobs_path.read_text() + "{}\n")
    with pytest.raises(RuntimeError, match="does not match"):
        verify_plan(tmp_path)
