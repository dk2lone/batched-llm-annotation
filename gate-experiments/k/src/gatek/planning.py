"""Stage 3: build immutable inference jobs.

Paired jobs share a ``pair_id`` so the analysis can join them. Every pair uses the same
arm, dimension, seed and query row. The nearest demonstration rows are chosen by
informative-space (latent 2D) distance, never by a measured model representation, so the
causal ablation is not selected by its own outcome.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .metrics import euclidean_distances
from .prompts import render_prompt
from .synth import LABEL_LETTERS

JOB_TYPES = (
    "order_random",
    "order_nearest_first",
    "ablate_nearest",
    "ablate_random",
    "attention_capture",
)


@dataclass(frozen=True)
class Job:
    job_id: str
    pair_id: str
    job_type: str
    arm: str
    dim: int
    seed: int
    query_index: int
    demo_order: list[int]
    removed_indices: list[int]
    nearest_indices: list[int]
    capture_attention: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_seed(order_seed: int, key: str, query_index: int) -> int:
    """Process-independent seed. ``hash()`` is salted per process, so it is not usable."""
    digest = hashlib.sha256(f"{order_seed}|{key}|{query_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _job_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _make(
    job_type: str,
    arm: str,
    dim: int,
    seed: int,
    query_index: int,
    demo_order: list[int],
    removed: list[int],
    nearest: list[int],
    capture_attention: bool = False,
) -> Job:
    pair_id = f"{arm}|d{dim}|s{seed}|q{query_index}"
    payload = {
        "pair_id": pair_id,
        "job_type": job_type,
        "demo_order": demo_order,
        "removed": removed,
        "capture_attention": capture_attention,
    }
    return Job(
        job_id=_job_id(payload),
        pair_id=pair_id,
        job_type=job_type,
        arm=arm,
        dim=dim,
        seed=seed,
        query_index=query_index,
        demo_order=demo_order,
        removed_indices=removed,
        nearest_indices=nearest,
        capture_attention=capture_attention,
    )


def build_jobs(cfg: Config, data: dict[str, np.ndarray], keys: list[str]) -> list[Job]:
    """Build every planned job from the frozen synthetic arrays."""
    jobs: list[Job] = []
    for key in sorted(keys):
        arm, dim_s, seed_s = key.split("__")
        dim = int(dim_s[1:])
        seed = int(seed_s[1:])
        demo_z = data[f"latent__s{seed}__demo_z"]
        query_z = data[f"latent__s{seed}__query_z"]
        n_demo = len(demo_z)
        for qi in range(len(query_z)):
            rng = np.random.default_rng(_stable_seed(cfg.plan.order_seed, key, qi))
            latent_d = euclidean_distances(query_z[qi], demo_z)
            nearest_sorted = [int(i) for i in np.argsort(latent_d, kind="stable")]
            base_order = [int(i) for i in rng.permutation(n_demo)]
            jobs.append(
                _make("order_random", arm, dim, seed, qi, base_order, [], nearest_sorted[:1])
            )
            if dim in cfg.plan.ordering_dims:
                jobs.append(
                    _make(
                        "order_nearest_first",
                        arm,
                        dim,
                        seed,
                        qi,
                        nearest_sorted,
                        [],
                        nearest_sorted[:1],
                    )
                )
            if dim in cfg.plan.ablation_dims:
                k = min(cfg.plan.n_ablated_rows, n_demo - 2)
                nearest_k = nearest_sorted[:k]
                kept = [i for i in base_order if i not in set(nearest_k)]
                jobs.append(_make("ablate_nearest", arm, dim, seed, qi, kept, nearest_k, nearest_k))
                pool = (
                    [i for i in base_order if i not in set(nearest_k)]
                    if cfg.plan.random_removal_excludes_nearest
                    else list(base_order)
                )
                removed_random = sorted(
                    int(i) for i in rng.choice(np.array(pool), size=k, replace=False)
                )
                kept_random = [i for i in base_order if i not in set(removed_random)]
                jobs.append(
                    _make(
                        "ablate_random",
                        arm,
                        dim,
                        seed,
                        qi,
                        kept_random,
                        removed_random,
                        nearest_k,
                    )
                )
            if cfg.plan.include_attention_job and dim in cfg.plan.attention_dims:
                jobs.append(
                    _make(
                        "attention_capture",
                        arm,
                        dim,
                        seed,
                        qi,
                        base_order,
                        [],
                        nearest_sorted[:1],
                        capture_attention=True,
                    )
                )
    return jobs


def plan(cfg: Config, out_dir: Path) -> dict[str, Any]:
    """Write ``plan/jobs.jsonl`` and its manifest."""
    synth_dir = out_dir / "synthetic"
    npz = np.load(synth_dir / "data.npz")
    data = {k: npz[k] for k in npz.files}
    keys = sorted(
        {
            rec["key"]
            for rec in (
                json.loads(line)
                for line in (synth_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    )
    jobs = build_jobs(cfg, data, keys)
    ids = [j.job_id for j in jobs]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate job ids; the plan is not immutable")
    plan_dir = out_dir / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = plan_dir / "jobs.jsonl"
    jobs_text = "".join(json.dumps(job.as_dict(), sort_keys=True) + "\n" for job in jobs)
    if jobs_path.exists() and jobs_path.read_text(encoding="utf-8") != jobs_text:
        raise RuntimeError(
            f"{jobs_path} already contains a different frozen plan; use a new output directory"
        )
    jobs_path.write_text(jobs_text, encoding="utf-8")
    jobs_sha256 = hashlib.sha256(jobs_text.encode("utf-8")).hexdigest()
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.job_type] = counts.get(job.job_type, 0) + 1
    manifest = {
        "stage": "plan",
        "created_utc": datetime.now(UTC).isoformat(),
        "n_jobs": len(jobs),
        "jobs_by_type": counts,
        "jobs_jsonl": str(jobs_path),
        "jobs_sha256": jobs_sha256,
        "config_plan": cfg.plan.__dict__,
    }
    (plan_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def verify_plan(out_dir: Path) -> None:
    """Refuse inference or analysis if the frozen job file changed."""
    plan_dir = out_dir / "plan"
    jobs_path = plan_dir / "jobs.jsonl"
    manifest_path = plan_dir / "manifest.json"
    if not jobs_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"no frozen plan under {plan_dir}; run `gatek plan` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(jobs_path.read_bytes()).hexdigest()
    if manifest.get("jobs_sha256") != actual:
        raise RuntimeError(f"{jobs_path} does not match its manifest; use a new output directory")


def load_jobs(out_dir: Path) -> list[Job]:
    verify_plan(out_dir)
    path = out_dir / "plan" / "jobs.jsonl"
    return [
        Job(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tokenizer_preflight(cfg: Config, out_dir: Path, allow_download: bool = False) -> dict[str, Any]:
    """Count every planned prompt with the real tokenizer, without loading the model."""
    from .backends import load_tokenizer  # noqa: PLC0415

    tokenizer = load_tokenizer(
        cfg.run.model_id,
        cfg.run.revision,
        cfg.run.cache_dir,
        local_files_only=not allow_download,
    )
    jobs = load_jobs(out_dir)
    npz = np.load(out_dir / "synthetic" / "data.npz")
    counts: list[tuple[Job, int]] = []
    for job in jobs:
        key = f"{job.arm}__d{job.dim}__s{job.seed}"
        order = np.asarray(job.demo_order, dtype=int)
        demo_x = npz[f"{key}__demo_x"][order]
        demo_y = npz[f"latent__s{job.seed}__demo_y"][order]
        query_x = npz[f"{key}__query_x"][job.query_index]
        body = render_prompt(
            demo_x,
            [LABEL_LETTERS[int(v)] for v in demo_y],
            query_x,
            tuple(f"f{i}" for i in range(job.dim)),
            cfg.synthetic.number_format,
        )
        if getattr(tokenizer, "chat_template", None) is None:
            ids = tokenizer(body.text, add_special_tokens=True)["input_ids"]
        else:
            messages = [{"role": "user", "content": body.text}]
            try:
                ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=cfg.run.enable_thinking,
                )
            except TypeError:
                ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if len(ids) == 1 and isinstance(ids[0], (list, tuple, np.ndarray)):
            ids = ids[0]
        token_count = len(ids)
        if token_count < 8:
            raise RuntimeError(
                f"tokenizer returned an implausible {token_count} tokens for job {job.job_id}"
            )
        counts.append((job, token_count))

    by_dim = {
        str(dim): max(n for job, n in counts if job.dim == dim)
        for dim in sorted({job.dim for job, _ in counts})
    }
    over = [job.job_id for job, n in counts if n > cfg.run.max_prompt_tokens]
    largest_jobs = [
        {"job_id": job.job_id, "tokens": n}
        for job, n in sorted(counts, key=lambda item: item[1], reverse=True)[:20]
    ]
    result = {
        "stage": "tokenizer-preflight",
        "created_utc": datetime.now(UTC).isoformat(),
        "model_id": cfg.run.model_id,
        "requested_revision": cfg.run.revision,
        "resolved_revision": getattr(tokenizer, "init_kwargs", {}).get(
            "_commit_hash", getattr(tokenizer, "_commit_hash", None)
        ),
        "n_jobs": len(counts),
        "max_tokens": max(n for _, n in counts),
        "max_tokens_by_dim": by_dim,
        "largest_jobs": largest_jobs,
        "limit": cfg.run.max_prompt_tokens,
        "n_over_limit": len(over),
        "first_over_limit_job_ids": over[:20],
        "passed": not over,
    }
    path = out_dir / "plan" / "tokenizer_preflight.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
