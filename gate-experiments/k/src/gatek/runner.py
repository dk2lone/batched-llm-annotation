"""Stage 4: run planned jobs and append raw results.

The stage is resumable. Job ids already present in ``run/results.jsonl`` are skipped.
Full prompts are stored only when ``run.store_full_prompts`` is true; otherwise the record
keeps the prompt hash and the span metadata.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .backends import BackendResult, MockBackend, TransformersBackend
from .config import Config
from .metrics import (
    cosine_distances,
    distance_stats,
    euclidean_distances,
    spearman_rho,
)
from .planning import Job, load_jobs
from .synth import LABEL_LETTERS


class Backend(Protocol):
    name: str

    def infer(
        self,
        demo_x: np.ndarray,
        demo_letters: list[str],
        query_x: np.ndarray,
        names: tuple[str, ...],
        capture_attention: bool = False,
    ) -> BackendResult: ...


def make_backend(cfg: Config) -> Backend:
    if cfg.run.backend == "mock":
        return MockBackend(
            seed=cfg.run.mock_seed,
            repr_dim=cfg.run.mock_repr_dim,
            number_format=cfg.synthetic.number_format,
        )
    if cfg.run.backend == "transformers":
        return TransformersBackend(
            model_id=cfg.run.model_id,
            revision=cfg.run.revision,
            dtype=cfg.run.dtype,
            device_map=cfg.run.device_map,
            attn_implementation=cfg.run.attn_implementation,
            hidden_layers=cfg.run.hidden_layers,
            label_tokens=cfg.run.label_tokens,
            max_prompt_tokens=cfg.run.max_prompt_tokens,
            number_format=cfg.synthetic.number_format,
            enable_thinking=cfg.run.enable_thinking,
            cache_dir=cfg.run.cache_dir,
        )
    raise ValueError(f"unknown backend: {cfg.run.backend}")


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(str(json.loads(line)["job_id"]))
    return out


def _record(
    job: Job,
    cfg: Config,
    result: BackendResult,
    demo_letters: list[str],
    gold_letter: str,
    latent_d: np.ndarray,
    observed_d: np.ndarray,
    elapsed: float,
) -> dict[str, Any]:
    repr_cos = cosine_distances(result.query_rep, result.demo_reps)
    repr_euc = euclidean_distances(result.query_rep, result.demo_reps)
    primary = repr_cos if cfg.run.repr_distance == "cosine" else repr_euc
    attn_rho = None
    if result.attention_row_mass is not None:
        attn_rho = spearman_rho(np.array(result.attention_row_mass), latent_d)
    return {
        "job_id": job.job_id,
        "pair_id": job.pair_id,
        "job_type": job.job_type,
        "arm": job.arm,
        "dim": job.dim,
        "seed": job.seed,
        "query_index": job.query_index,
        "n_demo_used": len(job.demo_order),
        "removed_indices": job.removed_indices,
        "nearest_indices": job.nearest_indices,
        "gold_letter": gold_letter,
        "pred_letter": result.letter,
        "correct": int(result.letter == gold_letter),
        "parse_status": result.parse_status,
        "label_scores": result.label_scores,
        "margin": result.margin,
        "prediction_method": result.prediction_method,
        "prompt_sha256": result.prompt_sha256,
        "n_prompt_tokens": result.n_prompt_tokens,
        "token_source": result.token_source,
        "latent_stats": distance_stats(latent_d).as_dict(),
        "observed_stats": distance_stats(observed_d).as_dict(),
        "repr_stats_cosine": distance_stats(repr_cos).as_dict(),
        "repr_stats_euclidean": distance_stats(repr_euc).as_dict(),
        "repr_distance_primary": cfg.run.repr_distance,
        "spearman_latent_vs_repr": spearman_rho(latent_d, primary),
        "spearman_observed_vs_repr": spearman_rho(observed_d, primary),
        "repr_nearest_latent_rank": int(
            np.argsort(np.argsort(latent_d, kind="stable"), kind="stable")[int(np.argmin(primary))]
        ),
        "attention_row_mass": result.attention_row_mass,
        "attention_spearman_vs_latent": attn_rho,
        "demo_letters": demo_letters if cfg.run.store_full_prompts else None,
        "backend": result.meta.get("backend", cfg.run.backend),
        "model_id": cfg.run.model_id if cfg.run.backend == "transformers" else "mock",
        "hidden_layers": cfg.run.hidden_layers,
        "meta": result.meta,
        "elapsed_s": elapsed,
    }


def execute_job(
    job: Job,
    cfg: Config,
    backend: Backend,
    npz: Any,
) -> dict[str, Any]:
    """Execute one frozen job with an already-loaded backend.

    Keeping model construction outside this function lets remote workers reuse one
    model across many Modal inputs while preserving the local runner's computation.
    """
    key = f"{job.arm}__d{job.dim}__s{job.seed}"
    demo_x_all = npz[f"{key}__demo_x"]
    query_x_all = npz[f"{key}__query_x"]
    demo_y_all = npz[f"latent__s{job.seed}__demo_y"]
    query_y_all = npz[f"latent__s{job.seed}__query_y"]
    demo_z_all = npz[f"latent__s{job.seed}__demo_z"]
    query_z_all = npz[f"latent__s{job.seed}__query_z"]

    order = np.array(job.demo_order, dtype=int)
    demo_x = demo_x_all[order]
    demo_letters = [LABEL_LETTERS[int(v)] for v in demo_y_all[order]]
    query_x = query_x_all[job.query_index]
    gold_letter = LABEL_LETTERS[int(query_y_all[job.query_index])]
    latent_d = euclidean_distances(query_z_all[job.query_index], demo_z_all[order])
    observed_d = euclidean_distances(query_x, demo_x)

    t0 = time.time()
    result = backend.infer(
        demo_x,
        demo_letters,
        query_x,
        tuple(f"f{i}" for i in range(job.dim)),
        capture_attention=job.capture_attention,
    )
    return _record(
        job,
        cfg,
        result,
        demo_letters,
        gold_letter,
        latent_d,
        observed_d,
        time.time() - t0,
    )


def run(
    cfg: Config,
    out_dir: Path,
    limit: int | None = None,
    phase: str = "all",
) -> dict[str, Any]:
    """Run every planned job that is not already recorded."""
    jobs = load_jobs(out_dir)
    run_dir = out_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    done_all = _done_ids(results_path)
    planned_ids = {job.job_id for job in jobs}
    stale = done_all - planned_ids
    if stale:
        raise RuntimeError(
            f"{results_path} contains {len(stale)} job ids outside the frozen plan; "
            "use a new output directory"
        )
    if phase not in {"anchor", "all"}:
        raise ValueError(f"unknown run phase {phase!r}")
    eligible = (
        [
            j
            for j in jobs
            if j.arm == "sparse_zero_pad" and j.dim == 2 and j.job_type == "order_random"
        ]
        if phase == "anchor"
        else jobs
    )
    done = done_all & {job.job_id for job in eligible}
    todo = [j for j in eligible if j.job_id not in done]
    if limit is not None:
        todo = todo[:limit]

    npz = np.load(out_dir / "synthetic" / "data.npz")
    backend = make_backend(cfg)
    n_written = 0
    errors: list[dict[str, str]] = []
    started = time.time()
    with results_path.open("a", encoding="utf-8") as fh:
        for job in todo:
            try:
                rec = execute_job(job, cfg, backend, npz)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                errors.append({"job_id": job.job_id, "error": f"{type(exc).__name__}: {exc}"})
                continue
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            n_written += 1

    summary = {
        "stage": "run",
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": cfg.run.backend,
        "phase": phase,
        "model_id": cfg.run.model_id if cfg.run.backend == "transformers" else "mock",
        "n_planned": len(jobs),
        "n_eligible": len(eligible),
        "n_already_done": len(done),
        "n_written": n_written,
        "n_errors": len(errors),
        "errors": errors[:50],
        "wall_seconds": time.time() - started,
        "results_jsonl": str(results_path),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
