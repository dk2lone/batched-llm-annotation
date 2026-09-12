"""Stage 2: build the immutable job manifest. No model is loaded here."""

from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .data import Record, read_records
from .prompts import Message, build_joint_prompt, build_single_prompt, prompt_text
from .util import read_jsonl, sha256_file, sha256_json, sha256_text, write_json, write_jsonl


class PlanExistsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Job:
    job_id: str
    condition: str
    assignment: int
    real_batch_id: str
    doc_order: tuple[str, ...]
    target_ids: tuple[str, ...]
    target_slots: tuple[int, ...]
    n_prompt_docs: int
    prompt_sha256: str
    prompt_chars: int
    est_tokens: int
    max_tokens: int
    match_chars_target: int = 0
    match_chars_actual: int = 0


def _job_id(
    condition: str,
    assignment: int,
    batch_id: str,
    doc_order: tuple[str, ...],
    target_ids: tuple[str, ...],
) -> str:
    return sha256_json(
        {
            "condition": condition,
            "assignment": assignment,
            "batch": batch_id,
            "docs": list(doc_order),
            "targets": list(target_ids),
        }
    )[:16]


def build_messages(job: Job, docs: dict[str, Record]) -> list[Message]:
    """Rebuild the exact prompt from the frozen data files."""
    pairs = [(doc_id, docs[doc_id].text) for doc_id in job.doc_order]
    if job.condition == "joint":
        return build_joint_prompt(pairs)
    return build_single_prompt(pairs, job.target_ids[0])


def max_tokens_for(cfg: Config, n_requested: int) -> int:
    return cfg.model.max_tokens_base + cfg.model.max_tokens_per_label * n_requested


def estimate_tokens(cfg: Config, chars: int, max_tokens: int) -> int:
    return (
        int(math.ceil(chars / cfg.run.chars_per_token)) + max_tokens + cfg.run.token_safety_margin
    )


def _finish(
    cfg: Config,
    condition: str,
    assignment: int,
    batch_id: str,
    doc_order: list[str],
    targets: list[tuple[str, int]],
    docs: dict[str, Record],
    match_target: int = 0,
) -> Job:
    target_ids = tuple(t[0] for t in targets)
    slots = tuple(t[1] for t in targets)
    order = tuple(doc_order)
    stub = Job(
        job_id="",
        condition=condition,
        assignment=assignment,
        real_batch_id=batch_id,
        doc_order=order,
        target_ids=target_ids,
        target_slots=slots,
        n_prompt_docs=len(order),
        prompt_sha256="",
        prompt_chars=0,
        est_tokens=0,
        max_tokens=0,
    )
    text = prompt_text(build_messages(stub, docs))
    max_tokens = max_tokens_for(cfg, len(target_ids))
    actual = sum(docs[d].n_chars for d in order if d not in set(target_ids))
    return Job(
        job_id=_job_id(condition, assignment, batch_id, order, target_ids),
        condition=condition,
        assignment=assignment,
        real_batch_id=batch_id,
        doc_order=order,
        target_ids=target_ids,
        target_slots=slots,
        n_prompt_docs=len(order),
        prompt_sha256=sha256_text(text),
        prompt_chars=len(text),
        est_tokens=estimate_tokens(cfg, len(text), max_tokens),
        max_tokens=max_tokens,
        match_chars_target=match_target,
        match_chars_actual=actual,
    )


def choose_filler(
    pool: list[Record], want_chars: int, n: int, rng: random.Random, draw: int
) -> list[Record]:
    """Greedy nearest-fit draw without replacement, so the prompt has no duplicates."""
    used: set[int] = set()
    chosen: list[Record] = []
    remaining = want_chars
    size = len(pool)
    for step in range(n):
        ideal = remaining / (n - step)
        candidates = [i for i in rng.sample(range(size), min(draw, size)) if i not in used]
        if not candidates:
            candidates = [i for i in range(size) if i not in used][:draw]
        best = min(candidates, key=lambda i: abs(pool[i].n_chars - ideal))
        used.add(best)
        chosen.append(pool[best])
        remaining -= pool[best].n_chars
    return chosen


def build_plan(cfg: Config) -> tuple[list[Job], dict[str, Any]]:
    data_dir = cfg.path(cfg.data.out_dir)
    targets = read_records(data_dir / "targets.jsonl")
    pool = read_records(data_dir / "filler_pool.jsonl")
    docs: dict[str, Record] = {r.doc_id: r for r in targets + pool}
    if len(docs) != len(targets) + len(pool):
        raise ValueError("targets and filler pool are not disjoint")

    k = cfg.plan.k
    jobs: list[Job] = []
    conditions = set(cfg.plan.conditions)

    # Solo runs once. Its prompt does not depend on the assignment, so repeating it
    # would recompute an identical deterministic call.
    if "solo" in conditions:
        for record in targets:
            jobs.append(
                _finish(cfg, "solo", 0, "none", [record.doc_id], [(record.doc_id, 0)], docs)
            )

    batch_index_by_assignment: dict[int, list[list[Record]]] = {}
    for assignment in range(1, cfg.plan.assignments + 1):
        rng = random.Random(sha256_text(f"{cfg.plan.seed}:assignment:{assignment}"))
        shuffled = list(targets)
        rng.shuffle(shuffled)
        batches = [shuffled[i : i + k] for i in range(0, len(shuffled), k)]
        batch_index_by_assignment[assignment] = batches

        for b_index, batch in enumerate(batches):
            batch_id = f"a{assignment}b{b_index:04d}"
            order = [r.doc_id for r in batch]
            if "joint" in conditions:
                jobs.append(
                    _finish(
                        cfg,
                        "joint",
                        assignment,
                        batch_id,
                        order,
                        [(r.doc_id, slot) for slot, r in enumerate(batch)],
                        docs,
                    )
                )
            if "filler" not in conditions:
                continue
            n_filler_targets = max(1, int(round(len(batch) * cfg.plan.filler_fraction)))
            for slot, record in enumerate(batch):
                if slot >= n_filler_targets:
                    break
                want = sum(r.n_chars for r in batch if r.doc_id != record.doc_id)
                job_rng = random.Random(
                    sha256_text(f"{cfg.plan.seed}:filler:{assignment}:{b_index}:{slot}")
                )
                picks = choose_filler(
                    pool, want, len(batch) - 1, job_rng, cfg.plan.filler_candidate_draw
                )
                filler_order = [p.doc_id for p in picks]
                doc_order = filler_order[:slot] + [record.doc_id] + filler_order[slot:]
                jobs.append(
                    _finish(
                        cfg,
                        "filler",
                        assignment,
                        batch_id,
                        doc_order,
                        [(record.doc_id, slot)],
                        docs,
                        match_target=want,
                    )
                )

    check_plan_invariants(jobs, docs, cfg)
    meta = plan_meta(cfg, jobs, batch_index_by_assignment, len(targets), len(pool))
    return jobs, meta


def check_plan_invariants(jobs: list[Job], docs: dict[str, Record], cfg: Config) -> None:
    seen_ids: set[str] = set()
    for job in jobs:
        if len(set(job.doc_order)) != len(job.doc_order):
            raise ValueError(f"job {job.job_id} repeats a document inside one prompt")
        if job.job_id in seen_ids:
            raise ValueError(f"duplicate job id {job.job_id}")
        seen_ids.add(job.job_id)
        for doc_id in job.doc_order:
            if doc_id not in docs:
                raise ValueError(f"job {job.job_id} references unknown document {doc_id}")
        for target, slot in zip(job.target_ids, job.target_slots, strict=True):
            if job.doc_order[slot] != target:
                raise ValueError(f"job {job.job_id} lost the slot of target {target}")
    by_condition: dict[str, set[str]] = {}
    for job in jobs:
        by_condition.setdefault(job.condition, set()).update(job.target_ids)
    if "joint" in by_condition and "solo" in by_condition:
        if by_condition["joint"] != by_condition["solo"]:
            raise ValueError("solo and joint do not cover the same target documents")
    if "filler" in by_condition and cfg.plan.filler_fraction == 1.0:
        if by_condition["filler"] != by_condition["joint"]:
            raise ValueError("filler and joint do not cover the same target documents")


def plan_meta(
    cfg: Config,
    jobs: list[Job],
    batches: dict[int, list[list[Record]]],
    n_targets: int,
    n_pool: int,
) -> dict[str, Any]:
    by_condition: dict[str, int] = {}
    targets_by_condition: dict[str, set[str]] = {}
    for job in jobs:
        by_condition[job.condition] = by_condition.get(job.condition, 0) + 1
        targets_by_condition.setdefault(job.condition, set()).update(job.target_ids)
    filler = [j for j in jobs if j.condition == "filler"]
    deltas = [
        abs(j.match_chars_actual - j.match_chars_target) / max(1, j.match_chars_target)
        for j in filler
    ]
    est = [j.est_tokens for j in jobs]
    return {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "config_fingerprint": cfg.fingerprint(),
        "plan_seed": cfg.plan.seed,
        "k": cfg.plan.k,
        "assignments": cfg.plan.assignments,
        "n_targets": n_targets,
        "n_filler_pool": n_pool,
        "jobs_total": len(jobs),
        "jobs_by_condition": by_condition,
        "targets_by_condition": {c: len(v) for c, v in targets_by_condition.items()},
        "batches_per_assignment": {a: len(b) for a, b in batches.items()},
        "batch_sizes": sorted({len(b) for bs in batches.values() for b in bs}),
        "filler_length_match": {
            "mean_abs_relative_delta": sum(deltas) / len(deltas) if deltas else None,
            "max_abs_relative_delta": max(deltas) if deltas else None,
            "within_tolerance": (
                sum(d <= cfg.plan.length_match_tolerance for d in deltas) / len(deltas)
                if deltas
                else None
            ),
            "tolerance": cfg.plan.length_match_tolerance,
        },
        "token_preflight": {
            "counter": "char_estimate",
            "chars_per_token": cfg.run.chars_per_token,
            "safety_margin": cfg.run.token_safety_margin,
            "max_model_len": cfg.model.max_model_len,
            "max_est_tokens": max(est) if est else 0,
            "mean_est_tokens": sum(est) / len(est) if est else 0,
            "sum_est_tokens": sum(est),
            "jobs_over_budget": sum(1 for e in est if e > cfg.model.max_model_len),
        },
    }


def write_plan(cfg: Config, jobs: list[Job], meta: dict[str, Any], force: bool) -> Path:
    out_dir = cfg.path(cfg.plan.out_dir)
    plan_path = out_dir / "plan.jsonl"
    if plan_path.exists() and not force:
        raise PlanExistsError(
            f"{plan_path} exists. The plan is immutable. Pass --force to rebuild."
        )
    write_jsonl(plan_path, (asdict(j) for j in jobs))
    meta = dict(meta)
    meta["plan_sha256"] = sha256_file(plan_path)
    write_json(out_dir / "plan_meta.json", meta)
    return plan_path


def read_plan(cfg: Config) -> list[Job]:
    plan_path = cfg.path(cfg.plan.out_dir) / "plan.jsonl"
    jobs: list[Job] = []
    for row in read_jsonl(plan_path):
        row["doc_order"] = tuple(row["doc_order"])
        row["target_ids"] = tuple(row["target_ids"])
        row["target_slots"] = tuple(row["target_slots"])
        jobs.append(Job(**row))
    return jobs
