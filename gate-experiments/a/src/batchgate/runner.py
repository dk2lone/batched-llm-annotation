"""Stage 3: execute planned jobs and append auditable rows. Resumable."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends import Backend, GenRequest, GenResult, make_backend
from .config import Config
from .data import Record, read_records
from .plan import Job, build_messages, read_plan
from .prompts import parse_labels, prompt_text, target_status
from .util import append_jsonl, read_jsonl, sha256_file, sha256_text


class PreflightError(RuntimeError):
    pass


class PlanIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class Preflight:
    counter: str
    n_jobs: int
    max_tokens_seen: int
    mean_tokens: float
    sum_prompt_tokens: int
    sum_output_budget: int
    over_budget: list[tuple[str, int]]


def load_docs(cfg: Config) -> dict[str, Record]:
    data_dir = cfg.path(cfg.data.out_dir)
    records = read_records(data_dir / "targets.jsonl") + read_records(
        data_dir / "filler_pool.jsonl"
    )
    return {r.doc_id: r for r in records}


def verify_plan_file(cfg: Config) -> None:
    plan_dir = cfg.path(cfg.plan.out_dir)
    meta_path = plan_dir / "plan_meta.json"
    if not meta_path.exists():
        raise PlanIntegrityError(f"{meta_path} is missing. Run `plan` first.")
    import json  # noqa: PLC0415

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual = sha256_file(plan_dir / "plan.jsonl")
    if meta.get("plan_sha256") != actual:
        raise PlanIntegrityError(
            "plan.jsonl does not match plan_meta.json. The manifest changed after it was frozen."
        )


def completed_job_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["job_id"]) for row in read_jsonl(path, tolerate_truncated_tail=True)}


def preflight(
    cfg: Config, jobs: list[Job], docs: dict[str, Record], backend: Backend | None
) -> Preflight:
    budget = cfg.model.max_model_len
    over: list[tuple[str, int]] = []
    total_prompt = 0
    total_output = 0
    largest = 0
    counter_name = "char_estimate"
    for job in jobs:
        messages = build_messages(job, docs)
        counted = backend.count_tokens(messages) if backend is not None else None
        if counted is None:
            counted = int(len(prompt_text(messages)) / cfg.run.chars_per_token) + 1
        else:
            counter_name = getattr(backend, "name", "backend")
        need = counted + job.max_tokens + cfg.run.token_safety_margin
        largest = max(largest, need)
        total_prompt += counted
        total_output += job.max_tokens
        if need > budget:
            over.append((job.job_id, need))
    return Preflight(
        counter=counter_name,
        n_jobs=len(jobs),
        max_tokens_seen=largest,
        mean_tokens=total_prompt / len(jobs) if jobs else 0.0,
        sum_prompt_tokens=total_prompt,
        sum_output_budget=total_output,
        over_budget=over,
    )


def _rows_for(
    cfg: Config,
    job: Job,
    result: GenResult,
    docs: dict[str, Record],
    run_id: str,
    backend_meta: dict[str, Any],
    chunk_size: int,
) -> list[dict[str, Any]]:
    parsed = parse_labels(result.text, list(job.target_ids))
    now = dt.datetime.now(dt.UTC).isoformat()
    model = cfg.model
    rows: list[dict[str, Any]] = []
    for doc_id, slot in zip(job.target_ids, job.target_slots, strict=True):
        status = target_status(parsed, doc_id)
        gold = docs[doc_id].gold
        pred = parsed.labels.get(doc_id) if status == "ok" else None
        rows.append(
            {
                "run_id": run_id,
                "job_id": job.job_id,
                "condition": job.condition,
                "assignment": job.assignment,
                "real_batch_id": job.real_batch_id,
                "doc_id": doc_id,
                "slot": slot,
                "n_prompt_docs": job.n_prompt_docs,
                "n_requested": len(job.target_ids),
                "gold": gold,
                "pred": pred,
                "error": None if pred is None else int(pred != gold),
                "parse_status": status,
                "job_parse_status": parsed.status,
                "n_missing_ids": len(parsed.missing_ids),
                "n_unknown_ids": len(parsed.unknown_ids),
                "n_duplicate_ids": len(parsed.duplicate_ids),
                "parse_notes": list(parsed.notes),
                "prompt_sha256": job.prompt_sha256,
                "output_sha256": sha256_text(result.text),
                "prompt_chars": job.prompt_chars,
                "context_chars": job.match_chars_actual,
                "prompt_tokens": result.prompt_tokens,
                "output_chars": len(result.text),
                "max_tokens": job.max_tokens,
                "model_id": model.model_id,
                "model_revision": backend_meta.get("revision"),
                "backend": backend_meta.get("backend"),
                "dtype": model.dtype,
                "temperature": model.temperature,
                "top_p": model.top_p,
                "top_k": model.top_k,
                "model_seed": model.seed,
                "max_model_len": model.max_model_len,
                "enable_thinking": model.enable_thinking,
                "gen_seconds": result.gen_seconds,
                "chunk_size": chunk_size,
                "created_at": now,
                "raw_text": result.text,
            }
        )
    return rows


def run(
    cfg: Config,
    backend_name: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    progress: bool = True,
    phase: str = "all",
) -> dict[str, Any]:
    verify_plan_file(cfg)
    jobs = read_plan(cfg)
    docs = load_docs(cfg)
    out_path = cfg.path(cfg.run.out_path)

    done = completed_job_ids(out_path)
    if phase not in {"primary", "controls", "all"}:
        raise ValueError(f"unknown run phase {phase!r}")
    allowed = {
        "primary": {"solo", "joint"},
        "controls": {"filler"},
        "all": {"solo", "joint", "filler"},
    }[phase]
    eligible = [j for j in jobs if j.condition in allowed]
    todo = [j for j in eligible if j.job_id not in done]
    if limit is not None:
        todo = todo[:limit]

    chosen = backend_name or cfg.run.backend
    gold = {doc_id: record.gold for doc_id, record in docs.items()}
    backend = None if dry_run and chosen == "vllm" else make_backend(cfg, gold, chosen)

    check = preflight(cfg, todo, docs, backend)
    if check.over_budget:
        worst = ", ".join(f"{job_id}={tokens}" for job_id, tokens in check.over_budget[:5])
        raise PreflightError(
            f"{len(check.over_budget)} prompts exceed model.max_model_len="
            f"{cfg.model.max_model_len} tokens (worst: {worst}). "
            "Lower data.max_chars, lower plan.k, or raise model.max_model_len. Nothing was run."
        )

    summary: dict[str, Any] = {
        "backend": chosen,
        "phase": phase,
        "jobs_total": len(jobs),
        "jobs_eligible": len(eligible),
        "jobs_done_before": len(done),
        "jobs_to_run": len(todo),
        "out_path": str(out_path),
        "preflight": {
            "counter": check.counter,
            "max_prompt_plus_output_tokens": check.max_tokens_seen,
            "mean_prompt_tokens": check.mean_tokens,
            "sum_prompt_tokens": check.sum_prompt_tokens,
            "sum_output_token_budget": check.sum_output_budget,
            "max_model_len": cfg.model.max_model_len,
        },
    }
    if dry_run or not todo:
        summary["ran"] = 0
        return summary

    assert backend is not None
    backend_meta = backend.describe()
    run_id = sha256_text(f"{cfg.fingerprint()}:{chosen}:{dt.datetime.now(dt.UTC).isoformat()}")[:12]
    written = 0
    size = cfg.run.chunk_size
    for start in range(0, len(todo), size):
        chunk = todo[start : start + size]
        requests = [
            GenRequest(
                job_id=job.job_id,
                condition=job.condition,
                assignment=job.assignment,
                real_batch_id=job.real_batch_id,
                target_ids=job.target_ids,
                messages=build_messages(job, docs),
                max_tokens=job.max_tokens,
            )
            for job in chunk
        ]
        results = {r.job_id: r for r in backend.generate(requests)}
        rows: list[dict[str, Any]] = []
        for job in chunk:
            rows.extend(
                _rows_for(cfg, job, results[job.job_id], docs, run_id, backend_meta, len(chunk))
            )
        append_jsonl(out_path, rows)
        written += len(chunk)
        if progress:
            print(f"  ran {written}/{len(todo)} jobs", flush=True)
    summary["ran"] = written
    summary["run_id"] = run_id
    return summary
