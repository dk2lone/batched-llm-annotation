"""Build disjoint, pinned CPU row frames for Freeze B without model inference."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from step0.hashing import (
    doc_id_for,
    order_key,
    read_jsonl,
    refuse_overwrite,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)

from .labels import normalize_label

N_SCREEN = 2048
N_CONFIRM = 8192
N_FILLER = 4096
SEED = 20260912


@dataclass(frozen=True)
class Task:
    key: str
    dataset: str
    revision: str
    split: str
    scan_limit: int
    min_chars: int
    max_chars: int
    prefix: str
    config: str | None = None
    language: str | None = None


TASKS = (
    Task("civil_comments", "google/civil_comments", "f2970eb3a55777454c94069077cc8d9b5866312d",
         "validation", 97320, 40, 1000, "cc"),
    Task("ag_news", "fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4",
         "train", 120000, 100, 1000, "ag"),
    Task("multi_nli", "nyu-mll/multi_nli", "da70db2af9d09693783c3320c4249840212ee221",
         "train", 392702, 40, 1000, "mn"),
    Task("imdb", "stanfordnlp/imdb", "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
         "test", 25000, 100, 3000, "im"),
    Task("go_emotions", "google-research-datasets/go_emotions",
         "add492243ff905527e67aeb8b80c082af02207c3", "train", 43410, 20, 1000, "ge",
         config="simplified"),
    Task("multinerd", "Babelscape/multinerd", "2814b78e7af4b5a1f1886fe7ad49632de4d9dd25",
         "train", 350000, 20, 1000, "ne", language="en"),
)


def source_record(task: Task, row: dict[str, Any], source_index: int) -> dict[str, Any] | None:
    if task.language and row["lang"] != task.language:
        return None
    if task.key == "multi_nli":
        premise = " ".join(row["premise"].split())
        hypothesis = " ".join(row["hypothesis"].split())
        text = f"premise: {premise}\nhypothesis: {hypothesis}"
        if row["label"] == -1:
            return None
        reference: Any = row["label"]
    elif task.key == "multinerd":
        tokens = row["tokens"]
        tags = row["ner_tags"]
        if len(tokens) != len(tags) or not tokens:
            return None
        text = " ".join(tokens)
        reference = tags
    else:
        text = " ".join(row["text"].split())
        if task.key == "civil_comments":
            reference = int(float(row["toxicity"]) >= 0.5)
        elif task.key == "go_emotions":
            reference = row["labels"]
        else:
            reference = row["label"]
    if not task.min_chars <= len(text) <= task.max_chars:
        return None
    normalized = normalize_label(
        reference, task.key, n_tokens=len(row["tokens"]) if task.key == "multinerd" else None
    )
    return {
        "doc_id": doc_id_for(task.prefix, text),
        "text": text,
        "reference": reference,
        "reference_normalized": normalized,
        "n_chars": len(text),
        "n_tokens": len(row["tokens"]) if task.key == "multinerd" else None,
        "source_index": source_index,
        "source_id": row.get("id", row.get("pairID")),
    }


def historical_exclusions(root: Path, task: Task) -> set[str]:
    out: set[str] = set()
    if task.key == "civil_comments":
        for name in ("targets.jsonl", "filler_pool.jsonl"):
            path = root / "gate-experiments/a/artifacts/full/data" / name
            if not path.exists():
                raise FileNotFoundError(f"historical exclusion source is missing: {path}")
            out.update(str(row["doc_id"]) for row in read_jsonl(path))
    if task.key in {"civil_comments", "ag_news", "multi_nli"}:
        old_name = {"civil_comments": "civil_comments", "ag_news": "ag_news",
                    "multi_nli": "multi_nli"}[task.key]
        old_dir = root / "study-120h/freeze/20260912T041633Z/data" / old_name
        if not old_dir.exists():
            raise FileNotFoundError(f"sealed Step 0 row source is missing: {old_dir}")
        for name in ("items.jsonl", "filler_pool.jsonl"):
            out.update(str(row["doc_id"]) for row in read_jsonl(old_dir / name))
    return out


def assign_strata(rows: list[dict[str, Any]]) -> None:
    if len(rows) % 16:
        raise ValueError("confirmation frame must divide into 16 equal strata")
    by_length = sorted(range(len(rows)), key=lambda i: (rows[i]["n_chars"], rows[i]["doc_id"]))
    per = len(rows) // 16
    for rank, index in enumerate(by_length):
        rows[index]["length_stratum"] = rank // per


def select_frames(
    candidates: list[dict[str, Any]], excluded: set[str], *, seed: int = SEED
) -> dict[str, list[dict[str, Any]]]:
    seen: set[str] = set()
    eligible: list[dict[str, Any]] = []
    for row in candidates:
        doc_id = row["doc_id"]
        if doc_id in excluded or doc_id in seen:
            continue
        seen.add(doc_id)
        eligible.append(row)
    ordered = sorted(eligible, key=lambda r: (order_key(seed, r["doc_id"]), r["doc_id"]))
    need = N_SCREEN + N_CONFIRM + N_FILLER
    if len(ordered) < need:
        raise ValueError(f"only {len(ordered)} eligible unique rows, need {need}")
    frames = {
        "screen": ordered[:N_SCREEN],
        "confirm": ordered[N_SCREEN:N_SCREEN + N_CONFIRM],
        "filler": ordered[N_SCREEN + N_CONFIRM:need],
    }
    assign_strata(frames["confirm"])
    return frames


def build_task(
    root: Path,
    task: Task,
    out_dir: Path,
    *,
    seed: int = SEED,
    extra_excluded: set[str] | None = None,
) -> dict[str, Any]:
    """Write one task into a new directory; existing frames are never replaced."""
    from datasets import load_dataset  # noqa: PLC0415

    task_dir = out_dir / task.key
    refuse_overwrite(task_dir / "manifest.json")
    excluded = historical_exclusions(root, task) | (extra_excluded or set())
    stream = load_dataset(task.dataset, task.config, split=task.split,
                          revision=task.revision, streaming=True)
    pool: list[dict[str, Any]] = []
    for index, row in enumerate(stream):
        if index >= task.scan_limit:
            break
        item = source_record(task, row, index)
        if item is not None:
            pool.append(item)
    frames = select_frames(pool, excluded, seed=seed)
    files: dict[str, Any] = {}
    for role, rows in frames.items():
        path = task_dir / f"{role}.jsonl"
        write_jsonl(path, rows)
        files[path.name] = {"sha256": sha256_file(path), "rows": len(rows),
                            "row_ids_sha256": sha256_json([r["doc_id"] for r in rows])}
    manifest = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "task": task.__dict__,
        "selection_seed": seed,
        "selection": (
            "deduplicate and exclude, sort by SHA256(seed:doc_id), then contiguous "
            "screen/confirm/filler prefixes; 16 equal confirmation length strata"
        ),
        "historical_excluded_ids": len(excluded),
        "raw_candidates_after_length_filter": len(pool),
        "files": files,
        "text_redistribution": (
            "internal only; publish IDs, hashes, indices and reconstruction recipe "
            "unless rights independently permit text"
        ),
        "reference_limit": "dataset reference labels are not adjudicated human truth",
    }
    write_json(task_dir / "manifest.json", manifest)
    return manifest
