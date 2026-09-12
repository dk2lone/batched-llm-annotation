"""Stage 1: freeze the target set and the disjoint filler pool."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .util import doc_id_for, sha256_file, sha256_text, write_json, write_jsonl


class DownloadNotAllowedError(RuntimeError):
    pass


@dataclass(frozen=True)
class Record:
    doc_id: str
    text: str
    score: float
    gold: int
    n_chars: int
    source_index: int


def _clean(text: str) -> str:
    return " ".join(text.split())


def _order_key(seed: int, doc_id: str) -> str:
    """Seeded hash order. Reproducible across numpy and Python versions."""
    return sha256_text(f"{seed}:{doc_id}")


def candidates_from_rows(
    rows: Iterable[dict[str, Any]],
    cfg: Config,
) -> list[Record]:
    """Filter, deduplicate and score raw dataset rows. Order-independent by design."""
    data = cfg.data
    seen: set[str] = set()
    out: list[Record] = []
    for index, row in enumerate(rows):
        if data.scan_limit is not None and index >= data.scan_limit:
            break
        text = _clean(str(row[data.text_field]))
        if not data.min_chars <= len(text) <= data.max_chars:
            continue
        doc_id = doc_id_for(text)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        score = float(row[data.label_field])
        out.append(
            Record(
                doc_id=doc_id,
                text=text,
                score=score,
                gold=int(score >= data.toxicity_threshold),
                n_chars=len(text),
                source_index=index,
            )
        )
    return out


def select(records: list[Record], cfg: Config) -> tuple[list[Record], list[Record]]:
    """Pick the frozen targets and a disjoint filler pool.

    Selection rule, in full:
      1. keep rows whose whitespace-normalised text length is in [min_chars, max_chars];
      2. drop duplicate texts, keeping the first;
      3. sort every survivor by sha256("<seed>:<doc_id>");
      4. natural mode takes the first n_targets in that order;
         balanced mode takes the first n_targets/2 of each gold class in that order;
      5. the filler pool takes the next filler_pool_size rows not already used.
    """
    data = cfg.data
    ordered = sorted(records, key=lambda r: _order_key(data.seed, r.doc_id))
    if data.sampling == "natural":
        targets = ordered[: data.n_targets]
    else:
        half = data.n_targets // 2
        positives = [r for r in ordered if r.gold == 1][:half]
        negatives = [r for r in ordered if r.gold == 0][: data.n_targets - half]
        if len(positives) < half or len(negatives) < data.n_targets - half:
            raise ValueError(
                "balanced sampling needs more rows of one class than the candidate pool holds"
            )
        targets = sorted(positives + negatives, key=lambda r: _order_key(data.seed, r.doc_id))
    used = {r.doc_id for r in targets}
    pool = [r for r in ordered if r.doc_id not in used][: data.filler_pool_size]
    if len(targets) < data.n_targets or len(pool) < data.filler_pool_size:
        raise ValueError(
            f"candidate pool too small: got {len(targets)} targets and {len(pool)} filler rows"
        )
    return targets, pool


def prevalence(records: list[Record]) -> float:
    return sum(r.gold for r in records) / len(records) if records else 0.0


def load_hf_rows(cfg: Config) -> Iterator[dict[str, Any]]:
    """Stream the dataset. Imported lazily so no stage except prepare-data needs it."""
    from datasets import load_dataset  # noqa: PLC0415

    dataset = load_dataset(
        cfg.data.dataset,
        split=cfg.data.split,
        revision=cfg.data.revision,
        streaming=True,
    )
    yield from dataset


def resolve_revision(cfg: Config) -> str | None:
    """Ask the Hub for the dataset commit sha. Returns None when it is unavailable."""
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415

        info = HfApi().dataset_info(cfg.data.dataset, revision=cfg.data.revision)
        return str(info.sha)
    except Exception:  # noqa: BLE001 - provenance is best effort, never fatal
        return None


def synthetic_rows(n: int, seed: int) -> list[dict[str, Any]]:
    """Local stand-in records for the smoke test. No network, no real comments."""
    import random  # noqa: PLC0415

    rng = random.Random(seed)
    topics = ["traffic", "school board", "the new bridge", "the budget", "the game"]
    mild = ["I disagree with this.", "Please explain the numbers.", "Good point overall."]
    harsh = ["This is a stupid take.", "Only an idiot would write that.", "Get lost already."]
    rows: list[dict[str, Any]] = []
    for i in range(n):
        toxic = rng.random() < 0.2
        body = rng.choice(harsh if toxic else mild)
        filler_words = " ".join(rng.choice(topics) for _ in range(rng.randint(3, 30)))
        rows.append(
            {
                "text": f"[{i:05d}] {body} Regarding {filler_words}.",
                "toxicity": 0.8 if toxic else 0.05,
            }
        )
    return rows


def write_records(path: Path, records: list[Record]) -> int:
    return write_jsonl(path, (asdict(r) for r in records))


def read_records(path: Path) -> list[Record]:
    from .util import read_jsonl  # noqa: PLC0415

    return [Record(**row) for row in read_jsonl(path)]


def prepare_data(
    cfg: Config,
    allow_download: bool,
    synthetic: int | None = None,
) -> dict[str, Any]:
    """Write targets.jsonl, filler_pool.jsonl and provenance.json."""
    out_dir = cfg.path(cfg.data.out_dir)
    if synthetic is not None:
        source = "synthetic"
        revision = None
        rows: Iterable[dict[str, Any]] = synthetic_rows(synthetic, cfg.data.seed)
    else:
        if not allow_download:
            raise DownloadNotAllowedError(
                "prepare-data downloads "
                f"{cfg.data.dataset}. Re-run with --allow-download, or use --synthetic N."
            )
        source = cfg.data.dataset
        revision = resolve_revision(cfg)
        rows = load_hf_rows(cfg)

    candidates = candidates_from_rows(rows, cfg)
    targets, pool = select(candidates, cfg)

    targets_path = out_dir / "targets.jsonl"
    pool_path = out_dir / "filler_pool.jsonl"
    write_records(targets_path, targets)
    write_records(pool_path, pool)

    provenance: dict[str, Any] = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": source,
        "dataset": cfg.data.dataset,
        "dataset_revision": revision,
        "split": cfg.data.split,
        "sampling": cfg.data.sampling,
        "sampling_seed": cfg.data.seed,
        "toxicity_threshold": cfg.data.toxicity_threshold,
        "min_chars": cfg.data.min_chars,
        "max_chars": cfg.data.max_chars,
        "scan_limit": cfg.data.scan_limit,
        "selection_rule": (select.__doc__ or "").strip(),
        "n_candidates": len(candidates),
        "n_targets": len(targets),
        "n_filler_pool": len(pool),
        "candidate_prevalence": prevalence(candidates),
        "target_prevalence": prevalence(targets),
        "filler_pool_prevalence": prevalence(pool),
        "target_mean_chars": sum(r.n_chars for r in targets) / len(targets),
        "filler_pool_mean_chars": sum(r.n_chars for r in pool) / len(pool),
        "disjoint": not ({r.doc_id for r in targets} & {r.doc_id for r in pool}),
        "files": {
            "targets.jsonl": {
                "sha256": sha256_file(targets_path),
                "rows": len(targets),
                "row_ids_sha256": sha256_text("\n".join(r.doc_id for r in targets)),
            },
            "filler_pool.jsonl": {
                "sha256": sha256_file(pool_path),
                "rows": len(pool),
                "row_ids_sha256": sha256_text("\n".join(r.doc_id for r in pool)),
            },
        },
        "config_fingerprint": cfg.fingerprint(),
    }
    write_json(out_dir / "provenance.json", provenance)
    return provenance
