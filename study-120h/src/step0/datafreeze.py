"""CPU dataset freeze: pin revisions, audit licenses, exclude pilot ids, select frames.

No GPU, no model, no inference. Live retrieval is required. If retrieval cannot
complete, the exact blocker is written and Step 0 stays incomplete. Nothing here
fabricates a row.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .hashing import (
    doc_id_for,
    order_key,
    read_jsonl,
    refuse_overwrite,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from .lock import NO_NEW_LABELS, TASKS

HF_API = "https://huggingface.co/api/datasets/"
HF_MODEL_API = "https://huggingface.co/api/models/"
PYPI_API = "https://pypi.org/pypi/"
N_STRATA = 16
ACCEPTABLE_LICENSES = {"cc0-1.0", "apache-2.0", "mit", "cc-by-4.0", "cc-by-sa-4.0"}


class RetrievalBlockedError(RuntimeError):
    """Live retrieval failed. Step 0 stays incomplete; no row is fabricated."""


@dataclass(frozen=True)
class Item:
    doc_id: str
    text: str
    reference: int
    raw_label: float
    n_chars: int
    source_index: int
    length_stratum: int = -1


# --------------------------------------------------------------------------------------
# Remote metadata and license audit
# --------------------------------------------------------------------------------------


def fetch_json(url: str, timeout: float = 60.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RetrievalBlockedError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc
    return payload


def audit_model_pins(models: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    """Record the live access facts of every pinned model repo. Never assume them."""
    rows: dict[str, Any] = {}
    blockers: list[str] = []
    for model_id, spec in models.items():
        url = HF_MODEL_API + model_id
        info = fetch_json(url, timeout)
        card = info.get("cardData") or {}
        gated = info.get("gated")
        row = {
            "model_id": model_id,
            "evidence_url": url,
            "retrieved_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "declared_revision": spec["revision"],
            "live_main_sha": str(info.get("sha")),
            "revision_matches_live_main": spec["revision"] == str(info.get("sha")),
            "declared_license_in_protocol": spec["license"],
            "live_card_license": card.get("license"),
            "gated": gated,
            "gated_is_false": gated is False,
            "private": bool(info.get("private")),
            "disabled": bool(info.get("disabled")),
            "last_modified": info.get("lastModified"),
            "redistribution_of_weights": "not claimed and not performed by this study",
        }
        if gated is not False:
            blockers.append(f"{model_id} reports gated={gated!r}; a human must accept the terms")
        if row["private"] or row["disabled"]:
            blockers.append(f"{model_id} is private or disabled")
        if not row["revision_matches_live_main"]:
            blockers.append(
                f"{model_id} pinned revision {spec['revision']} is not the live main sha "
                f"{row['live_main_sha']}; confirm the pin is deliberate"
            )
        rows[model_id] = row
    return {
        "models": rows,
        "blockers": blockers,
        "ok": not blockers,
        "rule": (
            "no model repo is described as ungated, public, or usable without the live record "
            "above. A gated, private, or disabled repo blocks Step 1."
        ),
    }


def audit_engine_pin(serving: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    """Confirm the predeclared serving engine version exists and record its artifacts."""
    pin = str(serving["engine_version_pin"])
    name, _, version = pin.partition("==")
    url = f"{PYPI_API}{name}/{version}/json"
    info = fetch_json(url, timeout)
    files = [
        {
            "filename": entry["filename"],
            "packagetype": entry["packagetype"],
            "upload_time": entry["upload_time"],
            "sha256": entry["digests"]["sha256"],
        }
        for entry in info.get("urls", [])
    ]
    wanted = str(serving["engine_wheel_pin"])
    match = next((f for f in files if f["filename"] == wanted), None)
    return {
        "engine": serving["engine"],
        "engine_version_pin": pin,
        "evidence_url": url,
        "retrieved_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "version_exists_on_pypi": info.get("info", {}).get("version") == version,
        "requires_python": info.get("info", {}).get("requires_python"),
        "release_files": files,
        "pinned_wheel": wanted,
        "pinned_wheel_found": match is not None,
        "pinned_wheel_sha256": None if match is None else match["sha256"],
        "pinned_wheel_upload_time": None if match is None else match["upload_time"],
        "ok": bool(info.get("info", {}).get("version") == version and match is not None),
        "rule": (
            "Step 1 installs exactly this version and records the installed version string. "
            "CUDA, driver, GPU, and torch facts are host facts recorded at Step 1."
        ),
    }


def fetch_dataset_info(dataset: str, timeout: float = 60.0) -> dict[str, Any]:
    url = HF_API + dataset
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RetrievalBlockedError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc
    return payload


def audit_license(spec: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    """Compare the declared license and revision against the live card. Never guess."""
    card = info.get("cardData") or {}
    live_license = card.get("license")
    if isinstance(live_license, list):
        live_license_text = ", ".join(str(x) for x in live_license)
    else:
        live_license_text = str(live_license)
    live_sha = str(info.get("sha"))
    declared = str(spec["revision"])
    return {
        "dataset": spec["dataset"],
        "declared_revision": declared,
        "live_main_sha": live_sha,
        "revision_pinned_matches_live_main": declared == live_sha,
        "declared_license_in_protocol": spec["license"],
        "live_card_license": live_license_text,
        "license_is_in_permissive_allowlist": live_license_text in ACCEPTABLE_LICENSES,
        "redistribution_permitted_by_protocol": spec["redistribution_permitted"],
        "gated": bool(info.get("gated")),
        "disabled": bool(info.get("disabled")),
        "last_modified": info.get("lastModified"),
        "labelling_limits": spec["labelling_limits"],
        "reference_label_rule": spec["reference_label_rule"],
        "audited_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "audit_url": HF_API + spec["dataset"],
    }


def license_blockers(audit: dict[str, Any]) -> list[str]:
    """Hard blockers. An unknown license is a recorded restriction, not a blocker."""
    out: list[str] = []
    if audit["gated"]:
        out.append("the dataset repo is gated; access terms must be accepted by a human first")
    if audit["disabled"]:
        out.append("the dataset repo is disabled")
    if not audit["revision_pinned_matches_live_main"]:
        out.append(
            "the pinned revision is not the live main sha. This is allowed only if the pin was "
            "deliberate. Confirm the pin before continuing."
        )
    return out


def license_restrictions(spec: dict[str, Any], audit: dict[str, Any]) -> list[str]:
    """Recorded limits that do not stop the work but must never be overstated."""
    out: list[str] = []
    if not spec["redistribution_permitted"]:
        out.append(
            "no redistribution right is established for this corpus. The release publishes "
            "document ids, row indices, character counts, and SHA-256 digests only, never text."
        )
    if not audit["license_is_in_permissive_allowlist"]:
        out.append(
            f"the live card license {audit['live_card_license']!r} is not in the permissive "
            "allowlist, so no redistribution or relicensing claim is made from it."
        )
    return out


# --------------------------------------------------------------------------------------
# Row loading
# --------------------------------------------------------------------------------------


def load_rows(spec: dict[str, Any], timeout: float = 600.0) -> Iterator[dict[str, Any]]:
    del timeout
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise RetrievalBlockedError(f"the `datasets` package is unavailable: {exc}") from exc
    try:
        stream = load_dataset(
            spec["dataset"], split=spec["split"], revision=spec["revision"], streaming=True
        )
        yield from stream
    except Exception as exc:  # noqa: BLE001 - any failure must be recorded verbatim
        raise RetrievalBlockedError(
            f"load_dataset({spec['dataset']!r}, split={spec['split']!r}, "
            f"revision={spec['revision']!r}) failed: {type(exc).__name__}: {exc}"
        ) from exc


def _clean(text: str) -> str:
    return " ".join(text.split())


def item_text(spec: dict[str, Any], row: dict[str, Any]) -> str:
    """One frozen document string per source row.

    A single-field task uses that field. A pair task, such as MultiNLI, joins its
    fields with the frozen template, so the document id is stable and the prompt sees
    exactly what the freeze recorded.
    """
    template = spec.get("text_template")
    if template:
        return str(
            template.format(**{field: _clean(str(row[field])) for field in spec["text_fields"]})
        )
    return _clean(str(row[spec["text_field"]]))


def _reference_for(spec: dict[str, Any], raw: Any) -> tuple[int, float]:
    if spec["dataset"] == "google/civil_comments":
        score = float(raw)
        return int(score >= 0.5), score
    return int(raw), float(raw)


def candidates(
    rows: Iterable[dict[str, Any]], spec: dict[str, Any], excluded: set[str]
) -> list[Item]:
    """Filter, deduplicate, and exclude. Order independent by construction."""
    seen: set[str] = set()
    out: list[Item] = []
    limit = int(spec["scan_limit"])
    for index, row in enumerate(rows):
        if index >= limit:
            break
        text = item_text(spec, row)
        if not spec["min_chars"] <= len(text) <= spec["max_chars"]:
            continue
        doc_id = doc_id_for(spec["prefix"], text)
        if doc_id in seen or doc_id in excluded:
            continue
        seen.add(doc_id)
        reference, raw = _reference_for(spec, row[spec["label_field"]])
        out.append(
            Item(
                doc_id=doc_id,
                text=text,
                reference=reference,
                raw_label=raw,
                n_chars=len(text),
                source_index=index,
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------


def select(pool: list[Item], spec: dict[str, Any]) -> tuple[list[Item], list[Item]]:
    """Select the frozen item frame and a disjoint filler pool.

    Rule, in full:
      1. sort every candidate by sha256("<selection_seed>:<doc_id>");
      2. natural sampling takes the first n_items in that order;
         stratified sampling takes the first n_items/n_classes of each reference class;
      3. the filler pool takes the next filler_pool_size candidates not already used;
      4. selected items are then sorted by character count and cut into 16 length strata
         of equal size.
    """
    seed = int(spec["selection_seed"])
    ordered = sorted(pool, key=lambda r: order_key(seed, r.doc_id))
    n_items = int(spec["n_items"])
    if spec["sampling"].startswith("stratified"):
        n_classes = int(spec["n_classes"])
        per_class, remainder = divmod(n_items, n_classes)
        if remainder:
            raise ValueError(f"n_items {n_items} is not divisible by n_classes {n_classes}")
        chosen: list[Item] = []
        for klass in range(n_classes):
            members = [r for r in ordered if r.reference == klass][:per_class]
            if len(members) < per_class:
                raise ValueError(
                    f"class {klass} has only {len(members)} candidates, needs {per_class}"
                )
            chosen.extend(members)
        items = sorted(chosen, key=lambda r: order_key(seed, r.doc_id))
    else:
        items = ordered[:n_items]
    if len(items) < n_items:
        raise ValueError(f"candidate pool too small: {len(items)} of {n_items} items")

    used = {r.doc_id for r in items}
    filler = [r for r in ordered if r.doc_id not in used][: int(spec["filler_pool_size"])]
    if len(filler) < int(spec["filler_pool_size"]):
        raise ValueError(
            f"filler pool too small: {len(filler)} of {spec['filler_pool_size']} rows"
        )
    return assign_strata(items), filler


def assign_strata(items: list[Item], n_strata: int = N_STRATA) -> list[Item]:
    """Cut the items into equal length strata, then restore the selection order."""
    if len(items) % n_strata:
        raise ValueError(f"{len(items)} items do not divide into {n_strata} strata")
    per = len(items) // n_strata
    by_length = sorted(range(len(items)), key=lambda i: (items[i].n_chars, items[i].doc_id))
    stratum_of = {idx: rank // per for rank, idx in enumerate(by_length)}
    return [
        Item(**{**asdict(item), "length_stratum": stratum_of[i]}) for i, item in enumerate(items)
    ]


# --------------------------------------------------------------------------------------
# Pilot exclusions
# --------------------------------------------------------------------------------------


def pilot_excluded_ids(gate_a_root: Path, spec: dict[str, Any]) -> tuple[set[str], dict[str, int]]:
    out: set[str] = set()
    counts: dict[str, int] = {}
    for rel in spec.get("excluded_id_sources", []):
        path = gate_a_root.parent.parent / rel if not Path(rel).is_absolute() else Path(rel)
        if not path.exists():
            raise RetrievalBlockedError(f"pilot exclusion file {path} is missing")
        ids = {str(row["doc_id"]) for row in read_jsonl(path)}
        counts[Path(rel).name] = len(ids)
        out |= ids
    return out, counts


# --------------------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------------------


def _describe(items: list[Item], spec: dict[str, Any]) -> dict[str, Any]:
    n = len(items)
    classes: dict[str, int] = {}
    for item in items:
        classes[str(item.reference)] = classes.get(str(item.reference), 0) + 1
    strata: dict[str, dict[str, float]] = {}
    for s in range(N_STRATA):
        members = [i.n_chars for i in items if i.length_stratum == s]
        if members:
            strata[str(s)] = {
                "n": len(members),
                "min_chars": min(members),
                "max_chars": max(members),
                "mean_chars": sum(members) / len(members),
            }
    return {
        "n": n,
        "class_counts": dict(sorted(classes.items())),
        "class_prevalence": {k: v / n for k, v in sorted(classes.items())},
        "mean_chars": sum(i.n_chars for i in items) / n,
        "min_chars": min(i.n_chars for i in items),
        "max_chars": max(i.n_chars for i in items),
        "length_strata": strata,
        "reference_label_rule": spec["reference_label_rule"],
    }


def freeze_task(
    task_key: str,
    out_dir: Path,
    gate_a_root: Path,
    *,
    rows: Iterable[dict[str, Any]] | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze one task. Raises RetrievalBlockedError rather than inventing evidence."""
    spec = TASKS[task_key]
    task_dir = out_dir / str(spec.get("task_id", task_key))
    refuse_overwrite(task_dir / "provenance.json")

    live = info if info is not None else fetch_dataset_info(spec["dataset"])
    audit = audit_license(spec, live)
    blockers = license_blockers(audit)
    if blockers:
        raise RetrievalBlockedError(
            f"{spec['dataset']}: " + "; ".join(blockers) + ". No frame is written."
        )
    restrictions = license_restrictions(spec, audit)

    excluded, exclusion_counts = pilot_excluded_ids(gate_a_root, spec)
    expected = spec.get("excluded_id_counts_expected") or {}
    for name, want in expected.items():
        got = exclusion_counts.get(f"{name}.jsonl", 0)
        if got != want:
            raise RetrievalBlockedError(
                f"pilot exclusion file {name}.jsonl held {got} ids, protocol expects {want}"
            )

    source_rows = rows if rows is not None else load_rows(spec)
    pool = candidates(source_rows, spec, excluded)
    items, filler = select(pool, spec)

    items_path = task_dir / "items.jsonl"
    filler_path = task_dir / "filler_pool.jsonl"
    write_jsonl(items_path, (asdict(i) for i in items))
    write_jsonl(filler_path, (asdict(i) for i in filler))

    provenance: dict[str, Any] = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "task_key": task_key,
        "task_id": spec.get("task_id", task_key),
        "dataset": spec["dataset"],
        "revision": spec["revision"],
        "split": spec["split"],
        "license_audit": audit,
        "license_restrictions": restrictions,
        "license_blockers": blockers,
        "license_finding": spec.get("license_finding", "no extra finding recorded"),
        "redistribution_permitted": spec["redistribution_permitted"],
        "text_is_published": False,
        "no_new_experimental_model_labels": NO_NEW_LABELS,
        "selection_rule": (select.__doc__ or "").strip(),
        "selection_seed": spec["selection_seed"],
        "sampling": spec["sampling"],
        "sampling_rationale": spec["sampling_rationale"],
        "min_chars": spec["min_chars"],
        "max_chars": spec["max_chars"],
        "scan_limit": spec["scan_limit"],
        "n_excluded_pilot_ids": len(excluded),
        "excluded_pilot_id_counts": exclusion_counts,
        "exclusion_rationale": spec.get("exclusion_rationale", "no pilot overlap for this task"),
        "n_candidates_after_exclusion": len(pool),
        "items": _describe(items, spec),
        "filler_pool": _describe(filler, spec),
        "disjoint_items_and_filler": not (
            {i.doc_id for i in items} & {f.doc_id for f in filler}
        ),
        "no_excluded_id_selected": not ({i.doc_id for i in items} & excluded),
        "files": {
            "items.jsonl": {
                "sha256": sha256_file(items_path),
                "rows": len(items),
                "row_ids_sha256": sha256_text("\n".join(i.doc_id for i in items)),
            },
            "filler_pool.jsonl": {
                "sha256": sha256_file(filler_path),
                "rows": len(filler),
                "row_ids_sha256": sha256_text("\n".join(f.doc_id for f in filler)),
            },
        },
    }
    write_json(task_dir / "provenance.json", provenance)
    return provenance


def load_items(task_dir: Path, name: str = "items.jsonl") -> list[Item]:
    return [Item(**row) for row in read_jsonl(task_dir / name)]
