"""Ordered pinned model policies and metadata-only access audit."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from step0.hashing import sha256_file, sha256_json


@dataclass(frozen=True)
class Model:
    repo: str
    revision: str
    family: str
    order: int


MODELS = (
    Model("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "qwen", 1),
    Model("allenai/OLMo-2-1124-7B-Instruct", "470b1fba1ae01581f270116362ee4aa1b97f4c84", "olmo", 2),
    Model(
        "ibm-granite/granite-3.3-8b-instruct",
        "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
        "granite",
        3,
    ),
    Model("microsoft/Phi-3.5-mini-instruct", "2fe192450127e6a83f7441aef6e3ca586c338b77", "phi", 4),
    Model("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", "qwen", 5),
    Model("allenai/OLMo-2-1124-7B", "7df9a82518afdecae4e8c026b27adccc8c1f0032", "olmo", 6),
    Model(
        "ibm-granite/granite-3.3-2b-instruct",
        "707f574c62054322f6b5b04b6d075f0a8f05e0f0",
        "granite",
        7,
    ),
    Model("google/gemma-3-4b-it", "093f9f388b31de276ce2de164bdc2081324b9767", "gemma", 8),
)


AUDIT_FILES = ("config.json", "tokenizer_config.json", "chat_template.jinja")


def file_fingerprints(info: Any) -> list[dict[str, Any]]:
    """Per-file size and git/LFS identity from Hub metadata; no weight download.

    This is the only checkpoint fingerprint available for a gated repository,
    whose file contents cannot be read without an accepted licence and a token.
    """
    rows = []
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        rows.append({
            "path": sibling.rfilename,
            "bytes": sibling.size,
            "blob_id": getattr(sibling, "blob_id", None),
            "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None,
        })
    return sorted(rows, key=lambda row: str(row["path"]))


def audit_model(model: Model) -> dict[str, Any]:
    """Inspect only Hub metadata and small pinned config files; never load weights."""
    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

    info = HfApi().model_info(model.repo, revision=model.revision, files_metadata=True)
    fingerprints = file_fingerprints(info)
    result: dict[str, Any] = {
        "repo": model.repo,
        "family": model.family,
        "order": model.order,
        "revision": model.revision,
        "retrieved_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pinned_sha_matches": info.sha == model.revision,
        "gated": info.gated,
        "private": info.private,
        "disabled": info.disabled,
        "license": getattr(info.card_data, "license", None),
        "n_files": len(fingerprints),
        "file_fingerprints": fingerprints,
        "file_fingerprints_sha256": sha256_json(fingerprints),
        "files": {},
    }
    for filename in AUDIT_FILES:
        try:
            path = Path(hf_hub_download(model.repo, filename, revision=model.revision))
        except Exception as exc:  # noqa: BLE001 - record a technical exclusion, never fabricate
            result["files"][filename] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        record: dict[str, Any] = {"sha256": sha256_file(path)}
        if filename.endswith(".json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            record.update({
                "model_max_length": payload.get("model_max_length"),
                "max_position_embeddings": payload.get("max_position_embeddings"),
                "chat_template_present": bool(payload.get("chat_template")),
                "architectures": payload.get("architectures"),
                "model_type": payload.get("model_type"),
                "torch_dtype": payload.get("torch_dtype") or payload.get("dtype"),
                "vocab_size": payload.get("vocab_size"),
            })
        result["files"][filename] = record
    embedded = bool(
        result["files"].get("tokenizer_config.json", {}).get("chat_template_present")
    )
    separate = "sha256" in result["files"].get("chat_template.jinja", {})
    result["chat_template"] = {
        "embedded_in_tokenizer_config": embedded,
        "separate_jinja_file": separate,
        "present": embedded or separate,
    }
    tokenizer = next(
        (row for row in fingerprints if row["path"] in ("tokenizer.json", "tokenizer.model")), None
    )
    result["tokenizer_vocabulary"] = tokenizer
    return result
