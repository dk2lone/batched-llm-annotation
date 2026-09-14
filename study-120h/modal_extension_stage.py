"""Stage and verify the pinned X1R4 checkpoints in the active Modal workspace."""

import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import modal

LOCAL_ROOT = Path(__file__).resolve().parent
MODEL_CACHE = "/models"
app = modal.App("ieee-bigdata-extension-stage")
volume = modal.Volume.from_name("ieee-bigdata-freezeb-models", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "huggingface-hub==1.31.0"
).add_local_dir(LOCAL_ROOT / "src", remote_path="/root/study-120h/src")


@app.function(image=image, volumes={MODEL_CACHE: volume}, timeout=7200)
def stage(
    model_id: str, revision: str, with_hashes: bool, excluded_alternatives: list[str]
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=MODEL_CACHE,
            ignore_patterns=excluded_alternatives,
        )
    )
    for relative_path in excluded_alternatives:
        (snapshot / relative_path).unlink(missing_ok=True)
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    rows = []
    for path in files:
        row: dict[str, Any] = {
            "path": str(path.relative_to(snapshot)),
            "bytes": path.stat().st_size,
        }
        if with_hashes:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            row["sha256"] = digest.hexdigest()
        rows.append(row)
    volume.commit()
    result: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
    }
    if with_hashes:
        sys.path.insert(0, "/root/study-120h/src")
        from step0.hashing import sha256_json

        result["files_sha256"] = sha256_json(rows)
    return result


@app.local_entrypoint()
def main() -> None:
    from huggingface_hub import get_token

    sys.path.insert(0, str(LOCAL_ROOT / "src"))
    from freezeb.extension_screen import expected_checkpoint_inventory, verify_checkpoint_inventory

    expected = expected_checkpoint_inventory(
        LOCAL_ROOT / "runtime-lock/20260913T205646Z-X1R4",
        LOCAL_ROOT / "runtime-lock/20260913T004054Z-BRT1/benchmark/checkpoint-manifests.json",
    )
    if len(expected) != 5:
        raise RuntimeError("the sealed evidence does not inventory all five checkpoints")
    staging_record = json.loads(
        (LOCAL_ROOT / "runtime-lock/20260913T205646Z-X1R4/evidence/staging/report.json").read_text(
            encoding="utf-8"
        )
    )
    excluded = {
        row["model_id"]: row["excluded_alternative_files"] for row in staging_record["models"]
    }
    token = get_token()
    if not token:
        raise RuntimeError("a local Hugging Face login is required to stage gated Llama")
    staged = stage.with_options(secrets=[modal.Secret.from_dict({"HF_TOKEN": token})])

    def one(model_id: str) -> dict[str, Any]:
        record = expected[model_id]
        actual = staged.remote(
            model_id,
            str(record["revision"]),
            "files_sha256" in record,
            excluded.get(model_id, []),
        )
        return verify_checkpoint_inventory(record, actual)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for result in executor.map(one, expected):
            print(json.dumps(result, sort_keys=True), flush=True)
