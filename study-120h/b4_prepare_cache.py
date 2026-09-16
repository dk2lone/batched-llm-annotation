"""Populate and verify one sealed Hugging Face cache from a token on stdin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache", type=Path, default=Path("/workspace/hf"))
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    if not token:
        raise RuntimeError("Hugging Face token was not supplied on stdin")
    stale_incomplete = sorted(args.cache.rglob("*.incomplete"))
    for path in stale_incomplete:
        path.unlink()
    snapshot = Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            cache_dir=args.cache,
            token=token,
            max_workers=1,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
    )
    token = ""
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weights = [snapshot / name for name in sorted(set(index["weight_map"].values()))]
    else:
        weights = sorted(snapshot.glob("*.safetensors"))
    missing = [path.name for path in weights if not path.is_file()]
    if not weights or missing:
        raise RuntimeError(f"sealed snapshot weights are incomplete: {missing}")
    incomplete = sorted(args.cache.rglob("*.incomplete"))
    if incomplete:
        raise RuntimeError(f"cache retained {len(incomplete)} incomplete files")
    print(
        json.dumps(
            {
                "cache_ready": True,
                "model": args.model,
                "revision": args.revision,
                "removed_stale_incomplete_files": len(stale_incomplete),
                "snapshot": str(snapshot),
                "weight_files": len(weights),
                "weight_bytes": sum(path.stat().st_size for path in weights),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
