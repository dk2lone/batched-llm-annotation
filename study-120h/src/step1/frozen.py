"""Read-only view of the sealed Step 0 freeze.

The freeze is the only authority for every design field. Nothing in this module writes
into it. `load` resolves `freeze/LATEST`, parses `protocol.lock.json`, and refuses to
continue when the seal does not verify, so a mutated freeze can never reach a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from step0 import integrity
from step0.hashing import read_jsonl
from step0.lock import ASSIGNMENT_GENERATOR, CONDITIONS, PANEL

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FREEZE_ROOT = PACKAGE_ROOT / "freeze"

#: The three task keys of the frozen protocol, and the frame directory of each.
TASK_KEYS: dict[str, str] = {
    "civil_comments": "civil_comments",
    "second_task_first_candidate": "ag_news",
    "second_task_conditional_fallback": "multi_nli",
}
SECOND_TASK_ORDER = ("ag_news", "multi_nli")


class FreezeError(RuntimeError):
    """Raised when the freeze is missing, unsealed, mutated, or internally inconsistent."""


@dataclass(frozen=True)
class Item:
    """One frozen record of an item frame. `text` is never printed into a report."""

    doc_id: str
    text: str
    reference: int
    n_chars: int
    length_stratum: int
    source_index: int


@dataclass(frozen=True)
class Frozen:
    """Everything Step 1 is allowed to read, behind one object."""

    freeze_dir: Path
    protocol: dict[str, Any]

    # -- identity ------------------------------------------------------------------
    @property
    def stamp(self) -> str:
        return str(self.protocol["freeze"]["stamp_utc"])

    @property
    def content_hash(self) -> str:
        return str(self.protocol["content_hash"])

    @property
    def serving(self) -> dict[str, Any]:
        return dict(self.protocol["panel"]["serving"])

    @property
    def decoding(self) -> dict[str, Any]:
        return dict(self.protocol["panel"]["decoding"])

    @property
    def cells(self) -> list[dict[str, Any]]:
        return [dict(c) for c in self.protocol["panel"]["cells"]]

    def cell(self, cell_id: str) -> dict[str, Any]:
        for candidate in self.cells:
            if candidate["cell_id"] == cell_id:
                return candidate
        raise FreezeError(f"{cell_id} is not a frozen cell")

    # -- decoding budget -----------------------------------------------------------
    def max_tokens(self, n_requested: int) -> int:
        """The frozen output cap. base + per_label * n_requested_labels."""
        decoding = self.decoding
        return int(decoding["max_tokens_base"]) + int(decoding["max_tokens_per_label"]) * int(
            n_requested
        )

    def prompt_token_budget(self, n_requested: int) -> int:
        """max_model_len minus the frozen output cap, under the pinned tokenizer."""
        return int(self.serving["max_model_len"]) - self.max_tokens(n_requested)

    # -- tasks ---------------------------------------------------------------------
    def task_spec(self, task_key: str) -> dict[str, Any]:
        for key, task_id in TASK_KEYS.items():
            if task_key in (key, task_id):
                return dict(self.protocol["tasks"][key])
        raise FreezeError(f"{task_key} is not a frozen task")

    def n_classes(self, task_id: str) -> int:
        return int(self.task_spec(task_id)["n_classes"])

    def permitted_labels(self, task_id: str) -> tuple[int, ...]:
        return tuple(range(self.n_classes(task_id)))

    def capability_floor(self, task_id: str) -> float:
        floors = self.protocol["solo_screen"]["criteria"]["capability_floor"]
        if task_id not in floors:
            raise FreezeError(f"no frozen capability floor for {task_id}")
        return float(floors[task_id])

    def items(self, task_id: str) -> list[Item]:
        """The frozen 2,048-item frame of one task, in file order."""
        path = self.freeze_dir / "data" / task_id / "items.jsonl"
        if not path.exists():
            raise FreezeError(f"{path} is missing from the freeze")
        return [
            Item(
                doc_id=str(row["doc_id"]),
                text=str(row["text"]),
                reference=int(row["reference"]),
                n_chars=int(row["n_chars"]),
                length_stratum=int(row["length_stratum"]),
                source_index=int(row["source_index"]),
            )
            for row in read_jsonl(path)
        ]

    def filler_pool(self, task_id: str) -> list[Item]:
        path = self.freeze_dir / "data" / task_id / "filler_pool.jsonl"
        if not path.exists():
            raise FreezeError(f"{path} is missing from the freeze")
        return [
            Item(
                doc_id=str(row["doc_id"]),
                text=str(row["text"]),
                reference=int(row["reference"]),
                n_chars=int(row["n_chars"]),
                length_stratum=int(row["length_stratum"]),
                source_index=int(row["source_index"]),
            )
            for row in read_jsonl(path)
        ]

    def dataset_revision(self, task_id: str) -> str:
        return str(self.task_spec(task_id)["revision"])

    def model_revision(self, model_id: str) -> str:
        models = self.protocol["models"]
        if model_id not in models:
            raise FreezeError(f"{model_id} is not a pinned model")
        return str(models[model_id]["revision"])

    def tokenizer_revision(self, model_id: str) -> str:
        models = self.protocol["models"]
        if model_id not in models:
            raise FreezeError(f"{model_id} is not a pinned model")
        return str(models[model_id]["tokenizer_revision"])

    # -- planned calls -------------------------------------------------------------
    def job_skeletons(self, cell_id: str) -> list[dict[str, Any]]:
        path = self.freeze_dir / "assignments" / cell_id / "job_skeletons.jsonl"
        if not path.exists():
            raise FreezeError(f"{path} is missing from the freeze")
        return list(read_jsonl(path))

    @cached_property
    def planning_second_task(self) -> str:
        """The frame the frozen assignments were planned on: AG News, the first candidate."""
        summary = self.freeze_dir / "assignments" / "SUMMARY.json"
        if not summary.exists():
            raise FreezeError(f"{summary} is missing from the freeze")
        import json  # noqa: PLC0415 - only needed on this path

        return str(
            json.loads(summary.read_text(encoding="utf-8"))["second_task_frame_used_for_planning"]
        )


def resolve_freeze_dir(stamp: str | None = None) -> Path:
    """`freeze/<stamp>`, or the directory named by `freeze/LATEST`."""
    if stamp:
        return FREEZE_ROOT / stamp
    pointer = FREEZE_ROOT / "LATEST"
    if not pointer.exists():
        raise FreezeError(f"{pointer} is missing; Step 0 has not sealed a freeze")
    return FREEZE_ROOT / pointer.read_text(encoding="utf-8").strip()


def check_generator_matches_freeze(protocol: dict[str, Any]) -> list[str]:
    """The installed generator constants must equal the frozen ones.

    Step 1 reuses `step0.assign` to rebuild the plan on the MultiNLI fallback frame. If the
    live `step0.lock` drifted away from the sealed protocol, that rebuild would silently use
    different rules, so the drift is a hard error rather than a warning.
    """
    problems: list[str] = []
    for name, live in (
        ("assignment_generator", ASSIGNMENT_GENERATOR),
        ("conditions", CONDITIONS),
        ("panel", PANEL),
    ):
        if live != protocol[name]:
            problems.append(f"live step0.lock {name} differs from the sealed protocol")
    return problems


def load(stamp: str | None = None, *, require_seal: bool = True) -> Frozen:
    """Load the sealed freeze. Refuse to continue on a broken seal or a drifted generator."""
    import json  # noqa: PLC0415 - only needed once, at load time

    freeze_dir = resolve_freeze_dir(stamp)
    lock_path = freeze_dir / "protocol.lock.json"
    if not lock_path.exists():
        raise FreezeError(f"{lock_path} is missing; {freeze_dir} is not a Step 0 freeze")
    protocol = json.loads(lock_path.read_text(encoding="utf-8"))

    if require_seal:
        report = integrity.verify(freeze_dir)
        if not report["ok"]:
            raise FreezeError(
                "the Step 0 seal does not verify, so the freeze is void: "
                + "; ".join(str(p) for p in report["problems"][:5])
            )
    problems = check_generator_matches_freeze(protocol)
    if problems:
        raise FreezeError("; ".join(problems))
    return Frozen(freeze_dir=freeze_dir, protocol=protocol)
