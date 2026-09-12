"""Where Step 1 writes. Nothing here is inside `freeze/`, which stays sealed and read-only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .frozen import PACKAGE_ROOT


@dataclass(frozen=True)
class Paths:
    root: Path = PACKAGE_ROOT / "step1-out"

    @property
    def preflight(self) -> Path:
        return self.root / "preflight"

    @property
    def host(self) -> Path:
        return self.root / "host"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def runs(self) -> Path:
        return self.root / "runs.jsonl"

    @property
    def screen(self) -> Path:
        return self.root / "screen"

    @property
    def benchmark(self) -> Path:
        return self.root / "benchmark"

    @property
    def blockers(self) -> Path:
        return self.root / "blockers"

    @staticmethod
    def model_slug(model_id: str) -> str:
        return model_id.replace("/", "_")

    def ledger(self, cell_id: str, frame_task_id: str, model_id: str) -> Path:
        slug = self.model_slug(model_id)
        return self.results / f"{cell_id}__{frame_task_id}__{slug}" / "solo.jsonl"

    def preflight_report(self, cell_id: str, frame_task_id: str, model_id: str) -> Path:
        slug = self.model_slug(model_id)
        return self.preflight / f"{cell_id}__{frame_task_id}__{slug}.json"

    def host_record(self, worker_id: str) -> Path:
        return self.host / f"{worker_id}.json"

    def load_failure(self, model_id: str) -> Path:
        return self.blockers / f"load-failure--{self.model_slug(model_id)}.json"


AMENDMENTS = PACKAGE_ROOT / "amendments"
STATUS_JSON = PACKAGE_ROOT / "STEP1_STATUS.json"
STATUS_MD = PACKAGE_ROOT / "STEP1_STATUS.md"
