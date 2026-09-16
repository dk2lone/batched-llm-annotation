"""The external watchdog is tested without contacting Runpod."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import pytest

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))

import runpod_watchdog as watchdog  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.value = dt.datetime(2026, 9, 16, tzinfo=dt.UTC)

    def now(self) -> dt.datetime:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += dt.timedelta(seconds=seconds)


class FakeRunpod:
    def __init__(self, snapshots: list[list[dict[str, Any]]]) -> None:
        self.snapshots = snapshots
        self.index = 0
        self.deleted: list[str] = []

    def pods(self) -> list[dict[str, Any]]:
        value = self.snapshots[min(self.index, len(self.snapshots) - 1)]
        self.index += 1
        return [pod for pod in value if pod.get("id") not in self.deleted]

    def delete(self, pod_id: str) -> None:
        self.deleted.append(pod_id)


def test_watchdog_deletes_only_exact_lease_and_records_evidence(tmp_path: Path) -> None:
    clock = Clock()
    lease = "0123456789abcdef"
    own = {"id": "owned", "name": f"b4-{lease}-lane", "desiredStatus": "RUNNING"}
    foreign = {"id": "foreign", "name": "b4-fedcba9876543210-lane", "desiredStatus": "RUNNING"}
    client = FakeRunpod([[foreign], [foreign, own], [foreign, own], [foreign, own]])
    audit = tmp_path / "audit.jsonl"
    watchdog.watch(
        client, lease, clock.now() + dt.timedelta(seconds=31), 1, audit, 15, clock.now, clock.sleep
    )
    assert client.deleted == ["owned"]
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [row["kind"] for row in rows] == [
        "armed",
        "pods_observed",
        "delete_requested",
        "termination_verified",
    ]
    assert rows[-1]["observed_pod_ids"] == ["owned"]


@pytest.mark.parametrize(
    ("lease", "seconds", "expected"),
    [
        ("short", 60, 1),
        ("0123456789abcdef", 29, 1),
        ("0123456789abcdef", 5 * 60 * 60 + 1, 1),
        ("0123456789abcdef", 60, 0),
        ("0123456789abcdef", 60, 5),
    ],
)
def test_watchdog_rejects_unbounded_or_ambiguous_targets(
    lease: str, seconds: int, expected: int
) -> None:
    now = dt.datetime(2026, 9, 16, tzinfo=dt.UTC)
    with pytest.raises(ValueError):
        watchdog.validate(lease, now + dt.timedelta(seconds=seconds), expected, now)


def test_matching_accepts_live_api_id_variants_and_rejects_bad_ids() -> None:
    prefix = "b4-0123456789abcdef-"
    assert watchdog.matching(
        [{"podId": "pod_one", "name": prefix + "lane", "desiredStatus": "RUNNING"}], prefix
    ) == {"pod_one": "RUNNING"}
    with pytest.raises(RuntimeError):
        watchdog.matching([{"id": "bad/id", "name": prefix + "lane"}], prefix)
