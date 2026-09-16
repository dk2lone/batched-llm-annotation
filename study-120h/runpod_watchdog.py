"""Delete every Runpod Pod belonging to one bounded B4 lease."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

API_ROOT = "https://api.runpod.io/v2"
LEASE = re.compile(r"[0-9a-f]{16}")
POD_ID = re.compile(r"[A-Za-z0-9_-]+")
MAX_TARGET_SECONDS = 5 * 60 * 60
DELETE_GRACE_SECONDS = 10 * 60


def utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("watchdog deadline must include a UTC offset")
    return parsed.astimezone(dt.UTC)


def validate(lease_id: str, deadline: dt.datetime, expected_pods: int, now: dt.datetime) -> str:
    if not LEASE.fullmatch(lease_id):
        raise ValueError("invalid B4 lease ID")
    seconds = (deadline - now).total_seconds()
    if not 30 <= seconds <= MAX_TARGET_SECONDS:
        raise ValueError("watchdog target must be 30 seconds through five hours ahead")
    if not 1 <= expected_pods <= 4:
        raise ValueError("watchdog must guard one through four Pods")
    return f"b4-{lease_id}-"


class Runpod:
    def __init__(self, api_key: str, root: str = API_ROOT) -> None:
        if not api_key:
            raise ValueError("RUNPOD_API_KEY is required")
        self.api_key = api_key
        self.root = root.rstrip("/")

    def request(self, method: str, path: str) -> Any:
        request = Request(
            self.root + path,
            method=method,
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=20) as response:
                data = response.read()
        except HTTPError as exc:
            if method == "DELETE" and exc.code == 404:
                return None
            raise RuntimeError(f"Runpod watchdog API failed with HTTP {exc.code}") from None
        return json.loads(data) if data.strip() else None

    def pods(self) -> list[dict[str, Any]]:
        value = self.request("GET", "/pods")
        if not isinstance(value, dict) or not isinstance(value.get("pods"), list):
            raise RuntimeError("Runpod Pod list response shape changed")
        return cast(list[dict[str, Any]], value["pods"])

    def delete(self, pod_id: str) -> None:
        if not POD_ID.fullmatch(pod_id):
            raise ValueError("invalid Runpod Pod ID")
        self.request("DELETE", "/pods/" + quote(pod_id, safe=""))


class PodClient(Protocol):
    def pods(self) -> list[dict[str, Any]]: ...

    def delete(self, pod_id: str) -> None: ...


def matching(pods: list[dict[str, Any]], prefix: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for pod in pods:
        name = pod.get("name")
        pod_id = pod.get("id", pod.get("podId"))
        if not isinstance(name, str) or not name.startswith(prefix):
            continue
        if not isinstance(pod_id, str) or not POD_ID.fullmatch(pod_id):
            raise RuntimeError("matching Pod has an invalid ID")
        status = str(pod.get("status", "UNKNOWN")).upper()
        if status != "TERMINATED":
            found[pod_id] = status
    return found


def append(path: Path, kind: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"kind": kind, "at_utc": dt.datetime.now(dt.UTC).isoformat(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def watch(
    client: PodClient,
    lease_id: str,
    deadline: dt.datetime,
    expected_pods: int,
    audit: Path,
    poll_seconds: float = 15,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    prefix = validate(lease_id, deadline, expected_pods, now())
    observed: set[str] = set()
    append(
        audit,
        "armed",
        lease_id=lease_id,
        pod_name_prefix=prefix,
        deadline_utc=deadline.isoformat(),
        expected_pods=expected_pods,
    )
    while now() < deadline:
        pods = matching(client.pods(), prefix)
        new = sorted(set(pods) - observed)
        if new:
            observed.update(new)
            append(audit, "pods_observed", pod_ids=new, observed_count=len(observed))
        sleep(max(0.1, min(poll_seconds, (deadline - now()).total_seconds())))

    cleanup_deadline = now() + dt.timedelta(seconds=DELETE_GRACE_SECONDS)
    while True:
        pods = matching(client.pods(), prefix)
        for pod_id in sorted(pods):
            append(audit, "delete_requested", pod_id=pod_id, status=pods[pod_id])
            client.delete(pod_id)
        remaining = matching(client.pods(), prefix)
        if not remaining:
            append(
                audit,
                "termination_verified",
                observed_pod_ids=sorted(observed | set(pods)),
                observed_count=len(observed | set(pods)),
            )
            return
        if now() >= cleanup_deadline:
            append(audit, "termination_unverified", pod_ids=sorted(remaining))
            raise RuntimeError("watchdog could not verify Pod deletion")
        sleep(min(poll_seconds, max(0.1, (cleanup_deadline - now()).total_seconds())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--expected-pods", required=True, type=int)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    watch(
        Runpod(os.environ.get("RUNPOD_API_KEY", "")),
        args.lease_id,
        utc(args.deadline_utc),
        args.expected_pods,
        args.audit,
        args.poll_seconds,
    )


if __name__ == "__main__":
    main()
