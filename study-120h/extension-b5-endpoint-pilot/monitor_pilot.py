#!/usr/bin/env python
"""Read-only watch over an in-flight B5 endpoint pilot run.

Samples once a minute into monitor-<run>.jsonl. Prints a line on stdout only when the
alert state changes, so the caller is woken for events worth acting on and nothing else.
Touches no run artifact: it never opens .active-run.lock, receipts or the seal.
"""

import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY))
from extension_b4_runpod import cli  # noqa: E402

RUN = STUDY / "extension-b5-endpoint-pilot" / "run-01"
LOG = STUDY / "extension-b5-endpoint-pilot" / "monitor-run-01.jsonl"
PLANNED = 82944
CEILING_HOURS = 7.0
MAX_RATE = 1.59
MIN_BALANCE = 13.0
STALL_SECONDS = 900
REPEAT_SECONDS = 1800

ATTEMPT = re.compile(rb'"attempt_id":\s*"([^"]+)"')
STATUS = re.compile(rb'"status":\s*"([^"]*)"')
PARSE = re.compile(rb'"parse_status":\s*"([^"]*)"')


def now():
    return dt.datetime.now(dt.UTC)


def ledger():
    """Completed first attempts, retries and non-ok attempts, straight off events.jsonl."""
    path = RUN / "events.jsonl"
    if not path.is_file():
        return {"calls": 0, "retries": 0, "non_ok": 0, "event_age_s": None}
    calls = retries = non_ok = 0
    with path.open("rb") as handle:
        for line in handle:
            m = ATTEMPT.search(line)
            if not m:
                continue
            if m.group(1).endswith(b"::attempt::1"):
                calls += 1
            else:
                retries += 1
            s, p = STATUS.search(line), PARSE.search(line)
            if (s and s.group(1) != b"ok") or (p and p.group(1) != b"ok"):
                non_ok += 1
    return {
        "calls": calls,
        "retries": retries,
        "non_ok": non_ok,
        "event_age_s": round(time.time() - path.stat().st_mtime, 1),
    }


def meter():
    """Settled hours plus the wall clock of any lane still open."""
    path = RUN / "meter.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []
    settled_ids = {r["phase_id"] for r in rows if r["kind"] == "settled"}
    settled = sum(float(r["charged_hours"]) for r in rows if r["kind"] == "settled")
    open_rows = [r for r in rows if r["kind"] == "start" and r["phase_id"] not in settled_ids]
    open_h = 0.0
    for r in open_rows:
        started = dt.datetime.fromisoformat(r["started_at_utc"])
        open_h += (now() - started).total_seconds() / 3600.0
    return {
        "settled_hours": round(settled, 5),
        "open_lanes": len(open_rows),
        "open_hours": round(open_h, 5),
        "total_hours": round(settled + open_h, 5),
        "open_cells": [r["cell_key"] for r in open_rows],
    }


def account():
    identity = cli(["user"])
    identity = identity.get("user", identity)
    # runpodctl carries no GPU utilization; call rate below is the live work signal.
    rows = cli(["pod", "list", "--all"]) or []
    pods = []
    for pod in rows:
        runtime = pod.get("runtime") or {}
        gpus = runtime.get("gpus") or [{}]
        pods.append({
            "id": pod.get("id"),
            "name": pod.get("name"),
            "status": pod.get("desiredStatus") or pod.get("status"),
            "cost": pod.get("costPerHr") or pod.get("cost"),
            "gpu_util": gpus[0].get("util"),
            "gpu_mem_util": gpus[0].get("memoryUtil"),
            "uptime_s": pod.get("uptimeSeconds") or runtime.get("uptime"),
        })
    return {
        "balance_usd": identity.get("clientBalance"),
        "spend_per_hr": identity.get("currentSpendPerHr"),
        "pods": pods,
    }


def alerts(sample):
    """Every condition that should pull a human or an agent back to this run."""
    out = []
    led, met, acc = sample["ledger"], sample["meter"], sample["account"]
    live = [p for p in acc["pods"] if str(p["status"]).upper() == "RUNNING"]
    if led["calls"] >= PLANNED:
        out.append("DONE all 82944 calls recorded")
    if not sample["launcher_alive"]:
        out.append("ALERT launcher process gone")
    if len(live) > 1:
        out.append(f"ALERT {len(live)} Pods running, one is the limit")
    if met["open_lanes"] and not live:
        out.append("ALERT meter lane open with no running Pod")
    if live and not met["open_lanes"]:
        out.append("ALERT Pod running with no open meter lane")
    for p in live:
        if p["cost"] and float(p["cost"]) > MAX_RATE:
            out.append(f"ALERT Pod {p['id']} bills {p['cost']} over the {MAX_RATE} limit")
    if met["total_hours"] >= CEILING_HOURS:
        out.append(f"ALERT metered {met['total_hours']}h at or over the {CEILING_HOURS}h ceiling")
    elif met["total_hours"] >= 6.0:
        out.append(f"WARN metered {met['total_hours']}h approaches the {CEILING_HOURS}h ceiling")
    if sample["projected_total_hours"] and sample["projected_total_hours"] > CEILING_HOURS:
        out.append(f"WARN projected total {sample['projected_total_hours']}h exceeds the ceiling")
    if led["non_ok"]:
        out.append(f"ALERT {led['non_ok']} non-ok scientific attempts")
    if isinstance(acc["balance_usd"], (int, float)) and acc["balance_usd"] < MIN_BALANCE:
        out.append(f"WARN balance {round(acc['balance_usd'], 2)} under the {MIN_BALANCE} Pod gate")
    if live and led["event_age_s"] and led["event_age_s"] > STALL_SECONDS:
        out.append(f"WARN no new event for {int(led['event_age_s'])}s on a running Pod")
    return out


def sample():
    led, met, acc = ledger(), meter(), account()
    pct = round(100.0 * led["calls"] / PLANNED, 3)
    projected = round(met["total_hours"] * PLANNED / led["calls"], 4) if led["calls"] else None
    return {
        "at_utc": now().isoformat(),
        "percent_complete": pct,
        "calls": led["calls"],
        "planned_calls": PLANNED,
        "missing_calls": PLANNED - led["calls"],
        "launcher_alive": launcher_alive(),
        "ledger": led,
        "meter": met,
        "account": acc,
        "projected_total_hours": projected,
        "receipts": len(list((RUN / "receipts").glob("*.json"))),
    }


def launcher_alive():
    for pid in os.popen("pgrep -f 'run_b5_endpoint.py --resume'").read().split():
        if pid.strip().isdigit():
            return True
    return False


def main():
    last_state, last_emit, prev = None, 0.0, None
    while True:
        try:
            row = sample()
            if prev:
                span = (dt.datetime.fromisoformat(row["at_utc"])
                        - dt.datetime.fromisoformat(prev["at_utc"])).total_seconds() / 60.0
                row["calls_per_min"] = round((row["calls"] - prev["calls"]) / span, 1) if span else None
            prev = row
            row["alerts"] = alerts(row)
        except Exception as exc:  # a transient API read must not kill the watch
            row = {"at_utc": now().isoformat(), "alerts": [], "error": f"{type(exc).__name__}: {exc}"}
        with LOG.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        state = "|".join(row["alerts"])
        if row["alerts"] and (state != last_state or time.time() - last_emit > REPEAT_SECONDS):
            pods = ",".join(p["id"] for p in row.get("account", {}).get("pods", [])) or "none"
            print(
                f"{row['at_utc']} {row.get('percent_complete')}% "
                f"calls={row.get('calls')} metered={row.get('meter', {}).get('total_hours')}h "
                f"pods={pods} :: {state}",
                flush=True,
            )
            last_emit = time.time()
        last_state = state
        if any(a.startswith("DONE") for a in row["alerts"]):
            return
        time.sleep(60)




def demo() -> None:
    """python monitor_pilot.py --demo — the alert table must fire on each gate."""
    base = {
        "launcher_alive": True,
        "projected_total_hours": 3.0,
        "ledger": {"calls": 100, "non_ok": 0, "event_age_s": 10.0},
        "meter": {"open_lanes": 1, "total_hours": 1.0},
        "account": {"balance_usd": 20.0, "pods": [{"id": "a", "status": "RUNNING", "cost": 1.59}]},
    }
    def with_(path, value):
        """path is "key" or "key.subkey"."""
        row = json.loads(json.dumps(base))
        head, _, tail = path.partition(".")
        if tail:
            row[head][tail] = value
        else:
            row[head] = value
        return row

    assert alerts(base) == [], alerts(base)
    assert any(a.startswith("ALERT launcher") for a in alerts(with_("launcher_alive", False)))
    two = with_("account.pods", [
        {"id": "a", "status": "RUNNING", "cost": 1.59},
        {"id": "b", "status": "RUNNING", "cost": 1.59},
    ])
    assert any("2 Pods running" in a for a in alerts(two))
    assert any("no running Pod" in a for a in alerts(with_("account.pods", [])))
    assert any("no open meter lane" in a for a in alerts(with_("meter.open_lanes", 0)))
    assert any("over the 1.59 limit" in a for a in alerts(with_("account.pods", [
        {"id": "a", "status": "RUNNING", "cost": 2.5}])))
    assert any("7.0h ceiling" in a and a.startswith("ALERT") for a in alerts(with_("meter.total_hours", 7.1)))
    assert any(a.startswith("WARN metered") for a in alerts(with_("meter.total_hours", 6.2)))
    assert any("projected total" in a for a in alerts(with_("projected_total_hours", 9.0)))
    assert any("non-ok" in a for a in alerts(with_("ledger.non_ok", 3)))
    assert any("Pod gate" in a for a in alerts(with_("account.balance_usd", 4.0)))
    assert any("no new event" in a for a in alerts(with_("ledger.event_age_s", 2000.0)))
    assert any(a.startswith("DONE") for a in alerts(with_("ledger.calls", PLANNED)))
    print("monitor_pilot demo ok")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
