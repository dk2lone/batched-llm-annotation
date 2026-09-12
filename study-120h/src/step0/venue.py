"""Venue and authorship record built from the live call pages.

Retrieves both candidate tracks, records the URL, retrieval date, and page hash,
extracts the eligibility, page limit, and deadline text, and flags that personal
eligibility still needs human confirmation.
"""

from __future__ import annotations

import datetime as dt
import html
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .hashing import sha256_text, write_json, write_text
from .lock import VENUE

PAGES = {
    "high_school_symposium": "https://bigdataieee.org/BigData2026/high-school-symposium/",
    "undergraduate_reu_consortium": (
        "https://bigdataieee.org/BigData2026/undergraduate-reu-consortium/"
    ),
}
USER_AGENT = "step0-venue-check/1.0 (IEEE BigData 2026 protocol freeze; CPU only)"


class VenueRetrievalError(RuntimeError):
    """The live page could not be read. The record stays incomplete."""


def fetch(url: str, timeout: float = 60.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return str(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise VenueRetrievalError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc


def to_text(page: str) -> str:
    body = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", page)
    body = re.sub(r"(?s)<[^>]+>", "\n", body)
    body = html.unescape(body)
    body = re.sub(r"[ \t\xa0]+", " ", body)
    return "\n".join(line.strip() for line in body.split("\n") if line.strip())


def _first(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0).strip() if match else None


def _labelled_date(label: str, text: str) -> str | None:
    """Read a date that the page puts on the line after its label."""
    match = re.search(rf"{re.escape(label)}:[ \t]*\n?([^\n]+)", text, re.IGNORECASE)
    return f"{label}: {match.group(1).strip()}" if match else None


def extract(text: str) -> dict[str, Any]:
    return {
        "first_author_rule": _first(
            r"(?:The first author must be[^.]*\.|Undergraduate students pursuing[^.]*\.)", text
        ),
        "primary_contributor_rule": _first(r"Co-authorship with[^.]*\.", text),
        "page_limit_sentence": _first(r"[^.\n]*should not exceed \d+ pages[^.]*\.", text),
        "page_limit": (
            int(m.group(1)) if (m := re.search(r"not exceed (\d+) pages", text)) else None
        ),
        "deadline_sentence": _labelled_date("Paper Submission Deadline", text),
        "notification_sentence": _labelled_date("Notification of Acceptance", text),
        "camera_ready_sentence": _labelled_date("Camera-Ready Paper Submission", text),
        "review_process": _first(r"The review process is single-blind[^.]*\.", text),
    }


def build(out_dir: Path) -> dict[str, Any]:
    retrieved = dt.datetime.now(dt.UTC)
    records: dict[str, Any] = {}
    for name, url in PAGES.items():
        page = fetch(url)
        text = to_text(page)
        write_text(out_dir / f"{name}.txt", text)
        records[name] = {
            "url": url,
            "retrieved_at_utc": retrieved.isoformat(),
            "page_sha256": sha256_text(page),
            "extracted_text_sha256": sha256_text(text),
            "extracted_text_file": f"{name}.txt",
            "facts": extract(text),
        }

    hs = records["high_school_symposium"]["facts"]
    ug = records["undergraduate_reu_consortium"]["facts"]
    record: dict[str, Any] = {
        "retrieved_at_utc": retrieved.isoformat(),
        "pages": records,
        "frozen_route": VENUE["frozen_route"],
        "frozen_route_url": VENUE["frozen_route_url"],
        "frozen_route_page_limit": ug["page_limit"],
        "frozen_route_deadline": ug["deadline_sentence"],
        "frozen_route_eligibility_text": ug["first_author_rule"],
        "frozen_route_condition": VENUE["conditional_on"],
        "ineligible_route": VENUE["ineligible_route"],
        "ineligible_route_url": VENUE["ineligible_route_url"],
        "ineligible_route_page_limit": hs["page_limit"],
        "ineligible_route_deadline": hs["deadline_sentence"],
        "ineligible_route_eligibility_text": hs["first_author_rule"],
        "ineligible_reason": VENUE["ineligible_reason"],
        "authorship_rule": VENUE["authorship_rule"],
        "human_confirmation_required": True,
        "human_confirmation_question": (
            "Is Daniel an undergraduate student pursuing an academic degree at the time of "
            "submission, and is he the primary contributor to this work?"
        ),
        "consistency_checks": {
            "page_limit_matches_protocol": ug["page_limit"] == VENUE["page_limit"],
            "deadline_text_mentions_september_20": bool(
                ug["deadline_sentence"] and "September 20" in ug["deadline_sentence"]
            ),
            "deadline_text_mentions_aoe": bool(
                ug["deadline_sentence"] and "AoE" in ug["deadline_sentence"]
            ),
            "high_school_route_requires_high_school_first_author": bool(
                hs["first_author_rule"] and "high school" in hs["first_author_rule"].lower()
            ),
        },
    }
    write_json(out_dir / "venue.json", record)
    write_text(out_dir / "VENUE.md", render(record))
    return record


def render(r: dict[str, Any]) -> str:
    lines = [
        "# Venue and authorship record",
        "",
        f"Retrieved from the live call pages on {r['retrieved_at_utc']}.",
        "",
        "## Frozen route",
        "",
        f"**{r['frozen_route']}**",
        "",
        f"- URL: {r['frozen_route_url']}",
        f"- Page limit: {r['frozen_route_page_limit']} pages, including figures, tables, "
        "and references",
        f"- {r['frozen_route_deadline']}",
        f"- {r['pages']['undergraduate_reu_consortium']['facts']['notification_sentence']}",
        f"- {r['pages']['undergraduate_reu_consortium']['facts']['camera_ready_sentence']}",
        "",
        "Eligibility text quoted from the page:",
        "",
        f"> {r['frozen_route_eligibility_text']}",
        "",
        f"> {r['pages']['undergraduate_reu_consortium']['facts']['primary_contributor_rule']}",
        "",
        f"This route is conditional on: {r['frozen_route_condition']}",
        "",
        "## Route ruled out",
        "",
        f"**{r['ineligible_route']}**",
        "",
        f"- URL: {r['ineligible_route_url']}",
        f"- Page limit: {r['ineligible_route_page_limit']} pages",
        f"- {r['ineligible_route_deadline']}",
        "",
        "Eligibility text quoted from the page:",
        "",
        f"> {r['ineligible_route_eligibility_text']}",
        "",
        r["ineligible_reason"],
        "",
        "## Authorship rule",
        "",
        r["authorship_rule"] + ".",
        "",
        "## Open item that a human must close",
        "",
        f"**{r['human_confirmation_question']}**",
        "",
        "This record cannot verify a personal fact. Until a human confirms it, the frozen route "
        "is provisional. Do not submit under either track without that confirmation.",
        "",
        "## Consistency checks",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for name, value in r["consistency_checks"].items():
        lines.append(f"| `{name}` | {value} |")
    lines.append("")
    return "\n".join(lines)
