"""Venue record tests. Extraction runs on a fixture page, so no network is needed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from step0 import venue

UNDERGRADUATE_PAGE = """
<html><head><style>p {color:red}</style><script>var x=1;</script></head><body>
<h1>Undergraduate and REU Consortium</h1>
<p>Undergraduate students pursuing an academic degree at the time of submission are
eligible to submit papers as first authors. Co-authorship with faculty members, mentors,
or researchers is allowed, but the undergraduate student must be the primary contributor
to the work.</p>
<p>Papers should not exceed 6 pages, including figures, tables and references.</p>
<p>The review process is single-blind, meaning that reviewers remain anonymous, but
authors are not.</p>
<ul><li>Paper Submission Deadline:</li><li>September 20, 2026, 11:59 PM AoE</li>
<li>Notification of Acceptance:</li><li>October 9, 2026</li>
<li>Camera-Ready Paper Submission:</li><li>October 24, 2026</li></ul>
</body></html>
"""

HIGH_SCHOOL_PAGE = """
<html><body>
<p>The first author must be a high school student at the time of submission.</p>
<p>Papers should not exceed 5 pages, including everything.</p>
<ul><li>Paper Submission Deadline:</li><li>September 20 2026</li>
<li>Notification of Acceptance:</li><li>October 9, 2026</li>
<li>Camera-Ready Paper Submission:</li><li>October 24, 2026</li></ul>
</body></html>
"""


def test_to_text_drops_scripts_styles_and_tags() -> None:
    text = venue.to_text(UNDERGRADUATE_PAGE)
    assert "var x" not in text
    assert "color:red" not in text
    assert "<p>" not in text
    assert "Undergraduate and REU Consortium" in text


def test_extract_reads_a_date_placed_on_the_next_line() -> None:
    facts = venue.extract(venue.to_text(UNDERGRADUATE_PAGE))
    deadline = "Paper Submission Deadline: September 20, 2026, 11:59 PM AoE"
    assert facts["deadline_sentence"] == deadline
    assert facts["notification_sentence"].endswith("October 9, 2026")
    assert facts["page_limit"] == 6
    assert "primary contributor" in facts["primary_contributor_rule"]
    assert facts["first_author_rule"].startswith("Undergraduate students pursuing")
    assert "single-blind" in facts["review_process"]


def test_extract_reads_the_high_school_first_author_rule() -> None:
    facts = venue.extract(venue.to_text(HIGH_SCHOOL_PAGE))
    assert facts["page_limit"] == 5
    assert "high school student" in facts["first_author_rule"]


def test_a_missing_fact_is_none_rather_than_a_guess() -> None:
    facts = venue.extract("nothing useful here")
    assert facts["page_limit"] is None
    assert facts["deadline_sentence"] is None
    assert facts["first_author_rule"] is None


def test_build_writes_the_record_and_keeps_the_human_question_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = {
        venue.PAGES["high_school_symposium"]: HIGH_SCHOOL_PAGE,
        venue.PAGES["undergraduate_reu_consortium"]: UNDERGRADUATE_PAGE,
    }
    monkeypatch.setattr(venue, "fetch", lambda url, timeout=60.0: pages[url])
    record = venue.build(tmp_path)

    assert all(record["consistency_checks"].values())
    assert record["human_confirmation_required"] is True
    assert record["frozen_route_page_limit"] == 6
    assert record["ineligible_route_page_limit"] == 5
    stored = json.loads((tmp_path / "venue.json").read_text())
    assert len(stored["pages"]["undergraduate_reu_consortium"]["page_sha256"]) == 64
    text = (tmp_path / "VENUE.md").read_text()
    assert "Open item that a human must close" in text
    assert "never assign first authorship" in text.lower()
    assert (tmp_path / "undergraduate_reu_consortium.txt").exists()


def test_a_failed_page_fetch_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(url: str, timeout: float = 60.0) -> str:
        raise venue.VenueRetrievalError(f"GET {url} failed: URLError: offline")

    monkeypatch.setattr(venue, "fetch", boom)
    with pytest.raises(venue.VenueRetrievalError, match="failed"):
        venue.build(tmp_path)
    assert not (tmp_path / "venue.json").exists()
