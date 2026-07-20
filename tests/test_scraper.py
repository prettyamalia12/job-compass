"""Tests for Next.js RSC payload extraction in the scraper."""

import json

from job_compass.scraper import _extract_nextjs_job


def _push(chunk_id: int, content: str) -> str:
    """Build a script tag containing one RSC payload push."""
    return f"<script>self.__next_f.push([{chunk_id},{json.dumps(content)}])</script>"


def _fixture_html() -> str:
    """Minimal Next.js page with one embedded job posting."""
    job = (
        '5:{"jobs":[{"job_req_id":"TEST123","instance_id":"TEST123-1",'
        '"job_title":"Product Manager",'
        '"hiring_organization":{"name":"Acme","url":""},'
        '"job_description":"$a",'
        '"job_location":{"address":{"city":"Singapore","region":"",'
        '"country":"Singapore"}},'
        '"employment_type":"Full time","date_posted":"2026-07-20"}]}'
    )
    description = "a:T1000,<p><b>Own the roadmap.</b></p><p>Ship products.</p>"
    return (
        "<html><head><title>Jobs</title></head><body><div id=\"root\"></div>"
        f"{_push(1, job)}{_push(2, description)}"
        "</body></html>"
    )


def test_extract_nextjs_job():
    """Job metadata and description are extracted from RSC payloads."""
    text = _extract_nextjs_job(
        _fixture_html(), "https://jobs.example.com/jobs/TEST123-1"
    )

    assert "Product Manager" in text
    assert "Acme" in text
    assert "Singapore" in text
    assert "Full time" in text
    assert "Posted: 2026-07-20" in text
    assert "Own the roadmap." in text
    assert "Ship products." in text


def test_extract_nextjs_job_no_match():
    """Unknown instance ids yield no extraction."""
    text = _extract_nextjs_job(
        _fixture_html(), "https://jobs.example.com/jobs/NOPE-9"
    )

    assert text == ""


def test_extract_nextjs_job_no_payload():
    """Pages without RSC payloads yield no extraction."""
    assert _extract_nextjs_job("<html><body>Hi</body></html>", "https://x.co/jobs/A-1") == ""
