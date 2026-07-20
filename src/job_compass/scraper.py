"""Job page scraping utilities."""

import json
import re
from typing import Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

from job_compass.config import config


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _clean_text(text: str) -> str:
    """Normalize whitespace and cap length."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text[: config.max_scrape_chars]


def _extract_main_content(soup: BeautifulSoup) -> str:
    """Try to find the main job content area."""
    # Priority order for content selectors
    selectors = [
        ("main", None),
        ("article", None),
        ("div", re.compile(r"job|description|content|posting|details", re.I)),
        ("div", re.compile(r"main|body", re.I)),
    ]

    for tag, class_pattern in selectors:
        if class_pattern:
            elem = soup.find(tag, class_=class_pattern) or soup.find(tag, id=class_pattern)
        else:
            elem = soup.find(tag)
        if elem:
            return elem.get_text(separator="\n", strip=True)

    return soup.get_text(separator="\n", strip=True)


_RSC_CHUNK_RE = re.compile(r'self\.__next_f\.push\(\[\d+,"((?:[^"\\]|\\.)*)"\]\)')
_RSC_TEXT_ROW_RE = re.compile(r'(?<![0-9a-zA-Z":])([0-9a-f]{1,4}):T([0-9a-f]{1,6}),')


def _rsc_stream(html: str) -> str:
    """Join decoded Next.js RSC payload chunks in document order."""
    chunks: List[str] = []
    for match in _RSC_CHUNK_RE.finditer(html):
        try:
            chunks.append(json.loads(f'"{match.group(1)}"'))
        except json.JSONDecodeError:
            continue
    return "".join(chunks)


def _rsc_text_rows(stream: str) -> Dict[str, str]:
    """Map RSC text-row ids to their HTML content."""
    rows: Dict[str, str] = {}
    for match in _RSC_TEXT_ROW_RE.finditer(stream):
        content = stream[match.end() : match.end() + int(match.group(2), 16)]
        # The declared length counts wire-format (escaped) bytes, so the
        # decoded slice can overrun into the next row marker — truncate it.
        next_row = _RSC_TEXT_ROW_RE.search(content)
        if next_row:
            content = content[: next_row.start()]
        rows[match.group(1)] = content
    return rows


def _find_json_object(stream: str, key_pos: int, max_span: int = 5000) -> Optional[dict]:
    """Extract the balanced-brace JSON object containing key_pos."""
    start = stream.rfind("{", 0, key_pos)
    if start == -1:
        return None
    depth = 0
    for i in range(start, min(len(stream), key_pos + max_span)):
        if stream[i] == "{":
            depth += 1
        elif stream[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(stream[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _extract_nextjs_job(html: str, url: str) -> str:
    """Extract the target job posting from Next.js RSC payloads.

    Next.js app-router career sites embed content in
    ``self.__next_f.push([id, "..."])`` scripts instead of rendered HTML.
    The job detail is a JSON object keyed by the URL's instance id, with the
    description in a referenced text row.

    Args:
        html: Raw page HTML.
        url: Job posting URL (last path segment is the instance id).

    Returns:
        Job posting text, or "" if no matching payload was found.
    """
    stream = _rsc_stream(html)
    if not stream:
        return ""

    instance_id = url.rstrip("/").rsplit("/", 1)[-1]
    pos = stream.find(f'"instance_id":"{instance_id}"')
    if pos == -1:
        return ""

    job = _find_json_object(stream, pos)
    if not job:
        return ""

    desc_text = ""
    desc_ref = str(job.get("job_description", "")).lstrip("$")
    if desc_ref:
        desc_html = _rsc_text_rows(stream).get(desc_ref, "")
        if desc_html:
            desc_text = BeautifulSoup(desc_html, "html.parser").get_text(
                "\n", strip=True
            )

    address = (job.get("job_location") or {}).get("address") or {}
    location = ", ".join(
        part
        for part in [address.get("city"), address.get("region"), address.get("country")]
        if part
    )
    lines = [
        str(job.get("job_title", "")).strip(),
        str((job.get("hiring_organization") or {}).get("name", "")).strip(),
        location,
        str(job.get("employment_type", "")).strip(),
        f"Posted: {job['date_posted']}" if job.get("date_posted") else "",
    ]
    parts = [line for line in lines if line]
    if desc_text:
        parts.append(desc_text)
    return "\n\n".join(parts)


async def scrape_job_page(url: str) -> str:
    """Fetch and extract clean text from a job posting URL.

    Args:
        url: The job posting URL.

    Returns:
        Cleaned text content of the job description.

    Raises:
        httpx.HTTPStatusError: If the request fails.
        ValueError: If extracted text is too short.
    """
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=config.scrape_timeout_seconds
    ) as client:
        response = await client.get(url, headers=HEADERS)
        response.raise_for_status()
        html = response.text

    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = _extract_main_content(soup)
    cleaned = _clean_text(text)

    if len(cleaned) < 200:
        # JS-rendered Next.js pages embed content in RSC payload scripts,
        # which the soup extraction above removed with the <script> tags.
        cleaned = _clean_text(_extract_nextjs_job(html, url))

    if len(cleaned) < 200:
        raise ValueError(
            "Could not extract meaningful text from URL. "
            "The site may use JavaScript rendering or block scraping. "
            "Try the Paste Text option instead."
        )

    return cleaned
