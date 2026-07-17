"""Job page scraping utilities."""

import re

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
        raise ValueError(
            "Could not extract meaningful text from URL. "
            "The site may use JavaScript rendering or block scraping. "
            "Try the Paste Text option instead."
        )

    return cleaned
