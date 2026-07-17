"""Tests for the job description parser."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_compass.parser import parse_job_description


@pytest.fixture
def mock_parser_openai():
    """Patch the parser's OpenAI client so tests never hit the API."""
    with patch(
        "job_compass.parser.client.chat.completions.create",
        new_callable=AsyncMock,
    ) as mock_chat:
        yield mock_chat


def _parser_response(content: dict) -> MagicMock:
    """Build a mock ChatCompletion response with the given JSON content."""
    message = MagicMock()
    message.content = json.dumps(content)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.mark.asyncio
async def test_parse_sample_description(mock_parser_openai):
    """Test parsing a simple job description."""
    sample = """
    Senior Product Manager - AI Gaming

    We are looking for a Senior Product Manager to lead our AI-powered gaming platform.
    You will own the product roadmap, drive user engagement, and work with LLMs to create
    personalized player experiences.

    Requirements:
    - 5+ years of product management experience
    - Experience with AI/ML products
    - Gaming industry background preferred
    - Strong data analytics skills

    Location: Remote (US timezone)
    Salary: $140,000 - $180,000
    Visa sponsorship: Available
    """

    mock_parser_openai.return_value = _parser_response({
        "title": "Senior Product Manager - AI Gaming",
        "company": "AI Gaming Inc",
        "location": "Remote",
        "remote_policy": "fully_remote",
        "salary_range": "$140,000 - $180,000",
        "employment_type": "full_time",
        "seniority": "senior",
        "description_summary": "Senior PM role for AI gaming platform.",
        "key_requirements": [
            "5+ years of product management experience",
            "Experience with AI/ML products",
            "Gaming industry background preferred",
            "Strong data analytics skills",
        ],
        "nice_to_have": [],
        "red_flags": [],
        "green_flags": [],
        "visa_mentioned": True,
        "visa_sponsorship": True,
    })

    result = await parse_job_description(sample)

    assert result["title"] == "Senior Product Manager - AI Gaming"
    assert result["seniority"] == "senior"
    assert result["remote_policy"] in ["fully_remote", "remote"]
    assert result["visa_sponsorship"] is True
    assert len(result["key_requirements"]) >= 3
