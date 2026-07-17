"""Tests for the job description parser."""

import pytest

from job_compass.parser import parse_job_description


@pytest.mark.asyncio
async def test_parse_sample_description():
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

    result = await parse_job_description(sample)

    assert result["title"] == "Senior Product Manager - AI Gaming"
    assert result["seniority"] == "senior"
    assert result["remote_policy"] in ["fully_remote", "remote"]
    assert result["visa_sponsorship"] is True
    assert len(result["key_requirements"]) >= 3
