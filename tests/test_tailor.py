"""Tests for resume tailoring with anti-hallucination guardrails."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_compass.models import JobParseResult
from job_compass.tailor import tailor_resume


BASE_RESUME = """
# Oktavia Pretty Amalia

## Experience
- Led Freedom Planets launch, creating 400+ planets in 3 months and reaching 11.2K MAU.
- Built AI-powered internal tools that reduced operational effort by 80%.
- Created AI-powered mission creation tool for Freedom Planets quests.
- Automated AI-powered KYC review tool for onboarding flows.
- Conducted customer discovery with 15 merchants across three Thai islands.
- 6 years of product management experience in gaming and AI.
- Managed cross-functional team of 5 engineers.

## Skills
Product Strategy, Gamification, LiveOps, Growth & Experimentation, AI/LLMs,
Claude, ChatGPT, Cursor, A/B Testing, Customer Discovery, Product Analytics,
SQL, Roadmapping, Stakeholder Management.
"""


@pytest.fixture
def sample_job() -> JobParseResult:
    """Sample parsed job data for tailoring tests."""
    return JobParseResult(
        title="Senior Product Manager",
        company="Example Games",
        location="Remote",
        remote_policy="fully_remote",
        employment_type="full_time",
        seniority="senior",
        description_summary="Lead product for a mobile gaming platform.",
        key_requirements=[
            "6+ years of product management experience",
            "Gaming industry background",
            "AI/ML experience",
        ],
        nice_to_have=["Gamification experience"],
        red_flags=[],
        green_flags=[],
        visa_mentioned=False,
        visa_sponsorship=None,
        match_score=85,
        match_reasoning="Strong fit for gaming and AI background.",
        recommendation="apply",
        strength_matches=[],
        gap_concerns=[],
        questions_to_ask=[],
        raw_description="Job description text",
        parsed_at="2026-01-01T00:00:00",
    )


@pytest.fixture
def mock_openai():
    """Patch OpenAI chat and embedding clients so tests never hit the API."""
    with (
        patch("job_compass.tailor.client.chat.completions.create") as mock_chat,
        patch("job_compass.tailor._get_embeddings") as mock_embed,
    ):
        mock_embed.return_value = None
        yield mock_chat


def _response(content: str) -> MagicMock:
    """Build a mock ChatCompletion response with the given content."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _configure_mock(mock_chat: AsyncMock, tailor_response: dict[str, object]) -> None:
    """Set up the mock to return the tailor JSON or a cover letter string."""
    cover_letter = "Dear hiring manager, I am excited about this role. Sincerely, Oktavia."

    async def side_effect(*args, **kwargs) -> MagicMock:
        messages = kwargs.get("messages", [])
        system_prompt = messages[0].get("content", "") if messages else ""
        if "Return ONLY a JSON object" in system_prompt:
            return _response(json.dumps(tailor_response))
        return _response(cover_letter)

    mock_chat.side_effect = side_effect


def _valid_tailor_response() -> dict[str, object]:
    """Return a valid LLM response with source citations."""
    return {
        "summary": "Senior PM with 6 years of experience in gaming and AI.",
        "tailored_bullets": [
            'Led Freedom Planets launch, creating 400+ planets in 3 months and reaching 11.2K MAU. '
            '[SOURCE: "Led Freedom Planets launch, creating 400+ planets in 3 months and reaching 11.2K MAU."]',
            'Built AI-powered internal tools that reduced operational effort by 80%. '
            '[SOURCE: "Built AI-powered internal tools that reduced operational effort by 80%."]',
        ],
        "skills_section": ["Product Strategy", "Gamification", "AI/LLMs"],
        "changes_made": ["Emphasized gaming and AI experience"],
        "warnings": [],
        "estimated_ats_score": 85,
    }


@pytest.mark.asyncio
async def test_every_bullet_has_source(mock_openai, sample_job):
    """Every tailored bullet must include a [SOURCE: ...] citation."""
    _configure_mock(mock_openai, _valid_tailor_response())

    result = await tailor_resume(BASE_RESUME, sample_job)

    assert len(result.tailored_bullets) > 0
    for bullet in result.tailored_bullets:
        assert "[SOURCE:" in bullet, f"Missing SOURCE annotation: {bullet}"


@pytest.mark.asyncio
async def test_no_hallucinated_metrics(mock_openai, sample_job):
    """Validation rejects numbers that do not appear in the base resume."""
    bad_response = _valid_tailor_response()
    bad_response["summary"] = "Senior PM with 10 years of experience."  # 10 not in resume

    _configure_mock(mock_openai, bad_response)

    with pytest.raises(ValueError, match="Hallucinated numbers"):
        await tailor_resume(BASE_RESUME, sample_job)


@pytest.mark.asyncio
async def test_skills_not_added(mock_openai, sample_job):
    """Skills not demonstrated in the resume are flagged as warnings, not rejected."""
    bad_response = _valid_tailor_response()
    bad_response["skills_section"] = ["Kubernetes", "Quantum Computing"]

    _configure_mock(mock_openai, bad_response)

    result = await tailor_resume(BASE_RESUME, sample_job)

    assert any(
        "Skill not clearly demonstrated" in warning for warning in result.warnings
    )


@pytest.mark.asyncio
async def test_semantic_skill_match(mock_openai, sample_job):
    """Skills with keyword or concept overlap are accepted."""
    response = _valid_tailor_response()
    response["skills_section"] = ["AI-powered tools", "Gamification"]

    _configure_mock(mock_openai, response)

    result = await tailor_resume(BASE_RESUME, sample_job)

    skill_warnings = [
        warning for warning in result.warnings if "Skill not clearly demonstrated" in warning
    ]
    assert not skill_warnings, f"Unexpected skill warnings: {skill_warnings}"


@pytest.mark.asyncio
async def test_bridged_annotation_accepted(mock_openai, sample_job):
    """Bullets with [BRIDGED: ...] annotations pass validation."""
    response = _valid_tailor_response()
    response["tailored_bullets"] = [
        'Led customer discovery with 15 local merchants, driving consumer marketplace adoption. '
        '[BRIDGED: "Conducted customer discovery with 15 merchants across three Thai islands."]',
    ]

    _configure_mock(mock_openai, response)

    result = await tailor_resume(BASE_RESUME, sample_job, gap_bridges=["No explicit beauty industry experience"])

    assert len(result.tailored_bullets) == 1
    assert "[BRIDGED:" in result.tailored_bullets[0]
