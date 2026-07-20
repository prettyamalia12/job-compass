"""Tests for structured resume tailoring with anti-hallucination guardrails."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_compass.models import JobParseResult
from job_compass.tailor import (
    MAX_BULLETS_IRRELEVANT_ROLE,
    MAX_BULLETS_PER_ROLE,
    parse_resume_structure,
    tailor_resume,
)


BASE_RESUME = """# OKTAVIA PRETTY AMALIA
**Senior Product Manager**
📧 prettyamalia12@gmail.com | 🌐 prettyamalia.com | 📍 Indonesia (Open to Remote & Relocation)

---

## PROFESSIONAL SUMMARY

Senior Product Manager with 6+ years of experience building gaming and consumer products.

---

## CORE SKILLS

- Product Strategy
- Growth & Experimentation
- LiveOps
- LLMs (Claude, ChatGPT, Cursor)
- A/B Testing

---

## PROFESSIONAL EXPERIENCE

### Bitazza | Bangkok, Thailand (Remote)
**Senior Product Manager** | Feb 2024 – Jun 2026

- Owned the LiveOps and player engagement roadmap for Freedom World, growing the platform to 11.2K monthly active users through missions, quizzes, community goals, and reward systems.
- Led the launch of Freedom Planets, designing player onboarding, progression systems, rewards, and long-term engagement mechanics that scaled to 400+ player-owned planets within three months.
- Led the development of an AI-powered mission creation tool that reduced campaign setup time from one hour to minutes, lowering operational effort by 80%.
- Conducted customer discovery with 15 merchants across three Thai islands, shaping a gamified tourism campaign that increased merchant discovery and app adoption.

### Itsavirus | Bali, Indonesia (Hybrid)
**Product Owner** | Jun 2022 – Sep 2023

- Owned product strategy and launch of a blockchain racing game MVP in six months, defining gameplay loops, player economy, and progression.
- Served as team lead for five engineers, mentoring career development and aligning stakeholder priorities with engineering throughout product delivery.

**Android Engineer** | Jun 2020 – Jun 2022

- Developed a mobility platform for four white-label transportation applications for enterprise clients in the Netherlands.
- Built automated testing frameworks using Espresso and JUnit to improve product quality across multiple applications.

---

## EDUCATION

**Politeknik Caltex Riau** | Indonesia
Bachelor of Applied Science (BASc), Informatics Engineering | 2014 – 2018

---

## LANGUAGES

- **Bahasa Indonesia** — Native
- **English** — C2 Proficient

---

## TOOLS

Jira • Confluence • Notion • SQL • Figma
"""

LIVEOPS_BULLET = (
    "Owned the LiveOps and player engagement roadmap for Freedom World, growing "
    "the platform to 11.2K monthly active users through missions, quizzes, "
    "community goals, and reward systems."
)
PLANETS_BULLET = (
    "Led the launch of Freedom Planets, designing player onboarding, progression "
    "systems, rewards, and long-term engagement mechanics that scaled to 400+ "
    "player-owned planets within three months."
)
RACING_BULLET = (
    "Owned product strategy and launch of a blockchain racing game MVP in six "
    "months, defining gameplay loops, player economy, and progression."
)
MERCHANT_BULLET = (
    "Conducted customer discovery with 15 merchants across three Thai islands, "
    "shaping a gamified tourism campaign that increased merchant discovery and "
    "app adoption."
)


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


def _sourced(text: str, source: str) -> str:
    """Attach a SOURCE citation to a bullet."""
    return f'{text} [SOURCE: "{source}"]'


def _valid_tailor_response() -> dict[str, object]:
    """Return a valid structured LLM response with source citations."""
    return {
        "summary": "Senior Product Manager with 6 years of experience in gaming and AI.",
        "skills": ["Product Strategy", "LiveOps", "Gamification"],
        "experience": [
            {
                "role_index": 0,
                "is_relevant": True,
                "bullets": [
                    _sourced(LIVEOPS_BULLET, LIVEOPS_BULLET),
                    _sourced(PLANETS_BULLET, PLANETS_BULLET),
                ],
            },
            {
                "role_index": 1,
                "is_relevant": True,
                "bullets": [_sourced(RACING_BULLET, RACING_BULLET)],
            },
            {"role_index": 2, "is_relevant": False, "bullets": []},
        ],
        "changes_made": ["Emphasized gaming and AI experience"],
        "warnings": [],
        "estimated_ats_score": 85,
    }


def test_parse_resume_structure():
    """The base resume parses into name, contact, sections, and role hierarchy."""
    parsed = parse_resume_structure(BASE_RESUME)

    assert parsed.name == "OKTAVIA PRETTY AMALIA"
    assert parsed.title == "Senior Product Manager"
    assert "prettyamalia12@gmail.com" in parsed.contact
    assert "6+ years" in parsed.summary
    assert "Product Strategy" in parsed.skills
    assert [r.company for r in parsed.roles] == ["Bitazza", "Itsavirus", "Itsavirus"]
    assert [r.role for r in parsed.roles] == [
        "Senior Product Manager",
        "Product Owner",
        "Android Engineer",
    ]
    assert parsed.roles[0].dates == "Feb 2024 – Jun 2026"
    assert len(parsed.roles[0].bullets) == 4
    assert "Politeknik Caltex Riau" in parsed.education
    assert parsed.languages == ["Bahasa Indonesia — Native", "English — C2 Proficient"]
    assert parsed.tools == ["Jira", "Confluence", "Notion", "SQL", "Figma"]


@pytest.mark.asyncio
async def test_resume_structure_preserved(mock_openai, sample_job):
    """Output preserves the company/role hierarchy of the base resume."""
    _configure_mock(mock_openai, _valid_tailor_response())

    result = await tailor_resume(BASE_RESUME, sample_job)

    assert [e.company for e in result.experience] == ["Bitazza", "Itsavirus", "Itsavirus"]
    assert [e.role for e in result.experience] == [
        "Senior Product Manager",
        "Product Owner",
        "Android Engineer",
    ]
    assert result.experience[0].location == "Bangkok, Thailand (Remote)"
    assert result.name == "OKTAVIA PRETTY AMALIA"
    assert result.title == "Senior Product Manager"
    assert "Politeknik Caltex Riau" in result.education
    assert result.languages
    assert result.tools


@pytest.mark.asyncio
async def test_no_flat_bullet_dump(mock_openai, sample_job):
    """Experience is a role hierarchy, not one flat list of bullets."""
    _configure_mock(mock_openai, _valid_tailor_response())

    result = await tailor_resume(BASE_RESUME, sample_job)

    assert len(result.experience) > 1
    for experience in result.experience:
        assert isinstance(experience.bullets, list)
        assert len(experience.bullets) <= MAX_BULLETS_PER_ROLE


@pytest.mark.asyncio
async def test_max_4_bullets_per_role(mock_openai, sample_job):
    """Roles returning more than 4 bullets are clamped to 4."""
    response = _valid_tailor_response()
    response["experience"][0]["bullets"] = [
        _sourced(f"Tailored rewording {word} of the LiveOps roadmap.", LIVEOPS_BULLET)
        for word in ["one", "two", "three", "four", "five", "six"]
    ]
    _configure_mock(mock_openai, response)

    result = await tailor_resume(BASE_RESUME, sample_job)

    assert len(result.experience[0].bullets) == MAX_BULLETS_PER_ROLE


@pytest.mark.asyncio
async def test_irrelevant_roles_minimized(mock_openai, sample_job):
    """Irrelevant roles keep at most 1-2 bullets."""
    response = _valid_tailor_response()
    response["experience"][2] = {
        "role_index": 2,
        "is_relevant": False,
        "bullets": [
            _sourced(f"{word} take on mobility work.", "Developed a mobility platform for four white-label transportation applications for enterprise clients in the Netherlands.")
            for word in ["First", "Second", "Third"]
        ],
    }
    _configure_mock(mock_openai, response)

    result = await tailor_resume(BASE_RESUME, sample_job)

    android_role = result.experience[2]
    assert android_role.is_relevant is False
    assert len(android_role.bullets) <= MAX_BULLETS_IRRELEVANT_ROLE


@pytest.mark.asyncio
async def test_no_source_annotations_in_final(mock_openai, sample_job):
    """Final output never contains [SOURCE: or [BRIDGED: artifacts."""
    _configure_mock(mock_openai, _valid_tailor_response())

    result = await tailor_resume(BASE_RESUME, sample_job)

    dumped = json.dumps(result.model_dump())
    assert "[SOURCE:" not in dumped
    assert "[BRIDGED:" not in dumped


@pytest.mark.asyncio
async def test_dates_preserved(mock_openai, sample_job):
    """All original employment dates appear unchanged in the output."""
    _configure_mock(mock_openai, _valid_tailor_response())

    result = await tailor_resume(BASE_RESUME, sample_job)

    output_dates = [e.dates for e in result.experience]
    for dates in ["Feb 2024 – Jun 2026", "Jun 2022 – Sep 2023", "Jun 2020 – Jun 2022"]:
        assert dates in output_dates


@pytest.mark.asyncio
async def test_companies_preserved(mock_openai, sample_job):
    """All original companies appear in the output."""
    _configure_mock(mock_openai, _valid_tailor_response())

    result = await tailor_resume(BASE_RESUME, sample_job)

    companies = {e.company for e in result.experience}
    assert {"Bitazza", "Itsavirus"} <= companies


@pytest.mark.asyncio
async def test_missing_source_annotation_raises(mock_openai, sample_job):
    """LLM bullets without citations are rejected before annotation stripping."""
    response = _valid_tailor_response()
    response["experience"][0]["bullets"] = ["Owned the LiveOps roadmap."]
    _configure_mock(mock_openai, response)

    with pytest.raises(ValueError, match="missing \\[SOURCE:"):
        await tailor_resume(BASE_RESUME, sample_job)


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
    bad_response["skills"] = ["Kubernetes", "Quantum Computing"]

    _configure_mock(mock_openai, bad_response)

    result = await tailor_resume(BASE_RESUME, sample_job)

    assert any(
        "Skill not clearly demonstrated" in warning for warning in result.warnings
    )


@pytest.mark.asyncio
async def test_semantic_skill_match(mock_openai, sample_job):
    """Skills with keyword or concept overlap are accepted."""
    response = _valid_tailor_response()
    response["skills"] = ["AI-powered tools", "Gamification"]

    _configure_mock(mock_openai, response)

    result = await tailor_resume(BASE_RESUME, sample_job)

    skill_warnings = [
        warning for warning in result.warnings if "Skill not clearly demonstrated" in warning
    ]
    assert not skill_warnings, f"Unexpected skill warnings: {skill_warnings}"


@pytest.mark.asyncio
async def test_bridged_annotation_accepted(mock_openai, sample_job):
    """[BRIDGED: ...] bullets pass validation and ship clean in final output."""
    response = _valid_tailor_response()
    response["experience"][0]["bullets"] = [
        'Led customer discovery with 15 local merchants, driving consumer marketplace '
        f'adoption. [BRIDGED: "{MERCHANT_BULLET}"]',
    ]

    _configure_mock(mock_openai, response)

    result = await tailor_resume(
        BASE_RESUME, sample_job, gap_bridges=["No explicit beauty industry experience"]
    )

    bullet = result.experience[0].bullets[0]
    assert "15 local merchants" in bullet
    assert "[BRIDGED:" not in bullet
