"""Mock LLM responses for demo/mock mode.

Used when MOCK_MODE=1 or when no KIMI_API_KEY is configured, so the full
UI flow can be exercised without making paid API calls.
"""

from typing import Any, Dict, List, Optional

from job_compass.config import config
from job_compass.models import TailoredExperience, TailoredResume

MOCK_JOB_DESCRIPTION = """Senior Product Manager — LiveOps & Monetization
Nebula Play (Series B gaming studio) — Remote (APAC overlap) — Full-time

About Nebula Play
Nebula Play is a mobile gaming studio behind Starfall Saga (2M+ downloads).
We're hiring a Senior Product Manager to own our live-ops and monetization
roadmap for our flagship title.

What you'll do
- Own the live-ops calendar: events, seasonal content, and in-game economies
- Design and run A/B experiments on monetization funnels (IAP + rewarded ads)
- Partner with 5 engineers, data science, and art to ship weekly releases
- Analyze player behavior with SQL and Amplitude; turn insights into roadmap bets
- Define pricing and progression systems that respect players while growing revenue

Requirements
- 5+ years of product management, 3+ in mobile free-to-play games
- Shipped live-ops systems: events, battle passes, or player economies
- Strong experimentation practice (A/B testing, statistical literacy)
- Comfortable writing SQL to self-serve analysis
- Experience with economy balancing and monetization design

Nice to have
- Experience with AI-assisted tooling or LLM-powered features
- Figma fluency for prototyping flows with designers
- Working knowledge of Unity or Unreal constraints

Compensation
$90,000 - $120,000 USD depending on experience, plus revenue-share bonus.

We offer visa sponsorship for exceptional candidates and are remote-first
within UTC+7 to UTC+12 overlap hours.
"""

MOCK_PARSED: Dict[str, Any] = {
    "title": "Senior Product Manager — LiveOps & Monetization",
    "company": "Nebula Play",
    "location": "Remote (APAC)",
    "remote_policy": "fully_remote",
    "salary_range": "$90,000 - $120,000 USD",
    "employment_type": "full_time",
    "seniority": "senior",
    "description_summary": (
        "Own the live-ops and monetization roadmap for Starfall Saga, a mobile "
        "free-to-play title with 2M+ downloads. You'll design events and player "
        "economies, run monetization experiments, and ship weekly releases with a "
        "cross-functional squad."
    ),
    "key_requirements": [
        "5+ years of product management, 3+ in mobile free-to-play games",
        "Shipped live-ops systems: events, battle passes, or player economies",
        "Strong A/B experimentation practice and statistical literacy",
        "SQL for self-serve analysis",
        "Economy balancing and monetization design experience",
    ],
    "nice_to_have": [
        "AI-assisted tooling or LLM-powered features",
        "Figma fluency for prototyping",
        "Unity or Unreal constraint awareness",
    ],
    "red_flags": [
        "Revenue-share bonus structure may indicate base salary at the lower end",
        "'Respect players while growing revenue' is vague — probe monetization ethics in interview",
    ],
    "green_flags": [
        "Clear live-ops ownership with real autonomy",
        "Salary disclosed up front",
        "Remote-first with explicit timezone range",
        "Visa sponsorship explicitly offered",
        "Small squad (5 engineers) with weekly ship cadence",
    ],
    "visa_mentioned": True,
    "visa_sponsorship": True,
}

MOCK_SCORED: Dict[str, Any] = {
    "match_score": 87,
    "match_reasoning": (
        "Strong fit on the core of the role: live-ops, gamification, and "
        "monetization map directly onto your Freedom Planets launch and "
        "engagement-systems work. The main gap is mobile free-to-play depth — "
        "your background is broader product management rather than pure F2P "
        "games. SQL and experimentation experience line up well."
    ),
    "recommendation": "apply",
    "strength_matches": [
        "Live-ops and gamification systems (11.2K MAU growth at Freedom Planets)",
        "0→1 launches under tight cadences (400+ planets in 3 months)",
        "A/B testing and product analytics practice",
        "Cross-functional leadership of a 5-engineer squad",
        "Monetization strategy and player-economy design",
    ],
    "gap_concerns": [
        "Mobile free-to-play-specific experience not explicit in profile",
        "No mention of Amplitude (profile lists other analytics tooling)",
        "Unity/Unreal familiarity unknown",
    ],
    "questions_to_ask": [
        "What does the live-ops calendar look like today, and what's been the biggest miss?",
        "How is monetization performance measured — and what happened with the last economy rebalance?",
        "What's the split between roadmap ownership vs. stakeholder-driven requests?",
    ],
}


def mock_scrape_job_page(url: str) -> str:
    """Pretend to scrape — return a canned job description regardless of URL."""
    return MOCK_JOB_DESCRIPTION


def mock_parse_job_description(text: str) -> Dict[str, Any]:
    """Return canned structured parse, lightly echoing the input's company if named."""
    result = dict(MOCK_PARSED)
    return result


def mock_score_job_fit(
    job_data: Dict[str, Any], profile: Dict[str, Any]
) -> Dict[str, Any]:
    """Return canned scoring result."""
    return dict(MOCK_SCORED)


def mock_tailor_resume(
    base_resume: str,
    job_data: Any,
    tone: str = "professional",
    max_length_pages: int = 2,
    gap_bridges: Optional[List[str]] = None,
) -> TailoredResume:
    """Return a canned tailored resume for the demo job."""
    name = config.profile.name
    title = config.profile.title
    return TailoredResume(
        name=name,
        title=title,
        contact="jakarta, indonesia · prettyamalia.com",
        summary=(
            "Senior Product Manager with 6 years building live-ops, gamification, "
            "and monetization systems. Launched Freedom Planets to 400+ planets in "
            "3 months and grew engagement 11.2K MAU. Looking to own live-ops "
            "roadmaps for free-to-play titles that respect players."
        ),
        skills=[
            "LiveOps & Event Design",
            "Monetization Strategy",
            "A/B Testing",
            "SQL",
            "Product Analytics (Amplitude)",
            "Player Economy Balancing",
            "Roadmapping",
            "Stakeholder Management",
            "Figma",
            "Jira",
        ],
        experience=[
            TailoredExperience(
                company="Freedom Planets",
                location="Remote",
                role="Senior Product Manager",
                dates="2022 - Present",
                bullets=[
                    "Owned live-ops calendar shipping weekly events and seasonal content across 400+ planets",
                    "Designed progression and reward systems that grew monthly active users by 11.2K",
                    "Ran 30+ A/B experiments on onboarding and retention funnels; lifted D7 retention 14%",
                    "Partnered with 5 engineers and data science to rebalance the in-game economy each quarter",
                ],
                is_relevant=True,
                original_bullets=[
                    "Launched Freedom Planets feature set",
                    "Grew MAU",
                ],
            ),
            TailoredExperience(
                company="PT Kreatif Teknologi",
                location="Jakarta, Indonesia",
                role="Product Manager",
                dates="2020 - 2022",
                bullets=[
                    "Shipped AI-powered internal tooling that reduced operational effort by 80%",
                    "Defined roadmap and wrote SQL-backed analyses prioritizing the experiment backlog",
                ],
                is_relevant=True,
                original_bullets=["Shipped internal tools"],
            ),
        ],
        education="B.Sc. Computer Science — University of Indonesia",
        languages=["English (fluent)", "Indonesian (native)"],
        tools=["SQL", "Amplitude", "Figma", "Jira", "Notion"],
        cover_letter=(
            f"Hi Nebula Play team,\n\n"
            f"I'm excited about the Senior Product Manager, LiveOps & Monetization role. "
            f"At Freedom Planets I owned a live-ops cadence shipping weekly events across "
            f"400+ planets and grew MAU by 11.2K through progression redesigns — the exact "
            f"loop your Starfall Saga roadmap describes.\n\n"
            f"I haven't worked on mobile free-to-play specifically, but my economy balancing, "
            f"experimentation practice, and SQL depth transfer directly. I'd love to walk you "
            f"through how I'd approach the first 90 days of your live-ops calendar.\n\n"
            f"Best,\n{name}"
        ),
        changes_made=[
            "Surfaced LiveOps and monetization bullets to the top role",
            "Added Amplitude and player-economy keywords from the job description",
            "Rewrote summary to lead with live-ops ownership and quantified growth",
            "Marked earlier non-gaming roles as minimized",
        ],
        warnings=[
            "No direct mobile free-to-play title in base resume",
            "Amplitude not evidenced in base resume — verify before submitting",
        ],
        estimated_ats_score=82,
        created_at="2026-10-30T00:00:00",
    )


def mock_cover_letter(job_data: Any, base_resume: str, tone: str = "professional") -> str:
    """Return a canned gap-bridging cover letter."""
    return mock_tailor_resume(base_resume, job_data, tone).cover_letter
