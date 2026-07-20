"""Tests for PDF generation from structured TailoredResume data."""

from io import BytesIO

import pytest
from pypdf import PdfReader

from job_compass.models import TailoredExperience, TailoredResume
from job_compass.pdf_generator import generate_cover_letter_pdf, generate_pdf, generate_resume_pdf


@pytest.fixture
def sample_experience() -> list[TailoredExperience]:
    """Sample structured experience blocks."""
    return [
        TailoredExperience(
            company="Bitazza",
            location="Bangkok, Thailand (Remote)",
            role="Senior Product Manager",
            dates="Feb 2024 – Jun 2026",
            bullets=[
                "Owned the LiveOps and player engagement roadmap, growing the platform to 11.2K monthly active users.",
                "Led the launch of Freedom Planets, scaling to 400+ player-owned planets within three months.",
            ],
            is_relevant=True,
            original_bullets=[
                "Owned the LiveOps and player engagement roadmap for Freedom World.",
            ],
        ),
        TailoredExperience(
            company="Itsavirus",
            location="Bali, Indonesia (Hybrid)",
            role="Product Owner",
            dates="Jun 2022 – Sep 2023",
            bullets=[
                "Owned product strategy and launch of a blockchain racing game MVP in six months.",
            ],
            is_relevant=True,
            original_bullets=[],
        ),
        TailoredExperience(
            company="Mapid.io",
            location="Bandung, Indonesia (On-site)",
            role="Android Engineer",
            dates="Feb 2019 – Aug 2019",
            bullets=[],
            is_relevant=False,
            original_bullets=["Built Android applications using Java and Mapbox."],
        ),
    ]


@pytest.fixture
def sample_tailored_resume(sample_experience) -> TailoredResume:
    """Sample structured tailored resume data for PDF generation."""
    return TailoredResume(
        name="Oktavia Pretty Amalia",
        title="Senior Product Manager",
        contact="prettyamalia12@gmail.com | prettyamalia.com | Indonesia",
        summary="Senior PM with 6 years of experience in gaming and AI.",
        skills=["Product Strategy", "Gamification", "AI/LLMs"],
        experience=sample_experience,
        education=(
            "Politeknik Caltex Riau | Indonesia\n"
            "Bachelor of Applied Science (BASc), Informatics Engineering | 2014 – 2018"
        ),
        volunteer="Stella's Child | Bali, Indonesia\nIT Education Mentor | Oct – Dec 2022",
        languages=["Bahasa Indonesia — Native", "English — C2 Proficient"],
        tools=["Jira", "Confluence", "SQL", "Figma"],
        cover_letter="Dear hiring manager, I am excited about this role. Sincerely, Oktavia.",
        changes_made=["Emphasized gaming and AI experience"],
        warnings=[],
        estimated_ats_score=85,
        created_at="2026-01-01T00:00:00",
    )


def _extract_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_generate_resume_pdf(sample_tailored_resume):
    """Resume PDF contains the full structured resume."""
    pdf_bytes = generate_resume_pdf(sample_tailored_resume)

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000

    text = _extract_text(pdf_bytes)
    # Header
    assert "Oktavia Pretty Amalia" in text
    assert "Senior Product Manager" in text
    assert "prettyamalia12@gmail.com" in text
    # Sections
    assert "PROFESSIONAL SUMMARY" in text
    assert "CORE SKILLS" in text
    assert "PROFESSIONAL EXPERIENCE" in text
    assert "EDUCATION" in text
    assert "VOLUNTEER EXPERIENCE" in text
    assert "LANGUAGES" in text
    assert "TOOLS" in text
    # Content
    assert "Senior PM with 6 years of experience" in text
    assert "Product Strategy" in text
    assert "Bitazza" in text
    assert "Bangkok, Thailand (Remote)" in text
    assert "Feb 2024 – Jun 2026" in text
    assert "Freedom Planets" in text
    assert "11.2K" in text
    assert "Politeknik Caltex Riau" in text
    assert "Bahasa Indonesia" in text
    assert "Jira" in text
    # Bullet markers render
    assert "•" in text


def test_resume_pdf_drops_irrelevant_roles(sample_tailored_resume):
    """Roles with no bullets are dropped from the PDF entirely."""
    text = _extract_text(generate_resume_pdf(sample_tailored_resume))

    assert "Mapid.io" not in text
    assert "Bitazza" in text
    assert "Itsavirus" in text


def test_resume_pdf_minimizes_irrelevant_roles(sample_tailored_resume):
    """Irrelevant roles with 1-2 bullets still render to avoid timeline gaps."""
    sample_tailored_resume.experience[2].bullets = [
        "Built Android applications using Java and Mapbox for a geospatial social platform.",
    ]

    text = _extract_text(generate_resume_pdf(sample_tailored_resume))

    assert "Mapid.io" in text
    assert "Java and Mapbox" in text


def test_resume_pdf_max_two_pages(sample_tailored_resume):
    """The resume stays within the 2-page limit."""
    pdf_bytes = generate_resume_pdf(sample_tailored_resume)
    reader = PdfReader(BytesIO(pdf_bytes))

    assert len(reader.pages) <= 2


def test_resume_pdf_strips_annotations(sample_tailored_resume):
    """SOURCE/BRIDGED annotations never leak into the PDF."""
    sample_tailored_resume.experience[0].bullets[0] += ' [SOURCE: "Owned the LiveOps roadmap."]'

    text = _extract_text(generate_resume_pdf(sample_tailored_resume))

    assert "[SOURCE:" not in text
    assert "Owned the LiveOps" in text


def test_generate_cover_letter_pdf(sample_tailored_resume):
    """Cover letter PDF contains name and cover letter body."""
    pdf_bytes = generate_cover_letter_pdf(sample_tailored_resume)

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000

    text = _extract_text(pdf_bytes)
    assert "Oktavia Pretty Amalia" in text
    assert "Dear hiring manager" in text
    assert "Sincerely, Oktavia" in text


def test_generate_pdf_dispatch(sample_tailored_resume):
    """generate_pdf dispatches to the correct generator."""
    resume_pdf = generate_pdf(sample_tailored_resume, "resume")
    cover_pdf = generate_pdf(sample_tailored_resume, "cover_letter")

    assert resume_pdf.startswith(b"%PDF")
    assert cover_pdf.startswith(b"%PDF")

    resume_text = _extract_text(resume_pdf)
    cover_text = _extract_text(cover_pdf)
    assert "PROFESSIONAL EXPERIENCE" in resume_text
    assert "Dear hiring manager" in cover_text


def test_generate_pdf_invalid_type(sample_tailored_resume):
    """Invalid PDF type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown PDF type"):
        generate_pdf(sample_tailored_resume, "invalid")
