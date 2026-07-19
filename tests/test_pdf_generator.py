"""Tests for PDF generation from TailoredResume data."""

from io import BytesIO

import pytest
from pypdf import PdfReader

from job_compass.models import TailoredResume
from job_compass.pdf_generator import generate_cover_letter_pdf, generate_pdf, generate_resume_pdf


@pytest.fixture
def sample_tailored_resume() -> TailoredResume:
    """Sample tailored resume data for PDF generation."""
    return TailoredResume(
        original_bullets=[
            "- Led Freedom Planets launch, creating 400+ planets in 3 months and reaching 11.2K MAU.",
            "- Built AI-powered internal tools that reduced operational effort by 80%.",
        ],
        tailored_bullets=[
            'Led Freedom Planets launch, creating 400+ planets in 3 months and reaching 11.2K MAU. '
            '[SOURCE: "Led Freedom Planets launch, creating 400+ planets in 3 months and reaching 11.2K MAU."]',
            'Built AI-powered internal tools that reduced operational effort by 80%. '
            '[SOURCE: "Built AI-powered internal tools that reduced operational effort by 80%."]',
        ],
        skills_section=["Product Strategy", "Gamification", "AI/LLMs"],
        summary="Senior PM with 6 years of experience in gaming and AI.",
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
    """Resume PDF contains name, title, and sections."""
    pdf_bytes = generate_resume_pdf(sample_tailored_resume)

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000

    text = _extract_text(pdf_bytes)
    assert "Oktavia Pretty Amalia" in text
    assert "Senior Product Manager" in text
    assert "SUMMARY" in text
    assert "SKILLS" in text
    assert "EXPERIENCE" in text
    assert "Senior PM with 6 years of experience" in text
    assert "Product Strategy" in text
    assert "Freedom Planets" in text


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
    assert "EXPERIENCE" in resume_text
    assert "Dear hiring manager" in cover_text


def test_generate_pdf_invalid_type(sample_tailored_resume):
    """Invalid PDF type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown PDF type"):
        generate_pdf(sample_tailored_resume, "invalid")
