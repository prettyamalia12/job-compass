"""ATS-optimized PDF generation for tailored resumes and cover letters."""

import io
import re
from typing import List, Literal

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from job_compass.config import config
from job_compass.models import TailoredResume


def _latin1_safe(text: str) -> str:
    """Replace characters not supported by Helvetica with ASCII equivalents."""
    replacements = {
        "•": "-",
        "—": "-",
        "–": "-",
        """: "\"",
        """: "\"",
        "'": "'",
        "’": "'",
        "…": "...",
        "→": "->",
        "←": "<-",
        "↑": "^",
        "↓": "v",
        "×": "x",
        "·": "-",
        "◦": "-",
        "✓": "v",
        "✔": "v",
        "✗": "x",
        "✘": "x",
        "★": "*",
        "☆": "*",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


class ResumePDF(FPDF):
    """Simple, ATS-friendly PDF layout with Helvetica fonts."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(255, 255, 255)

    def _clean_text(self, text: str) -> str:
        """Remove markdown/source citations and normalize whitespace."""
        text = re.sub(r'\[SOURCE:\s*"[^"]+"\]', "", text)
        text = re.sub(r"[#*`]+", "", text)
        return _latin1_safe(text.strip())

    def header_section(self, name: str, title: str) -> None:
        """Add name and title header."""
        self.set_font("helvetica", "B", 18)
        self.cell(
            0,
            10,
            self._clean_text(name),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            align="L",
        )
        self.set_font("helvetica", "I", 12)
        self.cell(
            0,
            8,
            self._clean_text(title),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            align="L",
        )
        self.ln(4)

    def section_header(self, title: str) -> None:
        """Add a section header with a horizontal line."""
        self.set_font("helvetica", "B", 11)
        self.cell(
            0,
            8,
            title.upper(),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(2)

    def body_text(self, text: str) -> None:
        """Add body text with 1.15 line spacing."""
        self.set_font("helvetica", "", 10)
        self.multi_cell(0, 5.75, self._clean_text(text))
        self.ln(1)

    def skills_section(self, skills: List[str]) -> None:
        """Add comma-separated skills."""
        self.section_header("Skills")
        self.body_text(", ".join(skills))

    def summary_section(self, summary: str) -> None:
        """Add professional summary."""
        self.section_header("Summary")
        self.body_text(summary)

    def experience_section(self, bullets: List[str]) -> None:
        """Add experience bullets with - markers."""
        self.section_header("Experience")
        for bullet in bullets:
            clean = self._clean_text(bullet)
            if not clean:
                continue
            self.set_font("helvetica", "", 10)
            self.multi_cell(0, 5.75, f"- {clean}")
            self.ln(1)

    def cover_letter_body(self, cover_letter: str) -> None:
        """Add cover letter body."""
        self.body_text(cover_letter)


def generate_resume_pdf(tailored: TailoredResume) -> bytes:
    """Generate a PDF resume from TailoredResume data.

    Args:
        tailored: The tailored resume data.

    Returns:
        PDF file bytes.
    """
    pdf = ResumePDF()
    pdf.add_page()
    pdf.header_section(config.profile.name, config.profile.title)
    pdf.summary_section(tailored.summary)
    pdf.skills_section(tailored.skills_section)
    pdf.experience_section(tailored.tailored_bullets)
    return bytes(pdf.output())


def generate_cover_letter_pdf(tailored: TailoredResume) -> bytes:
    """Generate a PDF cover letter from TailoredResume data.

    Args:
        tailored: The tailored resume data containing the cover letter.

    Returns:
        PDF file bytes.
    """
    pdf = ResumePDF()
    pdf.add_page()
    pdf.header_section(config.profile.name, config.profile.title)
    pdf.cover_letter_body(tailored.cover_letter)
    return bytes(pdf.output())


def generate_pdf(tailored: TailoredResume, pdf_type: Literal["resume", "cover_letter"]) -> bytes:
    """Generate a PDF of the specified type.

    Args:
        tailored: The tailored resume data.
        pdf_type: Either "resume" or "cover_letter".

    Returns:
        PDF file bytes.

    Raises:
        ValueError: If pdf_type is not "resume" or "cover_letter".
    """
    if pdf_type == "resume":
        return generate_resume_pdf(tailored)
    if pdf_type == "cover_letter":
        return generate_cover_letter_pdf(tailored)
    raise ValueError(f"Unknown PDF type: {pdf_type}")
