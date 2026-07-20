"""ATS-optimized PDF generation for structured tailored resumes."""

import re
from typing import List, Literal

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from job_compass.models import TailoredExperience, TailoredResume

# Safety net: final data should never carry citations, but strip them if present.
_ANNOTATION_RE = re.compile(
    r"\s*\[(?:SOURCE|BRIDGED):\s*(?:\"[^\"]*\"|'[^']*'|[^\]]+)\]\s*$"
)

# Layout constants (mm). 1pt = 0.3528mm; 0.2in = 5.08mm; 0.15in = 3.81mm.
_LINE_10PT = 4.0  # 10pt * 1.15 spacing
_LINE_9PT = 3.7  # 9pt * 1.15 spacing
_BULLET_INDENT = 5.1
_BLOCK_GAP = 3.8


def _clean_text(text: str) -> str:
    """Strip markdown/citations and drop characters outside cp1252 (e.g. emoji)."""
    text = _ANNOTATION_RE.sub("", text)
    text = re.sub(r"[#*`]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.encode("cp1252", "ignore").decode("cp1252")


class ResumePDF(FPDF):
    """Structured, ATS-friendly resume layout with Helvetica fonts."""

    def __init__(self):
        super().__init__()
        # cp1252 covers bullets (•), en/em dashes, and curly quotes
        self.core_fonts_encoding = "cp1252"
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(255, 255, 255)

    @property
    def content_width(self) -> float:
        """Usable width between margins."""
        return self.w - self.l_margin - self.r_margin

    def _guard_page_break(self, needed: float = 25.0) -> None:
        """Start a new page if less than `needed` mm remains."""
        if self.get_y() > self.page_break_trigger - needed:
            self.add_page()

    def header_section(self, name: str, title: str, contact: str = "") -> None:
        """Add name (18pt bold), title (12pt italic), contact (9pt)."""
        self.set_font("helvetica", "B", 18)
        self.cell(
            0, 8, _clean_text(name), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L"
        )
        self.set_font("helvetica", "I", 12)
        self.cell(
            0, 6, _clean_text(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L"
        )
        if contact:
            self.set_font("helvetica", "", 9)
            self.multi_cell(
                0, 4, _clean_text(contact), new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )
        self.ln(2)

    def section_header(self, title: str) -> None:
        """Add a section header (11pt bold uppercase) with a horizontal line."""
        self.set_font("helvetica", "B", 11)
        self.cell(0, 7, title.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.line(
            self.l_margin, self.get_y(),
            self.l_margin + self.content_width, self.get_y(),
        )
        self.ln(2)

    def body_text(self, text: str, size: int = 10) -> None:
        """Add wrapped body text with 1.15 line spacing."""
        self.set_font("helvetica", "", size)
        line_height = _LINE_10PT if size >= 10 else _LINE_9PT
        self.multi_cell(
            0, line_height, _clean_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )
        self.ln(1)

    def summary_section(self, summary: str) -> None:
        """Add the professional summary (10pt, no bullets)."""
        if not summary.strip():
            return
        self.section_header("Professional Summary")
        self.body_text(summary, size=10)

    def skills_section(self, skills: List[str]) -> None:
        """Add core skills (9pt, inline comma-separated)."""
        if not skills:
            return
        self.section_header("Core Skills")
        self.body_text(", ".join(skills), size=9)

    def _experience_role(self, experience: TailoredExperience) -> None:
        """Render one company block: company, role + dates, bullets."""
        company_line = experience.company
        if experience.location:
            company_line = f"{company_line} | {experience.location}"
        self.set_font("helvetica", "B", 10)
        self.cell(
            0, _LINE_10PT, _clean_text(company_line),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

        self.set_font("helvetica", "I", 10)
        dates = _clean_text(experience.dates)
        dates_width = self.get_string_width(dates) + 2
        self.cell(
            self.content_width - dates_width, _LINE_10PT,
            _clean_text(experience.role),
            new_x=XPos.RIGHT, new_y=YPos.TOP,
        )
        self.cell(
            dates_width, _LINE_10PT, dates, align="R",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.ln(0.5)

        self.set_font("helvetica", "", 9)
        for bullet in experience.bullets[:4]:
            clean = _clean_text(bullet)
            if not clean:
                continue
            self.set_left_margin(self.l_margin + _BULLET_INDENT)
            self.multi_cell(
                0, _LINE_9PT, f"• {clean}", new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )
            self.set_left_margin(self.l_margin - _BULLET_INDENT)
            self.set_x(self.l_margin)
            self.ln(0.8)

    def experience_section(self, experiences: List[TailoredExperience]) -> None:
        """Add professional experience, one block per role with bullets.

        Roles flagged irrelevant still render if they have bullets (capped at
        1-2 by tailoring) so the career timeline has no gaps; roles with no
        bullets are dropped entirely.
        """
        shown = [e for e in experiences if e.bullets]
        if not shown:
            return
        self.section_header("Professional Experience")
        for experience in shown:
            self._guard_page_break()
            self._experience_role(experience)
            self.ln(_BLOCK_GAP)

    def _text_block_section(self, header: str, text: str) -> None:
        """Render a short text block: first line 10pt bold, rest 9pt."""
        lines = [line for line in text.split("\n") if line.strip()]
        if not lines:
            return
        self.section_header(header)
        self.set_font("helvetica", "B", 10)
        self.multi_cell(
            0, _LINE_10PT, _clean_text(lines[0]), new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )
        if len(lines) > 1:
            self.set_font("helvetica", "", 9)
            self.multi_cell(
                0, _LINE_9PT, _clean_text("\n".join(lines[1:])),
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
        self.ln(1)

    def education_section(self, education: str) -> None:
        """Add education: school (10pt bold), degree + dates (9pt)."""
        if education.strip():
            self._text_block_section("Education", education)

    def volunteer_section(self, volunteer: str) -> None:
        """Add volunteer experience (optional)."""
        if volunteer.strip():
            self._text_block_section("Volunteer Experience", volunteer)

    def languages_section(self, languages: List[str]) -> None:
        """Add languages (9pt, inline)."""
        if not languages:
            return
        self.section_header("Languages")
        self.body_text(", ".join(languages), size=9)

    def tools_section(self, tools: List[str]) -> None:
        """Add tools (9pt, inline comma-separated)."""
        if not tools:
            return
        self.section_header("Tools")
        self.body_text(", ".join(tools), size=9)

    def cover_letter_body(self, cover_letter: str) -> None:
        """Add cover letter body."""
        self.body_text(cover_letter, size=10)


def generate_resume_pdf(tailored: TailoredResume) -> bytes:
    """Generate a structured PDF resume from TailoredResume data.

    Args:
        tailored: The structured tailored resume data.

    Returns:
        PDF file bytes.
    """
    pdf = ResumePDF()
    pdf.add_page()
    pdf.header_section(tailored.name, tailored.title, tailored.contact)
    pdf.summary_section(tailored.summary)
    pdf.skills_section(tailored.skills)
    pdf.experience_section(tailored.experience)
    pdf.education_section(tailored.education)
    if tailored.volunteer:
        pdf.volunteer_section(tailored.volunteer)
    pdf.languages_section(tailored.languages)
    pdf.tools_section(tailored.tools)
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
    pdf.header_section(tailored.name, tailored.title, tailored.contact)
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
