"""Pydantic data models for API requests and responses."""

from typing import List, Optional

from pydantic import BaseModel, HttpUrl, Field


class JobUrlInput(BaseModel):
    """Request: analyze a job by URL."""

    url: HttpUrl = Field(..., description="URL of the job posting")


class JobParseResult(BaseModel):
    """Response: structured job analysis with match score."""

    # Parsed fields
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: str = Field(..., description="Job location")
    remote_policy: str = Field(..., description="fully_remote, hybrid, onsite, or unknown")
    salary_range: Optional[str] = Field(None, description="Salary range if disclosed")
    employment_type: str = Field(..., description="full_time, contract, freelance, or unknown")
    seniority: str = Field(..., description="senior, lead, staff, principal, manager, or unknown")
    description_summary: str = Field(..., description="2-3 sentence summary of the role")
    key_requirements: List[str] = Field(default_factory=list, description="Must-have skills")
    nice_to_have: List[str] = Field(default_factory=list, description="Bonus skills")
    red_flags: List[str] = Field(default_factory=list, description="Potential concerns")
    green_flags: List[str] = Field(default_factory=list, description="Positive signals")
    visa_mentioned: bool = Field(..., description="Whether visa is mentioned at all")
    visa_sponsorship: Optional[bool] = Field(
        None, description="True if offered, False if not, None if not mentioned"
    )

    # Scored fields
    match_score: int = Field(..., ge=0, le=100, description="0-100 fit score")
    match_reasoning: str = Field(..., description="Explanation of the score")
    recommendation: str = Field(
        ..., description="strong_apply, apply, consider, or skip"
    )
    strength_matches: List[str] = Field(
        default_factory=list, description="Candidate strengths that align"
    )
    gap_concerns: List[str] = Field(
        default_factory=list, description="Areas where candidate is weak"
    )
    questions_to_ask: List[str] = Field(
        default_factory=list, description="Good interview questions"
    )

    # Metadata
    raw_description: str = Field(..., description="Truncated original text")
    parsed_at: str = Field(..., description="ISO timestamp of analysis")


class ResumeTailorInput(BaseModel):
    """Request: tailor a resume for a specific job."""

    job_url: Optional[str] = Field(
        None,
        description="Original job posting URL. Required for caching/persistence.",
    )
    job_data: JobParseResult = Field(..., description="Structured job analysis")
    base_resume_markdown: str = Field(..., description="Candidate resume in Markdown")
    tone: str = Field("professional", description="Tone of generated content")
    max_length_pages: int = Field(2, ge=1, le=5, description="Target resume length in pages")
    gap_bridges: List[str] = Field(
        default_factory=list,
        description="Gap concerns to bridge by reframing existing experience",
    )


class TailoredExperience(BaseModel):
    """One role within a tailored resume, preserving base resume hierarchy."""

    company: str = Field(..., description="Company name (never altered by tailoring)")
    location: str = Field("", description="Company location / work mode")
    role: str = Field(..., description="Job title held (never altered by tailoring)")
    dates: str = Field(..., description="Employment dates (never altered by tailoring)")
    bullets: List[str] = Field(
        default_factory=list,
        description="3-4 tailored bullets, clean text without source annotations",
    )
    is_relevant: bool = Field(
        True,
        description="False if the role is irrelevant to the target job — drop or minimize",
    )
    original_bullets: List[str] = Field(
        default_factory=list,
        description="Original bullets from the base resume, for diff/verification",
    )


class TailoredResume(BaseModel):
    """Response: structured tailored resume and cover letter for a job."""

    name: str = Field(..., description="Candidate name")
    title: str = Field(..., description="Candidate headline/title")
    contact: str = Field("", description="Contact line (email, site, location)")
    summary: str = Field(..., description="Tailored professional summary")
    skills: List[str] = Field(
        ..., description="8-12 skills surfaced from the base resume that match the job"
    )
    experience: List[TailoredExperience] = Field(
        ..., description="Work experience preserving the base resume hierarchy"
    )
    education: str = Field("", description="Education section text")
    volunteer: Optional[str] = Field(None, description="Volunteer experience text")
    languages: List[str] = Field(default_factory=list, description="Languages")
    tools: List[str] = Field(default_factory=list, description="Tools")
    cover_letter: str = Field(..., description="Honest gap-bridging cover letter")
    changes_made: List[str] = Field(
        ..., description="List of changes made to align with the job"
    )
    warnings: List[str] = Field(
        ..., description="Gaps or requirements without resume evidence"
    )
    estimated_ats_score: int = Field(
        ..., ge=0, le=100, description="Estimated ATS keyword match score"
    )
    created_at: str = Field(..., description="ISO timestamp of tailoring")


class CoverLetterInput(BaseModel):
    """Request: generate a standalone cover letter."""

    job_url: Optional[str] = Field(None, description="Original job posting URL")
    job_data: JobParseResult = Field(..., description="Structured job analysis")
    base_resume_markdown: str = Field(..., description="Candidate resume in Markdown")
    tone: str = Field("professional", description="Tone of generated content")


class BaseResumeInput(BaseModel):
    """Request: save the candidate's base resume for future sessions."""

    content: str = Field(..., description="Base resume in Markdown")


class PdfGenerateInput(BaseModel):
    """Request: generate a PDF from a tailored resume."""

    tailored_resume: TailoredResume = Field(..., description="Tailored resume data")
    type: str = Field("resume", description="PDF type: resume or cover_letter")
    company: Optional[str] = Field(
        None, description="Target company name, used in the PDF filename"
    )
