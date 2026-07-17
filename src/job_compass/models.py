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
