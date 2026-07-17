"""LLM-based job description parsing."""

import json
from typing import Any

from openai import AsyncOpenAI

from job_compass.config import config

client = AsyncOpenAI(api_key=config.openai_api_key)

PARSE_SYSTEM_PROMPT = """You are an expert job description parser.

Extract structured data from the job description below. Return ONLY a JSON object with these exact fields. No markdown, no explanations outside the JSON.

{
  "title": "exact job title",
  "company": "company name (infer from context or URL domain if needed)",
  "location": "city/country or 'Remote' or 'Multiple locations'",
  "remote_policy": "fully_remote" | "hybrid" | "onsite" | "unknown",
  "salary_range": "salary range if stated, else null",
  "employment_type": "full_time" | "contract" | "freelance" | "unknown",
  "seniority": "senior" | "lead" | "staff" | "principal" | "manager" | "unknown",
  "description_summary": "2-3 sentence summary of what this role does day-to-day",
  "key_requirements": ["list of 5-8 must-have skills/experience"],
  "nice_to_have": ["list of 3-5 bonus skills/experience"],
  "red_flags": ["potential concerns: unrealistic expectations, toxic signals, vague descriptions, missing salary when expected, etc."],
  "green_flags": ["positive signals: clear growth path, good culture mentions, exciting tech stack, strong mission"],
  "visa_mentioned": true | false,
  "visa_sponsorship": true | false | null
}

Rules:
- visa_sponsorship: true if explicitly offered, false if explicitly NOT offered, null if not mentioned at all
- remote_policy: "fully_remote" only if explicitly stated. "hybrid" if mentions office days. "unknown" if unclear.
- seniority: infer from title and requirements. "lead" for team lead roles, "staff" for IC senior roles.
- company: if not in the text, infer from the URL domain (e.g., "google.com" → "Google")"""


async def parse_job_description(text: str) -> dict[str, Any]:
    """Parse raw job text into structured fields using LLM.

    Args:
        text: Raw job description text.

    Returns:
        Dictionary with parsed job fields.

    Raises:
        ValueError: If the LLM returns invalid JSON.
        RuntimeError: If the API call fails.
    """
    response = await client.chat.completions.create(
        model=config.models.parser_model,
        messages=[
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Job description:\n\n{text}"},
        ],
        temperature=0.1,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty response")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc
