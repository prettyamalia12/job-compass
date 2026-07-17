"""LLM-based job-candidate fit scoring."""

import json
from typing import Any

from openai import AsyncOpenAI

from job_compass.config import config

client = AsyncOpenAI(api_key=config.openai_api_key)

SCORE_SYSTEM_PROMPT = """You are a senior product director evaluating whether a candidate should apply for a job.

Evaluate the fit between the candidate and job. Return ONLY a JSON object with these exact fields:

{
  "match_score": 0-100,
  "match_reasoning": "2-3 sentences explaining the score. Be specific about what matches and what gaps exist.",
  "recommendation": "strong_apply" | "apply" | "consider" | "skip",
  "strength_matches": ["specific candidate strengths that align with job requirements"],
  "gap_concerns": ["areas where candidate may be weak or missing experience"],
  "questions_to_ask": ["good questions to ask in an interview based on gaps or ambiguities"]
}

Scoring rubric:
- 90-100: Dream fit. Strong_apply.
- 75-89: Good fit. Apply.
- 60-74: Possible fit with gaps. Consider.
- 0-59: Poor fit. Skip.

Consider ALL of these in scoring:
1. Role alignment (title, seniority, responsibilities)
2. Industry fit (gaming, AI, consumer, etc.)
3. Skills overlap (technical + soft skills)
4. Remote/location compatibility
5. Visa sponsorship availability
6. Salary alignment (if disclosed)
7. Company stage and culture signals
8. Career trajectory fit

Be honest and critical. A low score with clear reasoning is more valuable than an inflated score."""


async def score_job_fit(job_data: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Score how well a job matches the candidate profile.

    Args:
        job_data: Parsed job description fields.
        profile: Candidate profile dictionary.

    Returns:
        Dictionary with match score, reasoning, and recommendations.

    Raises:
        ValueError: If the LLM returns invalid JSON.
        RuntimeError: If the API call fails.
    """
    response = await client.chat.completions.create(
        model=config.models.scorer_model,
        messages=[
            {"role": "system", "content": SCORE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CANDIDATE PROFILE:\n{json.dumps(profile, indent=2)}\n\n"
                    f"JOB DETAILS:\n{json.dumps(job_data, indent=2)}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty response")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc
