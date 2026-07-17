"""Resume tailoring with anti-hallucination guardrails."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI

from job_compass.config import config
from job_compass.models import JobParseResult, TailoredResume

client = AsyncOpenAI(api_key=config.openai_api_key)

EMBEDDING_MODEL = "text-embedding-3-small"

STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "can",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "now",
    "then",
    "also",
    "after",
    "before",
    "up",
    "down",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "once",
    "here",
    "there",
    "while",
    "during",
    "through",
    "between",
    "among",
    "against",
    "about",
    "into",
    "onto",
    "upon",
    "within",
    "without",
    "above",
    "below",
    "behind",
    "beyond",
    "across",
    "around",
    "near",
    "along",
    "next",
    "previous",
    "later",
    "soon",
    "early",
    "late",
    "quickly",
    "slowly",
    "well",
    "better",
    "best",
    "bad",
    "worse",
    "worst",
    "good",
    "great",
    "new",
    "old",
    "first",
    "last",
    "long",
    "short",
    "high",
    "low",
    "big",
    "small",
    "large",
    "little",
    "young",
    "able",
    "unable",
    "possible",
    "impossible",
    "available",
    "certain",
    "clear",
    "different",
    "similar",
    "usual",
    "normal",
    "common",
    "rare",
    "recent",
    "current",
    "past",
    "future",
    "local",
    "global",
    "internal",
    "external",
    "public",
    "private",
    "major",
    "minor",
    "main",
    "key",
    "primary",
    "secondary",
    "final",
    "initial",
    "total",
    "full",
    "empty",
    "used",
    "using",
    "use",
    "uses",
    "worked",
    "working",
    "works",
    "work",
    "managed",
    "managing",
    "manages",
    "manage",
    "led",
    "leading",
    "leads",
    "lead",
    "developed",
    "developing",
    "develops",
    "develop",
    "designed",
    "designing",
    "designs",
    "design",
    "created",
    "creating",
    "creates",
    "create",
    "built",
    "building",
    "builds",
    "build",
    "implemented",
    "implementing",
    "implements",
    "implement",
    "launched",
    "launching",
    "launches",
    "launch",
    "increased",
    "increasing",
    "increases",
    "increase",
    "decreased",
    "decreasing",
    "decreases",
    "decrease",
    "improved",
    "improving",
    "improves",
    "improve",
    "reduced",
    "reducing",
    "reduces",
    "reduce",
    "responsible",
    "responsibilities",
    "responsibility",
    "experience",
    "experienced",
    "experiences",
    "skill",
    "skills",
    "ability",
    "abilities",
    "expertise",
    "proficiency",
    "knowledge",
    "familiarity",
    "strong",
    "excellent",
    "proven",
    "track",
    "record",
    "background",
    "role",
    "position",
    "job",
    "candidate",
    "applicant",
    "professional",
    "career",
    "company",
    "team",
    "organization",
    "product",
    "project",
    "program",
    "platform",
    "system",
    "application",
    "feature",
    "service",
    "solution",
    "tool",
    "technology",
    "technologies",
    "framework",
    "language",
    "methodology",
    "process",
    "practice",
    "standard",
    "including",
    "included",
    "includes",
    "include",
    "etc",
    "eg",
    "ie",
    "etc.",
    "e.g.",
    "i.e.",
    "am",
    "are",
    "was",
    "were",
    "shall",
    "ought",
}


@dataclass
class ResumeChunk:
    """A chunk of a resume with extracted keywords and optional embedding."""

    id: int
    text: str
    keywords: set[str] = field(default_factory=set)
    embedding: Optional[np.ndarray] = None
    keyword_count: int = 0


TAILOR_SYSTEM_PROMPT = """You are an expert resume writer with strict anti-hallucination rules.

You will be given:
1. A job description (requirements, title, company)
2. A set of resume chunks that are the ONLY source material you may use

Your task is to tailor a resume for this job using ONLY the provided resume chunks. You may NOT invent metrics, companies, dates, products, responsibilities, or skills. You may only reword, reorder, and emphasize existing evidence.

Return ONLY a JSON object with these exact fields:

{
  "summary": "2-3 sentence professional summary tailored to the job. Only use facts from the provided chunks.",
  "tailored_bullets": [
    "bullet text that includes [SOURCE: \"exact original chunk text\"] at the end",
    ...
  ],
  "skills_section": ["skill1", "skill2", ...],
  "changes_made": ["brief description of each change made"],
  "warnings": ["any gaps where the job requirement has no matching evidence chunk"],
  "estimated_ats_score": 75
}

RULES:
- Every tailored_bullet MUST end with [SOURCE: "exact original chunk text"]. The source text must be a verbatim substring from the provided chunks.
- The skills_section MUST only contain skills explicitly mentioned in the provided chunks.
- estimated_ats_score must be an integer 0-100. Estimate how well the tailored resume matches the job requirements based ONLY on the provided evidence.
- Do NOT invent numbers, percentages, metrics, or outcomes that are not in the chunks.
- If a requirement has no matching chunk, note it in warnings.
- Do NOT include markdown formatting in the JSON strings."""


COVER_LETTER_SYSTEM_PROMPT = """You are an honest, senior product manager writing a cover letter.

You will be given a job description and a candidate's resume. Write a cover letter that:
1. Shows genuine enthusiasm for the role and company
2. Highlights 2-3 specific experiences from the resume that directly match the job requirements
3. Honestly addresses any gaps with a willingness to learn or transferable skills
4. Never invents metrics, companies, dates, or experiences not in the resume

Return ONLY the cover letter text. No markdown, no explanations."""


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from a text string."""
    tokens = re.findall(r"[a-zA-Z0-9]+(?:\+*[a-zA-Z0-9]*)", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in STOP_WORDS}


def _chunk_text(text: str) -> List[str]:
    """Split resume text into logical bullets/experience chunks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n(?=\s*[-•·◦*]\s+|\s*\d+[.)]\s+)", text)
    chunks: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        sub_parts = [p.strip() for p in part.split("\n\n") if p.strip()]
        chunks.extend(sub_parts)
    if not chunks:
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [c for c in chunks if len(c) > 15 and not c.startswith("#")]


def chunk_resume(text: str) -> List[ResumeChunk]:
    """Split a resume into experience bullets/chunks.

    Args:
        text: Markdown resume text.

    Returns:
        List of ResumeChunk objects with keywords extracted.
    """
    raw_chunks = _chunk_text(text)
    chunks: List[ResumeChunk] = []
    for i, raw in enumerate(raw_chunks):
        keywords = _extract_keywords(raw)
        chunks.append(
            ResumeChunk(
                id=i,
                text=raw,
                keywords=keywords,
                keyword_count=len(keywords),
            )
        )
    return chunks


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


async def _get_embeddings(texts: List[str]) -> Optional[List[np.ndarray]]:
    """Fetch OpenAI embeddings for a list of texts. Returns None on failure."""
    if not config.openai_api_key or not texts:
        return None
    try:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        return [
            np.array(item.embedding, dtype=np.float32) for item in response.data
        ]
    except Exception:
        return None


async def match_chunks_to_requirements(
    chunks: List[ResumeChunk],
    requirements: List[str],
    top_k: int = 3,
) -> Dict[str, List[ResumeChunk]]:
    """Match each job requirement to the most relevant resume chunks.

    Uses keyword overlap and, if available, embedding cosine similarity.

    Args:
        chunks: Resume chunks from chunk_resume.
        requirements: Job requirements to match.
        top_k: Maximum number of matching chunks per requirement.

    Returns:
        Dictionary mapping each requirement to its best matching chunks.
    """
    if not chunks or not requirements:
        return {}

    chunk_texts = [c.text for c in chunks]
    all_texts = chunk_texts + requirements
    embeddings = await _get_embeddings(all_texts)

    chunk_embeddings: List[Optional[np.ndarray]] = [None] * len(chunks)
    requirement_embeddings: List[Optional[np.ndarray]] = [None] * len(requirements)
    if embeddings:
        chunk_embeddings = embeddings[: len(chunks)]
        requirement_embeddings = embeddings[len(chunks) :]
        for idx, chunk in enumerate(chunks):
            chunk.embedding = chunk_embeddings[idx]

    requirement_keywords = [_extract_keywords(req) for req in requirements]
    matches: Dict[str, List[ResumeChunk]] = {}

    for req_idx, requirement in enumerate(requirements):
        req_keywords = requirement_keywords[req_idx]
        req_embedding = requirement_embeddings[req_idx]
        scored_chunks: List[tuple[float, ResumeChunk]] = []

        for chunk in chunks:
            if chunk.keywords and req_keywords:
                intersection = len(chunk.keywords & req_keywords)
                union = len(chunk.keywords | req_keywords)
                keyword_score = intersection / union if union > 0 else 0.0
            else:
                keyword_score = 0.0

            embedding_score = 0.0
            if chunk.embedding is not None and req_embedding is not None:
                embedding_score = _cosine_similarity(chunk.embedding, req_embedding)
            elif chunk_embeddings[chunk.id] is not None and req_embedding is not None:
                embedding_score = _cosine_similarity(
                    chunk_embeddings[chunk.id], req_embedding
                )

            combined_score = (keyword_score * 0.6) + (embedding_score * 0.4)
            scored_chunks.append((combined_score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for score, chunk in scored_chunks[:top_k] if score > 0.05]
        matches[requirement] = top_chunks

    return matches


def _build_evidence_chunks(matches: Dict[str, List[ResumeChunk]]) -> List[str]:
    """Deduplicate and serialize matched chunks for the LLM prompt."""
    evidence_chunks: List[str] = []
    seen_chunk_ids: set[int] = set()
    for chunks in matches.values():
        for chunk in chunks:
            if chunk.id not in seen_chunk_ids:
                evidence_chunks.append(f'CHUNK {chunk.id}: {chunk.text}')
                seen_chunk_ids.add(chunk.id)
    return evidence_chunks


async def _call_tailor_llm(
    job_data: JobParseResult,
    matches: Dict[str, List[ResumeChunk]],
    tone: str,
    max_length_pages: int,
) -> dict[str, Any]:
    """Call the LLM to tailor the resume."""
    evidence_chunks = _build_evidence_chunks(matches)
    requirements_text = "\n".join(
        f"- {req}" for req in list(job_data.key_requirements) + list(job_data.nice_to_have)
    )

    user_prompt = f"""Tone: {tone}

Job Title: {job_data.title}
Company: {job_data.company}
Seniority: {job_data.seniority}
Description Summary: {job_data.description_summary}

Job Requirements:
{requirements_text}

Resume Evidence Chunks (ONLY source material you may use):
{chr(10).join(evidence_chunks)}

Target length: {max_length_pages} page{'s' if max_length_pages > 1 else ''}.

Return the JSON object now."""

    response = await client.chat.completions.create(
        model=config.models.generator_model,
        messages=[
            {"role": "system", "content": TAILOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=3000,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty response")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc


def _extract_numbers(text: str) -> set[str]:
    """Extract all numeric tokens from text."""
    return set(re.findall(r"\d+(?:\.\d+)?%?", text))


def _extract_skills(text: str) -> set[str]:
    """Extract skill-like keywords from text.

    Prioritizes comma/newline delimited skills, then extracts adjacent word pairs.
    """
    skills: set[str] = set()
    for raw in re.split(r"[\n,]+", text):
        skill = raw.strip().strip("-.•·◦*").strip()
        if len(skill) > 2:
            skills.add(skill.lower())
    tokens = re.findall(
        r"[a-zA-Z][a-zA-Z0-9/+\-]*(?:\s+[a-zA-Z][a-zA-Z0-9/+\-]*)?",
        text.lower(),
    )
    skills.update(token.strip() for token in tokens if len(token.strip()) > 2)
    return skills


def _find_source_in_chunks(source: str, chunks: List[ResumeChunk]) -> bool:
    """Check if a source citation is a substring of any chunk."""
    for chunk in chunks:
        if source in chunk.text:
            return True
    return False


def validate_tailored_resume(
    base_resume: str,
    tailored: TailoredResume,
    chunks: Optional[List[ResumeChunk]] = None,
) -> None:
    """Validate a tailored resume against anti-hallucination rules.

    Raises:
        ValueError: If any anti-hallucination rule is violated.
    """
    base_numbers = _extract_numbers(base_resume)
    base_skills = _extract_skills(base_resume)
    base_skills_lower = {s.lower() for s in base_skills}

    for i, bullet in enumerate(tailored.tailored_bullets):
        if "[SOURCE:" not in bullet:
            raise ValueError(f"Tailored bullet {i + 1} missing [SOURCE: annotation")

        source_match = re.search(r'\[SOURCE:\s*"([^"]+)"\]', bullet)
        if not source_match:
            raise ValueError(f"Tailored bullet {i + 1} has malformed SOURCE annotation")

        source = source_match.group(1)
        if not source.strip():
            raise ValueError(f"Tailored bullet {i + 1} has empty SOURCE citation")

        if chunks and not _find_source_in_chunks(source, chunks):
            raise ValueError(
                f"Tailored bullet {i + 1} cites source not found in resume chunks: "
                f"{source[:50]}"
            )

    for field_name, text in [
        ("tailored_bullets", "\n".join(tailored.tailored_bullets)),
        ("summary", tailored.summary),
        ("cover_letter", tailored.cover_letter),
        ("changes_made", "\n".join(tailored.changes_made)),
    ]:
        text_numbers = _extract_numbers(text)
        hallucinated = text_numbers - base_numbers
        if hallucinated:
            raise ValueError(
                f"Hallucinated numbers in {field_name}: {sorted(hallucinated)}"
            )

    for skill in tailored.skills_section:
        skill_lower = skill.lower()
        if skill_lower not in base_skills_lower and not any(
            skill_lower in base_skill.lower() for base_skill in base_skills
        ):
            raise ValueError(f"Skill not found in base resume: {skill}")


async def tailor_resume(
    base_resume: str,
    job_data: JobParseResult,
    tone: str = "professional",
    max_length_pages: int = 2,
) -> TailoredResume:
    """Tailor a resume for a specific job with anti-hallucination guardrails.

    Args:
        base_resume: Markdown resume text.
        job_data: Parsed job description.
        tone: Desired tone (professional, casual, formal).
        max_length_pages: Target length in pages.

    Returns:
        TailoredResume with source-cited bullets and gap warnings.
    """
    chunks = chunk_resume(base_resume)
    all_requirements = list(job_data.key_requirements) + list(job_data.nice_to_have)
    matches = await match_chunks_to_requirements(chunks, all_requirements)

    warnings: List[str] = []
    for requirement in all_requirements:
        if not matches.get(requirement):
            warnings.append(f"No resume evidence found for: {requirement}")

    raw_result = await _call_tailor_llm(job_data, matches, tone, max_length_pages)

    original_bullets = [c.text for c in chunks]
    tailored_bullets = raw_result.get("tailored_bullets", [])
    skills_section = raw_result.get("skills_section", [])
    summary = raw_result.get("summary", "")
    changes_made = raw_result.get("changes_made", [])
    estimated_ats_score = raw_result.get("estimated_ats_score", 70)

    cover_letter = await generate_cover_letter(job_data, base_resume, tone)

    llm_warnings = raw_result.get("warnings", [])
    if isinstance(llm_warnings, list):
        warnings.extend(llm_warnings)

    tailored = TailoredResume(
        original_bullets=original_bullets,
        tailored_bullets=tailored_bullets,
        skills_section=skills_section,
        summary=summary,
        cover_letter=cover_letter,
        changes_made=changes_made,
        warnings=warnings,
        estimated_ats_score=max(0, min(100, int(estimated_ats_score))),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    validate_tailored_resume(base_resume, tailored, chunks)
    return tailored


async def generate_cover_letter(
    job_data: JobParseResult,
    base_resume: str,
    tone: str = "professional",
) -> str:
    """Generate a gap-bridging cover letter.

    Args:
        job_data: Parsed job description.
        base_resume: Markdown resume text.
        tone: Desired tone.

    Returns:
        Cover letter text.
    """
    requirements_text = "\n".join(
        f"- {req}" for req in list(job_data.key_requirements) + list(job_data.nice_to_have)
    )

    user_prompt = f"""Tone: {tone}

Job Title: {job_data.title}
Company: {job_data.company}
Description: {job_data.description_summary}

Requirements:
{requirements_text}

Candidate Resume:
{base_resume[:8000]}

Write a concise, honest cover letter (250-350 words)."""

    response = await client.chat.completions.create(
        model=config.models.generator_model,
        messages=[
            {"role": "system", "content": COVER_LETTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=1200,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty cover letter")

    return content.strip()


def make_cache_key(job_url: str, base_resume: str) -> str:
    """Create a stable cache key for a job URL + resume combination."""
    resume_hash = hashlib.sha256(base_resume.encode("utf-8")).hexdigest()
    return f"{job_url.strip()}::{resume_hash}"
