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
from job_compass.models import JobParseResult, TailoredExperience, TailoredResume

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


CONCEPT_KEYWORDS = {
    "ai": {
        "ai",
        "artificial intelligence",
        "machine learning",
        "ml",
        "llm",
        "llms",
        "claude",
        "chatgpt",
        "cursor",
        "gpt",
        "neural",
        "deep learning",
        "nlp",
        "computer vision",
        "powered",
    },
    "machine learning": {
        "machine learning",
        "ml",
        "ai",
        "artificial intelligence",
        "neural",
        "model",
        "deep learning",
        "nlp",
        "computer vision",
        "predictive",
    },
    "ml": {
        "ml",
        "machine learning",
        "ai",
        "artificial intelligence",
        "neural",
        "model",
        "deep learning",
        "nlp",
        "computer vision",
    },
    "gamification": {
        "gamification",
        "missions",
        "quizzes",
        "rewards",
        "engagement",
        "points",
        "badges",
        "challenges",
        "streaks",
        "leaderboards",
        "missions",
    },
    "growth": {
        "growth",
        "experimentation",
        "a/b testing",
        "conversion",
        "retention",
        "acquisition",
        "metrics",
        "funnel",
        "optimization",
        "experiments",
    },
    "product": {
        "product",
        "roadmap",
        "strategy",
        "pm",
        "management",
        "agile",
        "scrum",
        "backlog",
        "stakeholder",
        "launch",
        "lifecycle",
    },
    "data": {
        "data",
        "analytics",
        "sql",
        "metrics",
        "dashboards",
        "bi",
        "business intelligence",
        "analysis",
        "insights",
    },
    "cloud": {
        "cloud",
        "aws",
        "azure",
        "gcp",
        "serverless",
        "docker",
        "kubernetes",
        "devops",
        "ci/cd",
    },
    "api": {
        "api",
        "rest",
        "graphql",
        "microservices",
        "integration",
        "webhook",
        "sdk",
    },
    "testing": {
        "testing",
        "qa",
        "unit",
        "integration",
        "e2e",
        "automation",
        "tdd",
        "selenium",
        "pytest",
    },
    "design": {
        "design",
        "figma",
        "sketch",
        "adobe",
        "ui",
        "ux",
        "prototype",
        "wireframe",
        "visual",
    },
    "marketing": {
        "marketing",
        "seo",
        "sem",
        "social media",
        "campaign",
        "brand",
        "content",
        "email",
    },
    "sales": {
        "sales",
        "crm",
        "salesforce",
        "lead",
        "pipeline",
        "revenue",
        "quota",
        "negotiation",
    },
    "mobile": {
        "mobile",
        "android",
        "ios",
        "swift",
        "kotlin",
        "react native",
        "flutter",
    },
    "web": {
        "web",
        "javascript",
        "typescript",
        "react",
        "vue",
        "angular",
        "html",
        "css",
        "frontend",
    },
    "backend": {
        "backend",
        "python",
        "java",
        "node",
        "ruby",
        "php",
        "go",
        "rust",
        "server",
    },
    "security": {
        "security",
        "encryption",
        "authentication",
        "authorization",
        "oauth",
        "jwt",
        "penetration",
    },
    "devops": {
        "devops",
        "ci/cd",
        "jenkins",
        "github actions",
        "docker",
        "kubernetes",
        "terraform",
        "ansible",
    },
    "blockchain": {
        "blockchain",
        "web3",
        "crypto",
        "nft",
        "smart contract",
        "ethereum",
        "solidity",
    },
    "gaming": {
        "gaming",
        "game",
        "player",
        "liveops",
        "monetization",
        "economy",
        "progression",
        "level",
        "quest",
    },
}


@dataclass
class ResumeChunk:
    """A chunk of a resume with extracted keywords and optional embedding."""

    id: int
    text: str
    keywords: set[str] = field(default_factory=set)
    embedding: Optional[np.ndarray] = None
    keyword_count: int = 0


@dataclass
class ParsedRole:
    """One role (company + title + dates + bullets) parsed from a base resume."""

    company: str
    location: str
    role: str
    dates: str
    bullets: List[str] = field(default_factory=list)


@dataclass
class ParsedResume:
    """Structured representation of the candidate's base resume."""

    name: str = ""
    title: str = ""
    contact: str = ""
    summary: str = ""
    skills: List[str] = field(default_factory=list)
    roles: List[ParsedRole] = field(default_factory=list)
    education: str = ""
    volunteer: Optional[str] = None
    languages: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)


_SECTION_ALIASES = {
    "summary": {"professional summary", "summary"},
    "skills": {"core skills", "skills"},
    "experience": {"professional experience", "experience", "work experience"},
    "education": {"education"},
    "volunteer": {"volunteer experience", "volunteer"},
    "languages": {"languages"},
    "tools": {"tools"},
}


def _section_key(header: str) -> Optional[str]:
    """Map a markdown section header to a canonical section key."""
    normalized = header.strip().lower()
    for key, aliases in _SECTION_ALIASES.items():
        if normalized in aliases:
            return key
    return None


def _strip_markdown(text: str) -> str:
    """Remove markdown emphasis markers and trim whitespace."""
    return re.sub(r"[*_`]+", "", text).strip()


def _parse_header_lines(lines: List[str], parsed: ParsedResume) -> int:
    """Parse name/title/contact from the resume header. Returns next line index."""
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line == "---":
            idx += 1
            continue
        if line.startswith("# ") and not line.startswith("## "):
            parsed.name = _strip_markdown(line[2:])
        elif line.startswith("## "):
            break
        elif not parsed.title and re.fullmatch(r"\*\*.+\*\*", line):
            parsed.title = _strip_markdown(line)
        elif not parsed.contact and "|" in line:
            parsed.contact = _strip_markdown(line)
        idx += 1
    return idx


def _parse_experience_lines(lines: List[str], parsed: ParsedResume) -> None:
    """Parse company/role/bullet hierarchy from experience section lines."""
    company = ""
    location = ""
    current_role: Optional[ParsedRole] = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line == "---":
            continue
        if line.startswith("### "):
            header = _strip_markdown(line[4:])
            company, _, location = (part.strip() for part in header.partition("|"))
            current_role = None
            continue
        role_match = re.match(r"^\*\*(.+?)\*\*\s*\|\s*(.+)$", line)
        if role_match:
            current_role = ParsedRole(
                company=company,
                location=location,
                role=role_match.group(1).strip(),
                dates=role_match.group(2).strip(),
            )
            parsed.roles.append(current_role)
            continue
        bullet_match = re.match(r"^[-•·◦*]\s+(.+)$", line)
        if bullet_match and current_role is not None:
            current_role.bullets.append(_strip_markdown(bullet_match.group(1)))


def parse_resume_structure(text: str) -> ParsedResume:
    """Parse a markdown resume into its structured representation.

    Expects the standard format: ``# Name`` header, ``## SECTION`` headers, and
    ``### Company | Location`` blocks containing ``**Role** | Dates`` lines and
    ``- bullet`` lines. Falls back to the configured profile for name/title.

    Args:
        text: Markdown resume text.

    Returns:
        ParsedResume with sections and role hierarchy.

    Raises:
        ValueError: If no experience roles could be parsed.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    parsed = ParsedResume()

    idx = _parse_header_lines(lines, parsed)

    section_key: Optional[str] = None
    section_lines: Dict[str, List[str]] = {}
    for line in lines[idx:]:
        header_match = re.match(r"^##\s+(.+?)\s*$", line)
        if header_match:
            section_key = _section_key(header_match.group(1))
            continue
        if section_key is not None:
            section_lines.setdefault(section_key, []).append(line)

    def _section_text(key: str) -> str:
        raw = "\n".join(section_lines.get(key, []))
        raw = re.sub(r"\n{2,}", "\n", raw)
        cleaned = [
            _strip_markdown(line)
            for line in raw.split("\n")
            if line.strip() and line.strip() != "---"
        ]
        return "\n".join(cleaned).strip()

    def _section_bullets(key: str) -> List[str]:
        items: List[str] = []
        for line in section_lines.get(key, []):
            bullet_match = re.match(r"^\s*[-•·◦*]\s+(.+)$", line.strip())
            if bullet_match:
                items.append(_strip_markdown(bullet_match.group(1)))
        return items

    parsed.summary = " ".join(_section_text("summary").split())
    parsed.skills = _section_bullets("skills")
    _parse_experience_lines(section_lines.get("experience", []), parsed)
    parsed.education = _section_text("education")
    volunteer = _section_text("volunteer")
    parsed.volunteer = volunteer or None
    parsed.languages = _section_bullets("languages")

    tools_text = " ".join(_section_text("tools").split())
    if tools_text:
        parsed.tools = [t.strip() for t in re.split(r"[•·,]", tools_text) if t.strip()]

    if not parsed.name:
        parsed.name = config.profile.name
    if not parsed.title:
        parsed.title = config.profile.title
    if not parsed.roles:
        raise ValueError(
            "Could not parse any experience from the resume. Expected sections like "
            "'## PROFESSIONAL EXPERIENCE' with '### Company | Location' and "
            "'**Role** | Dates' lines."
        )
    return parsed


TAILOR_SYSTEM_PROMPT = """You are an expert resume writer with strict anti-hallucination rules.

You will be given:
1. A job description (requirements, title, company)
2. The candidate's work experience as STRUCTURED ROLES. Each role has an index, company, title, dates, and its original bullets. Those bullets are the ONLY source material you may use.
3. A list of gap concerns to bridge (optional)

Your task is to tailor the resume for this job using ONLY the provided role bullets. You may NOT invent metrics, companies, dates, products, responsibilities, or skills. You may only reword, reorder, and emphasize existing evidence.

Return ONLY a JSON object with these exact fields:

{
  "summary": "2-3 sentence professional summary tailored to the job. Only use facts from the provided bullets.",
  "skills": ["skill1", "skill2", "..."],
  "experience": [
    {
      "role_index": 0,
      "is_relevant": true,
      "bullets": [
        "tailored bullet ending with [SOURCE: \\"exact original bullet text\\"] or [BRIDGED: \\"exact original bullet text\\"]"
      ]
    }
  ],
  "changes_made": ["brief description of each change made"],
  "warnings": ["any gaps where the job requirement has no matching evidence"],
  "estimated_ats_score": 75
}

RULES:
- Include EVERY role index in "experience", in the same order as provided.
- For each role, select and tailor the 3-4 strongest, most job-relevant bullets, strongest first. NEVER more than 4 bullets per role.
- If a role is irrelevant to the target job, set "is_relevant": false and return at most 1-2 bullets (or an empty list).
- Every bullet MUST end with [SOURCE: "exact original bullet text"] or [BRIDGED: "exact original bullet text"]. The cited text must be a verbatim substring of THAT role's original bullets.
- Use [SOURCE: ...] for bullets that directly match the job requirements.
- Use [BRIDGED: ...] for bullets that reframe existing experience to address a gap concern. The reframed bullet must still be based on the cited original bullet, not invented.
- Rewrite bullets to use the job's keywords naturally, but preserve every fact, number, and outcome exactly.
- The skills list MUST contain 8-12 skills, each explicitly mentioned in the provided resume content.
- estimated_ats_score must be an integer 0-100. Estimate how well the tailored resume matches the job requirements based ONLY on the provided evidence.
- Do NOT invent numbers, percentages, metrics, or outcomes that are not in the provided bullets.
- If a requirement has no matching bullet, note it in warnings.
- Do NOT include markdown formatting in the JSON strings.

GAP BRIDGING:
If gap concerns are provided, for each gap find the closest relevant bullet across all roles and reframe it using the job's language and context. Do NOT add new fake experience. Example:
- Gap: "No explicit beauty/wellness industry experience"
- Closest bullet: "Conducted customer discovery with 15 merchants across three Thai islands"
- Reframed: "Led customer discovery with 15 local merchants, driving consumer marketplace adoption [BRIDGED: Conducted customer discovery with 15 merchants across three Thai islands]"
"""


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


def _build_role_evidence(roles: List[ParsedRole]) -> str:
    """Serialize parsed roles with their bullets for the LLM prompt."""
    blocks: List[str] = []
    for idx, role in enumerate(roles):
        header = f"ROLE {idx}: {role.company}"
        if role.location:
            header += f" | {role.location}"
        header += f" — {role.role} ({role.dates})"
        bullets = "\n".join(f"  BULLET: {bullet}" for bullet in role.bullets)
        blocks.append(f"{header}\n{bullets}")
    return "\n\n".join(blocks)


async def _call_tailor_llm(
    job_data: JobParseResult,
    roles: List[ParsedRole],
    tone: str,
    max_length_pages: int,
    gap_bridges: Optional[List[str]] = None,
) -> dict[str, Any]:
    """Call the LLM to tailor the resume, preserving role structure."""
    role_evidence = _build_role_evidence(roles)
    requirements_text = "\n".join(
        f"- {req}" for req in list(job_data.key_requirements) + list(job_data.nice_to_have)
    )

    gap_bridges_text = ""
    if gap_bridges:
        gap_bridges_text = (
            "\n\nGap Concerns to Bridge (reframe closest matching experience):\n"
            + "\n".join(f"- {gap}" for gap in gap_bridges)
        )

    user_prompt = f"""Tone: {tone}

Job Title: {job_data.title}
Company: {job_data.company}
Seniority: {job_data.seniority}
Description Summary: {job_data.description_summary}

Job Requirements:
{requirements_text}
{gap_bridges_text}

Candidate Experience (ONLY source material you may use):
{role_evidence}

Target length: {max_length_pages} page{'s' if max_length_pages > 1 else ''}.

Return the JSON object now."""

    response = await client.chat.completions.create(
        model=config.models.generator_model,
        messages=[
            {"role": "system", "content": TAILOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=4000,
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


def _extract_skill_tokens(skill: str) -> set[str]:
    """Extract meaningful tokens from a skill string."""
    tokens = re.findall(r"[a-zA-Z0-9/+\-]+", skill.lower())
    return {token for token in tokens if len(token) > 2 and token not in STOP_WORDS}


def _skill_keyword_overlap(skill: str, base_resume: str) -> bool:
    """Check if a skill has significant keyword overlap with the base resume."""
    skill_tokens = _extract_skill_tokens(skill)
    if not skill_tokens:
        return False
    base_tokens = _extract_keywords(base_resume)
    overlap = len(skill_tokens & base_tokens)
    return overlap > 0 and overlap / len(skill_tokens) >= 0.5


def _skill_concept_match(skill: str, base_resume: str) -> bool:
    """Check if a skill's underlying concept is demonstrated in the base resume."""
    skill_lower = skill.lower()
    base_lower = base_resume.lower()
    for concept, keywords in CONCEPT_KEYWORDS.items():
        if concept in skill_lower:
            for keyword in keywords:
                if keyword in base_lower:
                    return True
    return False


def _find_source_in_chunks(source: str, chunks: List[ResumeChunk]) -> bool:
    """Check if a source citation is a substring of any chunk.

    Both sides are normalized (markdown stripped, whitespace collapsed):
    the LLM cites the markdown-stripped bullets it was shown, while raw
    chunks may still contain emphasis markers like ``**11.2K MAU**``.
    """
    normalized = _normalize_citation(source)
    for chunk in chunks:
        if normalized in _normalize_citation(chunk.text):
            return True
    return False


def _normalize_citation(text: str) -> str:
    """Strip markdown emphasis and collapse whitespace for citation checks."""
    return re.sub(r"\s+", " ", _strip_markdown(text)).strip()


_ANNOTATION_RE = re.compile(
    r"\s*\[(?:SOURCE|BRIDGED):\s*(?:\"[^\"]*\"|'[^']*'|[^\]]+)\]\s*$"
)


def _strip_annotations(bullet: str) -> str:
    """Remove the trailing [SOURCE: ...] / [BRIDGED: ...] annotation."""
    return _ANNOTATION_RE.sub("", bullet).strip()


def validate_tailored_resume(
    base_resume: str,
    annotated_bullets: List[str],
    summary: str,
    cover_letter: str,
    skills: List[str],
    chunks: Optional[List[ResumeChunk]] = None,
) -> List[str]:
    """Validate LLM-tailored content against anti-hallucination rules.

    Validation runs on the internal annotated bullets (with [SOURCE: ...] /
    [BRIDGED: ...] citations). Annotations are stripped only after validation
    passes; they never appear in the final output.

    Returns:
        List of warnings for uncertain or missing evidence.

    Raises:
        ValueError: If a hard anti-hallucination rule is violated
        (missing source citation, hallucinated numbers).
    """
    warnings: List[str] = []
    base_numbers = _extract_numbers(base_resume)
    base_lower = base_resume.lower()

    for i, bullet in enumerate(annotated_bullets):
        if "[SOURCE:" not in bullet and "[BRIDGED:" not in bullet:
            raise ValueError(
                f"Tailored bullet {i + 1} missing [SOURCE: or [BRIDGED: annotation"
            )

        source_match = re.search(r'\[(?:SOURCE|BRIDGED):\s*"(.*)"\]', bullet)
        if not source_match:
            # Fallback for unquoted or single-quoted annotations
            source_match = re.search(
                r'\[(?:SOURCE|BRIDGED):\s*(?:\'([^\']*)\'|([^\]]+))\]',
                bullet,
            )
        if not source_match:
            raise ValueError(
                f"Tailored bullet {i + 1} has malformed SOURCE/BRIDGED annotation"
            )

        source = next((g for g in source_match.groups() if g is not None), "")
        if not source.strip():
            raise ValueError(f"Tailored bullet {i + 1} has empty SOURCE/BRIDGED citation")

        if chunks and not _find_source_in_chunks(source, chunks):
            raise ValueError(
                f"Tailored bullet {i + 1} cites source not found in resume chunks: "
                f"{source[:50]}"
            )

        # Number validation: BRIDGED bullets may reframe numbers from the cited source
        if "[BRIDGED:" in bullet:
            source_chunk_numbers = _extract_numbers(source)
            bullet_body = bullet[: source_match.start()].strip()
            body_numbers = _extract_numbers(bullet_body)
            if body_numbers - source_chunk_numbers:
                raise ValueError(
                    f"Tailored bullet {i + 1} has numbers not in cited source: "
                    f"{sorted(body_numbers - source_chunk_numbers)}"
                )

    for field_name, text in [
        ("tailored_bullets", "\n".join(annotated_bullets)),
        ("summary", summary),
        ("cover_letter", cover_letter),
    ]:
        # For bullets, only check SOURCE bullets; BRIDGED bullets are checked above
        if field_name == "tailored_bullets":
            source_bullets = [b for b in annotated_bullets if "[BRIDGED:" not in b]
            text = "\n".join(source_bullets)
        text_numbers = _extract_numbers(text)
        hallucinated = text_numbers - base_numbers
        if hallucinated:
            raise ValueError(
                f"Hallucinated numbers in {field_name}: {sorted(hallucinated)}"
            )

    for skill in skills:
        skill_lower = skill.lower()

        # Exact match
        if skill_lower in base_lower:
            continue

        # Keyword overlap
        if _skill_keyword_overlap(skill, base_resume):
            continue

        # Concept match
        if _skill_concept_match(skill, base_resume):
            continue

        warnings.append(f"Skill not clearly demonstrated in resume: {skill}")

    return warnings


MAX_BULLETS_PER_ROLE = 4
MAX_BULLETS_IRRELEVANT_ROLE = 2
MAX_SKILLS = 12


def _build_experience(
    roles: List[ParsedRole],
    llm_experience: Any,
) -> tuple[List[TailoredExperience], List[str]]:
    """Map LLM-tailored bullets onto the parsed role hierarchy.

    Companies, roles, locations, and dates always come from the parsed base
    resume — the LLM only supplies bullets — so they can never be hallucinated.

    Args:
        roles: Roles parsed from the base resume, in order.
        llm_experience: The "experience" array from the LLM JSON response.

    Returns:
        Tuple of (structured experiences with clean bullets, flat list of
        annotated bullets for validation).
    """
    by_index: Dict[int, dict[str, Any]] = {}
    if isinstance(llm_experience, list):
        for entry in llm_experience:
            if isinstance(entry, dict) and isinstance(entry.get("role_index"), int):
                by_index[entry["role_index"]] = entry

    experiences: List[TailoredExperience] = []
    annotated: List[str] = []
    for idx, role in enumerate(roles):
        entry = by_index.get(idx, {})
        is_relevant = bool(entry.get("is_relevant", False))
        cap = MAX_BULLETS_PER_ROLE if is_relevant else MAX_BULLETS_IRRELEVANT_ROLE
        raw_bullets = [
            bullet for bullet in entry.get("bullets", []) if isinstance(bullet, str)
        ][:cap]
        annotated.extend(raw_bullets)
        experiences.append(
            TailoredExperience(
                company=role.company,
                location=role.location,
                role=role.role,
                dates=role.dates,
                bullets=[_strip_annotations(bullet) for bullet in raw_bullets],
                is_relevant=is_relevant,
                original_bullets=list(role.bullets),
            )
        )
    return experiences, annotated


async def tailor_resume(
    base_resume: str,
    job_data: JobParseResult,
    tone: str = "professional",
    max_length_pages: int = 2,
    gap_bridges: Optional[List[str]] = None,
) -> TailoredResume:
    """Tailor a resume for a specific job with anti-hallucination guardrails.

    Parses the base resume into its structured sections, has the LLM select
    and reword bullets per role (with internal source citations), validates
    the citations and facts, then strips citations from the final output.

    Args:
        base_resume: Markdown resume text.
        job_data: Parsed job description.
        tone: Desired tone (professional, casual, formal).
        max_length_pages: Target length in pages.
        gap_bridges: Gap concerns to bridge by reframing existing experience.

    Returns:
        Structured TailoredResume preserving the base resume hierarchy.
    """
    parsed = parse_resume_structure(base_resume)
    chunks = chunk_resume(base_resume)
    all_requirements = list(job_data.key_requirements) + list(job_data.nice_to_have)
    matches = await match_chunks_to_requirements(chunks, all_requirements)

    warnings: List[str] = []
    for requirement in all_requirements:
        if not matches.get(requirement):
            warnings.append(f"No resume evidence found for: {requirement}")

    raw_result = await _call_tailor_llm(
        job_data, parsed.roles, tone, max_length_pages, gap_bridges
    )

    experiences, annotated = _build_experience(
        parsed.roles, raw_result.get("experience", [])
    )
    if not annotated:
        raise ValueError("LLM returned no tailored bullets")

    summary = str(raw_result.get("summary", "")).strip()
    skills = [
        skill for skill in raw_result.get("skills", []) if isinstance(skill, str)
    ][:MAX_SKILLS]
    changes_made = raw_result.get("changes_made", [])
    estimated_ats_score = raw_result.get("estimated_ats_score", 70)

    cover_letter = await generate_cover_letter(job_data, base_resume, tone)

    llm_warnings = raw_result.get("warnings", [])
    if isinstance(llm_warnings, list):
        warnings.extend(llm_warnings)

    validation_warnings = validate_tailored_resume(
        base_resume, annotated, summary, cover_letter, skills, chunks
    )
    warnings.extend(validation_warnings)

    return TailoredResume(
        name=parsed.name,
        title=parsed.title,
        contact=parsed.contact,
        summary=summary,
        skills=skills,
        experience=experiences,
        education=parsed.education,
        volunteer=parsed.volunteer,
        languages=parsed.languages,
        tools=parsed.tools,
        cover_letter=cover_letter,
        changes_made=changes_made,
        warnings=warnings,
        estimated_ats_score=max(0, min(100, int(estimated_ats_score))),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


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
