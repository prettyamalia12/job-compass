"""Job Compass FastAPI application."""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import aiosqlite
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from job_compass.config import config
from job_compass.models import (
    CoverLetterInput,
    JobParseResult,
    JobUrlInput,
    PdfGenerateInput,
    ResumeTailorInput,
    TailoredResume,
)
from job_compass.parser import parse_job_description
from job_compass.pdf_generator import generate_pdf
from job_compass.scorer import score_job_fit
from job_compass.scraper import scrape_job_page
from job_compass.tailor import (
    generate_cover_letter,
    make_cache_key,
    tailor_resume,
)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "job_compass.db")
DB_PATH = os.path.abspath(DB_PATH)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tailored_resumes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_url TEXT,
    job_title TEXT NOT NULL,
    company TEXT NOT NULL,
    base_resume_hash TEXT NOT NULL,
    tailored_resume_json TEXT NOT NULL,
    cover_letter TEXT NOT NULL,
    created_at TEXT NOT NULL,
    verified INTEGER DEFAULT 0
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_tailored_resumes_cache
ON tailored_resumes (job_url, base_resume_hash, verified);
"""


async def _init_db() -> None:
    """Ensure the SQLite database and tables exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_TABLE_SQL)
        await db.execute(CREATE_INDEX_SQL)
        await db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    await _init_db()
    print("🚀 Job Compass starting...")
    print(f"   Profile: {config.profile.name} — {config.profile.title}")
    print(f"   Parser: {config.models.parser_model}")
    print(f"   Scorer: {config.models.scorer_model}")
    print(f"   Generator: {config.models.generator_model}")
    print(f"   DB: {DB_PATH}")
    print(f"   API key: {'✅ Set' if config.openai_api_key else '❌ NOT SET — set OPENAI_API_KEY in .env'}")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="Job Compass",
    description="AI-powered job parser and match scorer for PM job seekers",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


async def _get_cached_tailored_resume(
    job_url: Optional[str],
    base_resume_hash: str,
) -> Optional[TailoredResume]:
    """Return a verified cached TailoredResume if one exists."""
    if not job_url:
        return None
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT tailored_resume_json FROM tailored_resumes "
            "WHERE job_url = ? AND base_resume_hash = ? AND verified = 1 "
            "ORDER BY created_at DESC LIMIT 1",
            (job_url, base_resume_hash),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return TailoredResume(**json.loads(row[0]))
    return None


async def _save_tailored_resume(
    job_url: Optional[str],
    base_resume_hash: str,
    job_data: JobParseResult,
    tailored: TailoredResume,
) -> None:
    """Persist a tailored resume to SQLite."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO tailored_resumes "
            "(job_url, job_title, company, base_resume_hash, tailored_resume_json, "
            "cover_letter, created_at, verified) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_url,
                job_data.title,
                job_data.company,
                base_resume_hash,
                json.dumps(tailored.model_dump()),
                tailored.cover_letter,
                tailored.created_at,
                0,
            ),
        )
        await db.commit()


async def _mark_verified(job_url: Optional[str], base_resume_hash: str) -> None:
    """Mark the most recent cached record as verified."""
    if not job_url:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE tailored_resumes SET verified = 1 "
            "WHERE job_url = ? AND base_resume_hash = ?",
            (job_url, base_resume_hash),
        )
        await db.commit()


@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "profile": config.profile.name,
    }


@app.get("/api/profile")
async def get_profile():
    """Return the candidate profile."""
    return config.profile.__dict__


@app.post("/api/parse", response_model=JobParseResult)
async def parse_job(input_data: JobUrlInput):
    """Analyze a job posting by URL.

    1. Scrapes the job page
    2. Parses with LLM into structured fields
    3. Scores fit against your profile
    4. Returns full analysis
    """
    url = str(input_data.url)

    try:
        raw_text = await scrape_job_page(url)
        parsed = await parse_job_description(raw_text)
        scored = await score_job_fit(parsed, config.profile.__dict__)

        result = {
            **parsed,
            **scored,
            "raw_description": raw_text[: config.max_raw_display_chars],
            "parsed_at": datetime.utcnow().isoformat(),
        }
        return JobParseResult(**result)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.post("/api/parse-text")
async def parse_job_text(text: str):
    """Analyze pasted job description text directly.

    Use this when scraping fails or for job boards that block bots.
    """
    if len(text) < 200:
        raise HTTPException(
            status_code=400,
            detail="Text too short. Please paste the full job description.",
        )

    try:
        parsed = await parse_job_description(text)
        scored = await score_job_fit(parsed, config.profile.__dict__)

        return {
            **parsed,
            **scored,
            "raw_description": text[: config.max_raw_display_chars],
            "parsed_at": datetime.utcnow().isoformat(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.post("/api/tailor", response_model=TailoredResume)
async def tailor_job_resume(input_data: ResumeTailorInput):
    """Tailor a resume for the provided job.

    Returns a cached result if job_url + resume_hash is verified.
    Otherwise generates a new tailored resume and persists it.
    """
    base_resume = input_data.base_resume_markdown
    job_data = input_data.job_data
    tone = input_data.tone
    max_length_pages = input_data.max_length_pages
    job_url = input_data.job_url
    base_resume_hash = make_cache_key(job_url or "", base_resume).split("::")[-1]

    try:
        cached = await _get_cached_tailored_resume(job_url, base_resume_hash)
        if cached:
            return cached

        tailored = await tailor_resume(
            base_resume,
            job_data,
            tone,
            max_length_pages,
            input_data.gap_bridges,
        )
        await _save_tailored_resume(job_url, base_resume_hash, job_data, tailored)
        return tailored

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tailoring failed: {exc}") from exc


@app.post("/api/tailor-cover-letter")
async def tailor_cover_letter(input_data: CoverLetterInput) -> dict[str, str]:
    """Generate a standalone gap-bridging cover letter."""
    try:
        cover_letter = await generate_cover_letter(
            input_data.job_data,
            input_data.base_resume_markdown,
            input_data.tone,
        )
        return {"cover_letter": cover_letter}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cover letter failed: {exc}") from exc


@app.post("/api/tailor-verify")
async def verify_tailored_resume(input_data: ResumeTailorInput) -> dict[str, bool]:
    """Mark a tailored resume as verified for future caching."""
    base_resume_hash = make_cache_key(
        input_data.job_url or "", input_data.base_resume_markdown
    ).split("::")[-1]
    await _mark_verified(input_data.job_url, base_resume_hash)
    return {"verified": True}


@app.post("/api/generate-pdf")
async def generate_pdf_endpoint(input_data: PdfGenerateInput) -> Response:
    """Generate a PDF resume or cover letter from tailored resume data."""
    pdf_type = input_data.type
    if pdf_type not in ("resume", "cover_letter"):
        raise HTTPException(
            status_code=400,
            detail="type must be 'resume' or 'cover_letter'",
        )

    try:
        pdf_bytes = generate_pdf(input_data.tailored_resume, pdf_type)
        filename = (
            f"tailored_resume_{pdf_type}_"
            f"{input_data.tailored_resume.created_at[:10].replace('-', '')}.pdf"
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}") from exc


def main():
    """Entry point for `job-compass` CLI command."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
