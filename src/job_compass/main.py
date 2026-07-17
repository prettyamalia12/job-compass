"""Job Compass FastAPI application."""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from job_compass.config import config
from job_compass.models import JobParseResult, JobUrlInput
from job_compass.parser import parse_job_description
from job_compass.scorer import score_job_fit
from job_compass.scraper import scrape_job_page


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    print("🚀 Job Compass starting...")
    print(f"   Profile: {config.profile.name} — {config.profile.title}")
    print(f"   Parser: {config.models.parser_model}")
    print(f"   Scorer: {config.models.scorer_model}")
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


def main():
    """Entry point for `job-compass` CLI command."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
