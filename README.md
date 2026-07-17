# 🧭 Job Compass

AI-powered job parser and match scorer for product managers. Paste a job URL or description, get structured analysis + a 0-100 fit score in 30 seconds.

Built by [Oktavia Pretty Amalia](https://prettyamalia.com) as a portfolio piece and practical job-hunting tool.

## What It Does

1. **Scrapes** any public job posting (LinkedIn, Wellfound, company careers pages)
2. **Parses** the description into structured data — requirements, salary, remote policy, visa info
3. **Scores** the job against your profile — match %, red/green flags, gap analysis
4. **Recommends** — strong_apply, apply, consider, or skip

## Quick Start

```bash
# 1. Clone your repo
git clone https://github.com/prettyamalia12/job-compass.git
cd job-compass

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install
pip install -e ".[dev]"

# 4. Configure
cp .env.example .env
# Edit .env with your OpenAI API key

# 5. Customize your profile
# Edit src/job_compass/config.py → YOUR_PROFILE

# 6. Run
make run
# → http://localhost:8000
```

Or with `uvicorn` directly:
```bash
uvicorn job_compass.main:app --reload --port 8000
```

## Usage

1. Open `http://localhost:8000`
2. Paste a job URL or full job description
3. Review the match score, flags, and recommendation
4. Decide whether to apply

## Architecture

```
job-compass/
├── src/job_compass/
│   ├── __init__.py
│   ├── config.py          # Your profile + model config
│   ├── models.py          # Pydantic schemas
│   ├── scraper.py         # HTTP scraping
│   ├── parser.py          # LLM parsing (GPT-4.1 Nano)
│   ├── scorer.py          # LLM scoring (GPT-4o-mini)
│   └── main.py            # FastAPI app
├── static/
│   └── index.html         # Frontend (no build step)
├── tests/
│   └── test_parser.py
├── pyproject.toml
├── Makefile
└── README.md
```

## Cost

| Task | Model | Cost per Job |
|---|---|---|
| Parse | GPT-4.1 Nano | ~$0.001 |
| Score | GPT-4o-mini | ~$0.003 |
| **Total** | | **~$0.004 per job** |

**50 jobs/day ≈ $6/month**

## Customization

### Edit your profile

Open `src/job_compass/config.py` and modify `YOUR_PROFILE`:

```python
Profile(
    name="Your Name",
    skills=["your", "skills", "here"],
    dealbreakers=["what", "to", "avoid"],
    ...
)
```

### Change models

In `config.py` → `ModelConfig`:

```python
parser_model="gpt-4o-mini"      # Higher quality parsing
scorer_model="gpt-4o"            # Higher quality scoring (10x cost)
```

## Development

```bash
make dev      # Hot-reload server
make test     # Run tests
make lint     # Check code style
make format   # Auto-fix code style
make clean    # Remove build artifacts
```

## Roadmap

- [ ] Day 2: Resume tailoring (upload base → get tailored bullets per job)
- [ ] Day 3: SQLite persistence + pipeline tracker
- [ ] Day 4: Cover letter generator
- [ ] Day 5: Interview prep per job (AI-generated Q&A)
- [ ] Week 2: Browser automation for form filling
- [ ] Week 2: Email integration for status tracking

## License

MIT — use it, fork it, improve it.
