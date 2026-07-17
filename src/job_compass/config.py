"""Configuration and profile settings."""

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Profile:
    """Your professional profile — edit this to match your background."""

    name: str = "Oktavia Pretty Amalia"
    title: str = "Senior Product Manager"
    years_experience: int = 6
    location: str = "Indonesia"
    open_to: List[str] = field(default_factory=lambda: ["remote", "relocation"])
    visa_required: bool = True

    skills: List[str] = field(default_factory=lambda: [
        "Product Strategy",
        "Gamification",
        "LiveOps",
        "Growth & Experimentation",
        "AI/LLMs (Claude, ChatGPT, Cursor)",
        "A/B Testing",
        "Customer Discovery",
        "Product Analytics",
        "SQL",
        "Roadmapping",
        "Stakeholder Management",
        "Game Product Design",
        "Monetization Strategy",
        "Figma",
        "Jira",
    ])

    industries: List[str] = field(default_factory=lambda: [
        "Gaming",
        "AI",
        "Consumer",
        "Blockchain (gaming-focused)",
        "Healthcare SaaS",
        "Mobility",
    ])

    target_roles: List[str] = field(default_factory=lambda: [
        "Senior Product Manager",
        "Product Lead",
        "Lead Product Manager",
    ])

    strengths: List[str] = field(default_factory=lambda: [
        "0→1 product launches (Freedom Planets: 400+ planets in 3 months)",
        "Gamification & engagement systems (11.2K MAU growth)",
        "AI-powered internal tools (80% operational effort reduction)",
        "Player economy & progression design",
        "Cross-functional team leadership (5 engineers)",
        "Engineering background (Android → PM transition)",
    ])

    dealbreakers: List[str] = field(default_factory=lambda: [
        "pure crypto",
        "fintech",
        "AdTech",
        "Java-based onsite",
        "no visa sponsorship when required",
        "no remote flexibility",
    ])

    preferences: dict = field(default_factory=lambda: {
        "min_salary_usd": 80000,
        "timezone_flexibility": "preferred",
        "team_size": "small_to_medium",
        "stage": "seed_to_series_b",
    })


@dataclass(frozen=True)
class ModelConfig:
    """LLM model selection and pricing (as of July 2026)."""

    # GPT-4.1 Nano: $0.10/$0.40 per 1M tokens — cheapest capable parser
    parser_model: str = "gpt-4.1-nano"
    parser_input_cost: float = 0.10  # per 1M tokens
    parser_output_cost: float = 0.40  # per 1M tokens

    # GPT-4o-mini: $0.15/$0.60 per 1M tokens — best cost/reasoning for scoring
    scorer_model: str = "gpt-4o-mini"
    scorer_input_cost: float = 0.15
    scorer_output_cost: float = 0.60

    # GPT-4.1 Mini: $0.40/$1.60 per 1M tokens — reserve for resume generation
    generator_model: str = "gpt-4.1-mini"
    generator_input_cost: float = 0.40
    generator_output_cost: float = 1.60


@dataclass(frozen=True)
class AppConfig:
    """Application configuration."""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    profile: Profile = field(default_factory=Profile)
    models: ModelConfig = field(default_factory=ModelConfig)
    max_scrape_chars: int = 12_000
    max_raw_display_chars: int = 2_500
    scrape_timeout_seconds: int = 30


# Singleton instance — import this everywhere
config = AppConfig()
