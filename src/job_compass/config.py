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

    # Kimi K2 (Moonshot AI): $0.60/$2.50 per 1M tokens — fast, cheap, JSON-capable
    parser_model: str = "kimi-k2-0905-preview"
    parser_input_cost: float = 0.60  # per 1M tokens
    parser_output_cost: float = 2.50  # per 1M tokens

    # Kimi K2 (Moonshot AI): strong reasoning for scoring at the same price
    scorer_model: str = "kimi-k2-0905-preview"
    scorer_input_cost: float = 0.60
    scorer_output_cost: float = 2.50

    # Kimi K2 (Moonshot AI): resume/cover-letter generation
    generator_model: str = "kimi-k2-0905-preview"
    generator_input_cost: float = 0.60
    generator_output_cost: float = 2.50


@dataclass(frozen=True)
class AppConfig:
    """Application configuration."""

    # Kimi (Moonshot AI) — OpenAI-compatible API
    kimi_api_key: str = field(default_factory=lambda: os.getenv("KIMI_API_KEY", ""))
    kimi_base_url: str = field(
        default_factory=lambda: os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
    )
    profile: Profile = field(default_factory=Profile)
    models: ModelConfig = field(default_factory=ModelConfig)
    max_scrape_chars: int = 12_000
    max_raw_display_chars: int = 2_500
    scrape_timeout_seconds: int = 30


# Singleton instance — import this everywhere
config = AppConfig()
