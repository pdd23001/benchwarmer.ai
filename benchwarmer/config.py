"""Environment and config for API keys, limits, and domain allowlists."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Load .env from project root so PERPLEXITY_API_KEY etc. are available
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parents[1]
    load_dotenv(_root / ".env")
except ImportError:
    pass


@dataclass
class SearchConfig:
    """Configuration for Perplexity Sonar and search agent."""

    perplexity_api_key: str = field(default_factory=lambda: os.environ.get("PERPLEXITY_API_KEY", ""))
    perplexity_base_url: str = field(
        default_factory=lambda: os.environ.get("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
    )
    sonar_model: str = field(
        default_factory=lambda: os.environ.get("PERPLEXITY_SONAR_MODEL", "sonar")
    )

    # Anthropic / Claude for code generation fallback
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    )

    max_github_results: int = int(os.environ.get("BENCHWARMER_MAX_GITHUB_RESULTS", "10"))
    max_paper_results: int = int(os.environ.get("BENCHWARMER_MAX_PAPER_RESULTS", "10"))
    min_implementation_confidence: float = float(
        os.environ.get("BENCHWARMER_MIN_IMPL_CONFIDENCE", "0.6")
    )
    scrape_domain_allowlist: List[str] = field(
        default_factory=lambda: [
            "github.com",
            "arxiv.org",
            "doi.org",
            "scholar.google.com",
            "papers.nips.cc",
            "dl.acm.org",
            "ieee.org",
        ]
    )
    scrape_max_depth: int = int(os.environ.get("BENCHWARMER_SCRAPE_MAX_DEPTH", "1"))


def get_search_config() -> SearchConfig:
    """Return the active search config (env-backed)."""
    return SearchConfig()
