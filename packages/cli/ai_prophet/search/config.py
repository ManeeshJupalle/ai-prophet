"""Configuration for AI Prophet search tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchConfig:
    """Search-related configuration."""

    provider: str = "brave"
    as_of: str | None = None
    missing_date_policy: str = "reject"
    sandbox_fetch_multiplier: int = 2
    max_queries_per_market: int = 1
    max_results_per_query: int = 3
    mock: bool = False
    connect_timeout: int = 10
    total_timeout: int = 30
    fetch_timeout: int = 15
    max_concurrent: int = 3
    max_html_bytes: int = 512 * 1024
    max_extract_chars: int = 5_000
