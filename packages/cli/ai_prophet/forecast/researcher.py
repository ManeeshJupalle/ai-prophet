"""Web research module for the ensemble forecasting agent.

Given an event, this module:

1. Generates 5 diverse search queries via an LLM.
2. Hits DuckDuckGo's HTML endpoint for each query.
3. Extracts clean text from each result URL with trafilatura.
4. Compiles a single research brief (~8K chars) for downstream strategies.

Every step is wrapped in best-effort error handling — if search or extraction
fails for one URL we move on and return whatever we managed to gather. The
function never raises during normal operation; callers receive a (possibly
empty) string.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from html import unescape
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from .llm_utils import call_llm_json

logger = logging.getLogger(__name__)

DDG_HTML_URL = "https://html.duckduckgo.com/html/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
SEARCH_TIMEOUT = 12.0
FETCH_TIMEOUT = 10.0
RESEARCH_BRIEF_MAX_CHARS = 8000


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

_QUERY_SYSTEM_PROMPT = """You generate web search queries for a forecasting analyst.

Given a binary prediction-market question, output 5 diverse search queries that
together would give a well-informed forecaster a strong view on the likely
outcome. Each query should target a distinct angle: current status, recent
news, historical base rate, expert commentary, and contradictory or risk-
oriented information.

Respond with ONLY a JSON object of the form:
{"queries": ["query 1", "query 2", "query 3", "query 4", "query 5"]}

No prose, no markdown fences, no commentary."""


def generate_search_queries(
    title: str,
    description: str | None = None,
    category: str | None = None,
) -> list[str]:
    """Use an LLM to produce 5 diverse search queries for the event."""
    parts = [f"Event title: {title}"]
    if category:
        parts.append(f"Category: {category}")
    if description:
        parts.append(f"Description: {description}")
    parts.append(f"Today: {date.today().isoformat()}")
    parts.append(
        "\nReturn 5 distinct, well-formed search queries that, together, would "
        "ground a forecast on this event."
    )
    user_prompt = "\n".join(parts)

    try:
        data = call_llm_json(
            _QUERY_SYSTEM_PROMPT,
            user_prompt,
            tier="research",
            temperature=0.4,
            max_tokens=400,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("query generation failed: %s — falling back to defaults", exc)
        return _fallback_queries(title, category)

    queries = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(queries, list):
        return _fallback_queries(title, category)

    cleaned = []
    for q in queries:
        if isinstance(q, str) and q.strip():
            cleaned.append(q.strip())
        if len(cleaned) == 5:
            break

    if not cleaned:
        return _fallback_queries(title, category)
    return cleaned


def _fallback_queries(title: str, category: str | None) -> list[str]:
    base = title.strip()
    suffix = f" {category}" if category else ""
    return [
        base,
        f"{base} latest news",
        f"{base} forecast {date.today().year}",
        f"{base} odds prediction",
        f"{base}{suffix} analysis",
    ]


# ---------------------------------------------------------------------------
# DuckDuckGo HTML search
# ---------------------------------------------------------------------------

_RESULT_BLOCK_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?(?:<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>)?',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(html: str) -> str:
    return unescape(_TAG_RE.sub("", html)).strip()


def _normalize_ddg_url(url: str) -> str:
    """DDG wraps outgoing links in /l/?uddg=...; unwrap them."""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.path.endswith("/l/") or "/l/?" in url:
        qs = parse_qs(parsed.query)
        target = qs.get("uddg") or qs.get("u")
        if target:
            return unquote(target[0])
    return url


def search_duckduckgo(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Search DDG's HTML endpoint and return up to ``max_results`` results.

    Each result is a dict with ``url``, ``title``, ``snippet``. On any failure
    an empty list is returned.
    """
    try:
        with httpx.Client(
            timeout=SEARCH_TIMEOUT,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            follow_redirects=True,
        ) as client:
            resp = client.post(DDG_HTML_URL, data={"q": query})
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.info("ddg search failed q=%r: %s", query, exc)
        return []

    html = resp.text
    out: list[dict[str, str]] = []
    for match in _RESULT_BLOCK_RE.finditer(html):
        raw_url = match.group(1)
        url = _normalize_ddg_url(raw_url)
        if not url.startswith(("http://", "https://")):
            continue
        title = _strip_tags(match.group(2) or "")
        snippet = _strip_tags(match.group(3) or "")
        if not title:
            continue
        out.append({"url": url, "title": title, "snippet": snippet})
        if len(out) >= max_results:
            break
    return out


# ---------------------------------------------------------------------------
# Page content extraction
# ---------------------------------------------------------------------------


def fetch_page_content(url: str, max_chars: int = 2000) -> str:
    """Fetch ``url`` and return clean text, truncated to ``max_chars``.

    Returns an empty string on any failure.
    """
    try:
        import trafilatura
    except ImportError:  # pragma: no cover — trafilatura is in deps
        logger.warning("trafilatura not available; skipping page fetch")
        return ""

    try:
        with httpx.Client(
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.info("fetch failed url=%s: %s", url, exc)
        return ""

    try:
        text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("trafilatura extract failed url=%s: %s", url, exc)
        return ""

    if not text:
        return ""
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0] + "…"
    return cleaned


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def research_event(
    title: str,
    description: str | None = None,
    category: str | None = None,
    rules: str | None = None,
    *,
    max_results_per_query: int = 3,
    max_chars: int = RESEARCH_BRIEF_MAX_CHARS,
) -> str:
    """Run the full research pipeline and return a compiled brief.

    Never raises. Returns whatever was gathered, even if every step failed.
    """
    queries = generate_search_queries(title, description, category)
    logger.info("research.queries n=%d", len(queries))

    sections: list[str] = []
    header_parts = [
        f"# Research brief for: {title}",
        f"Today's date: {date.today().isoformat()}",
    ]
    if category:
        header_parts.append(f"Category: {category}")
    if rules:
        header_parts.append(f"Rules: {rules.strip()}")
    sections.append("\n".join(header_parts))
    sections.append("Search queries used:\n" + "\n".join(f"  - {q}" for q in queries))

    seen_urls: set[str] = set()
    snippet_buf: list[str] = []
    body_buf: list[str] = []

    for query in queries:
        results = search_duckduckgo(query, max_results=max_results_per_query)
        if not results:
            continue
        snippet_buf.append(f"\n## Query: {query}")
        for r in results:
            url = r["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            snippet_buf.append(f"- [{r['title']}]({url})\n  {r['snippet']}")
            content = fetch_page_content(url, max_chars=1800)
            if content:
                body_buf.append(
                    f"\n### Source: {r['title']}\nURL: {url}\n\n{content}"
                )

    if snippet_buf:
        sections.append("# Search results" + "\n".join(snippet_buf))
    if body_buf:
        sections.append("# Extracted source content" + "\n".join(body_buf))

    brief = "\n\n".join(sections).strip()
    if len(brief) > max_chars:
        brief = brief[:max_chars].rsplit("\n", 1)[0] + "\n…[truncated]"
    return brief


__all__ = [
    "generate_search_queries",
    "search_duckduckgo",
    "fetch_page_content",
    "research_event",
]
