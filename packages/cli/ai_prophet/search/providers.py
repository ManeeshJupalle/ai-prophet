"""Provider backends for ``SearchClient``."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import aiohttp

from .sandbox import (
    provider_exclusive_date_cutoff,
    provider_exclusive_us_date_cutoff,
    provider_inclusive_iso_cutoff,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderSearchRequest:
    query: str
    limit: int
    as_of: datetime | None = None
    max_extract_chars: int = 5_000


class SearchProvider(Protocol):
    name: str

    async def search(
        self,
        session: aiohttp.ClientSession,
        request: ProviderSearchRequest,
    ) -> list[dict[str, Any]]:
        """Return normalized search result dictionaries."""


@dataclass(frozen=True)
class ExaProvider:
    api_key: str
    options: dict[str, Any] = field(default_factory=dict)

    name: str = "exa"
    endpoint: str = "https://api.exa.ai/search"

    async def search(
        self,
        session: aiohttp.ClientSession,
        request: ProviderSearchRequest,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "query": request.query,
            "numResults": request.limit,
            "contents": {
                "text": True,
                "summary": True,
            },
        }
        if request.as_of is not None:
            cutoff = provider_inclusive_iso_cutoff(request.as_of)
            payload["endPublishedDate"] = cutoff
            payload["endCrawlDate"] = cutoff
        payload.update(self.options)

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with session.post(self.endpoint, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
        except Exception as exc:
            logger.error("Exa search failed for query %r: %s", request.query, exc)
            return []

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(data.get("results", [])[: request.limit], start=1):
            url = item.get("url") or item.get("id")
            if not url:
                continue
            text = item.get("text") or item.get("summary") or ""
            normalized.append(
                {
                    "url": url,
                    "title": item.get("title", ""),
                    "snippet": item.get("summary", "") or _first_highlight(item),
                    "text": text[: request.max_extract_chars],
                    "score": item.get("score", 1.0 - ((index - 1) * 0.1)),
                    "provider": self.name,
                    "published_date": item.get("publishedDate") or item.get("published_date"),
                    "updated_date": item.get("updatedDate") or item.get("updated_date"),
                    "crawled_date": item.get("lastCrawledDate") or item.get("crawledDate"),
                }
            )
        return normalized


@dataclass(frozen=True)
class TavilyProvider:
    api_key: str
    options: dict[str, Any] = field(default_factory=dict)

    name: str = "tavily"
    endpoint: str = "https://api.tavily.com/search"

    async def search(
        self,
        session: aiohttp.ClientSession,
        request: ProviderSearchRequest,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "query": request.query,
            "max_results": request.limit,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": "text",
            "include_images": False,
        }
        if request.as_of is not None:
            payload["end_date"] = provider_exclusive_date_cutoff(request.as_of)
        payload.update(self.options)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with session.post(self.endpoint, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
        except Exception as exc:
            logger.error("Tavily search failed for query %r: %s", request.query, exc)
            return []

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(data.get("results", [])[: request.limit], start=1):
            url = item.get("url")
            if not url:
                continue
            text = item.get("raw_content") or item.get("content") or ""
            normalized.append(
                {
                    "url": url,
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "text": text[: request.max_extract_chars],
                    "score": item.get("score", 1.0 - ((index - 1) * 0.1)),
                    "provider": self.name,
                    "published_date": item.get("published_date")
                    or item.get("publishedDate")
                    or item.get("date"),
                    "updated_date": item.get("updated_date")
                    or item.get("updatedDate")
                    or item.get("last_updated"),
                }
            )
        return normalized


@dataclass(frozen=True)
class PerplexityProvider:
    api_key: str
    options: dict[str, Any] = field(default_factory=dict)

    name: str = "perplexity"
    endpoint: str = "https://api.perplexity.ai/search"

    async def search(
        self,
        session: aiohttp.ClientSession,
        request: ProviderSearchRequest,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "query": request.query,
            "max_results": request.limit,
            "max_tokens_per_page": max(1, request.max_extract_chars // 4),
        }
        if request.as_of is not None:
            cutoff = provider_exclusive_us_date_cutoff(request.as_of)
            payload["search_before_date_filter"] = cutoff
            payload["last_updated_before_filter"] = cutoff
        payload.update(self.options)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with session.post(self.endpoint, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
        except Exception as exc:
            logger.error("Perplexity search failed for query %r: %s", request.query, exc)
            return []

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(data.get("results", [])[: request.limit], start=1):
            url = item.get("url")
            if not url:
                continue
            snippet = item.get("snippet", "")
            normalized.append(
                {
                    "url": url,
                    "title": item.get("title", ""),
                    "snippet": snippet,
                    "text": snippet[: request.max_extract_chars],
                    "score": item.get("score", 1.0 - ((index - 1) * 0.1)),
                    "provider": self.name,
                    "published_date": item.get("date") or item.get("published_date"),
                    "updated_date": item.get("last_updated") or item.get("updated_date"),
                }
            )
        return normalized


def create_provider(
    provider: str,
    *,
    api_key: str,
    options: dict[str, Any] | None = None,
) -> SearchProvider:
    provider_name = provider.strip().lower()
    provider_options = options or {}
    if provider_name == "exa":
        return ExaProvider(api_key=api_key, options=provider_options)
    if provider_name == "tavily":
        return TavilyProvider(api_key=api_key, options=provider_options)
    if provider_name == "perplexity":
        return PerplexityProvider(api_key=api_key, options=provider_options)
    raise ValueError(f"Unsupported search provider: {provider}")


def _first_highlight(item: dict[str, Any]) -> str:
    highlights = item.get("highlights")
    if isinstance(highlights, list) and highlights:
        first = highlights[0]
        if isinstance(first, str):
            return first
    return ""
