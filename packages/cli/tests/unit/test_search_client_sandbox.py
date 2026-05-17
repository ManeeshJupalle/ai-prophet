from __future__ import annotations

import asyncio
from typing import Any

from ai_prophet.search import SearchClient
from ai_prophet.search.providers import (
    ExaProvider,
    PerplexityProvider,
    ProviderSearchRequest,
    create_provider,
)
from ai_prophet.search.sandbox import parse_as_of


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.seen_limits: list[int] = []

    async def search(self, session: object, request: ProviderSearchRequest) -> list[dict[str, Any]]:
        assert request.query == "market question"
        self.seen_limits.append(request.limit)
        return [
            {
                "url": "https://example.com/old",
                "title": "Old",
                "snippet": "old",
                "text": "old text",
                "score": 1.0,
                "provider": "fake",
                "published_date": "2026-04-30",
            },
            {
                "url": "https://example.com/new",
                "title": "New",
                "snippet": "new",
                "text": "new text",
                "score": 0.9,
                "provider": "fake",
                "published_date": "2026-05-02",
            },
        ]


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def post(self, endpoint: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"endpoint": endpoint, **kwargs})
        return FakeResponse(self.payload)


def test_search_client_keeps_public_class_and_filters_custom_provider_results():
    provider = FakeProvider()
    client = SearchClient(
        api_key="unused",
        provider=provider,
    )

    try:
        results = client.search("market question", limit=3, as_of="2026-05-01")
    finally:
        client.close()

    assert [item["url"] for item in results] == ["https://example.com/old"]
    assert results[0]["sandbox_status"] == "accepted"
    assert [item["url"] for item in client.last_rejected] == ["https://example.com/new"]
    assert provider.seen_limits == [3 * client.sandbox_fetch_multiplier]


def test_search_client_uses_live_search_when_no_as_of_is_provided():
    provider = FakeProvider()
    client = SearchClient(api_key="unused", provider=provider)

    try:
        results = client.search("market question", limit=3)
    finally:
        client.close()

    assert [item["url"] for item in results] == [
        "https://example.com/old",
        "https://example.com/new",
    ]
    assert all(item["sandbox_status"] == "live" for item in results)
    assert client.last_rejected == []
    assert provider.seen_limits == [3]


def test_trade_search_import_is_compatible():
    from ai_prophet.trade.search import SearchClient as TradeSearchClient

    assert TradeSearchClient is SearchClient


def test_create_provider_supports_new_provider_names():
    assert create_provider("exa", api_key="k").name == "exa"
    assert create_provider("tavily", api_key="k").name == "tavily"
    assert create_provider("perplexity", api_key="k").name == "perplexity"


def test_exa_provider_applies_cutoff_and_normalizes_results():
    session = FakeSession(
        {
            "results": [
                {
                    "url": "https://example.com/a",
                    "title": "A",
                    "summary": "summary",
                    "text": "content",
                    "publishedDate": "2026-04-30T12:00:00Z",
                    "lastCrawledDate": "2026-04-30T13:00:00Z",
                }
            ]
        }
    )
    provider = ExaProvider(api_key="exa-key")

    results = asyncio.run(
        provider.search(
            session,  # type: ignore[arg-type]
            ProviderSearchRequest(
                query="q",
                limit=2,
                as_of=parse_as_of("2026-05-01"),
            ),
        )
    )

    call = session.calls[0]
    assert call["endpoint"] == "https://api.exa.ai/search"
    assert call["headers"]["x-api-key"] == "exa-key"
    assert call["json"]["endPublishedDate"] == "2026-05-01T23:59:59.999Z"
    assert call["json"]["endCrawlDate"] == "2026-05-01T23:59:59.999Z"
    assert results[0]["url"] == "https://example.com/a"
    assert results[0]["published_date"] == "2026-04-30T12:00:00Z"
    assert results[0]["crawled_date"] == "2026-04-30T13:00:00Z"


def test_perplexity_provider_uses_next_day_before_filters():
    session = FakeSession(
        {
            "results": [
                {
                    "url": "https://example.com/a",
                    "title": "A",
                    "snippet": "snippet",
                    "date": "2026-04-30",
                    "last_updated": "2026-04-30",
                }
            ]
        }
    )
    provider = PerplexityProvider(api_key="pplx-key")

    results = asyncio.run(
        provider.search(
            session,  # type: ignore[arg-type]
            ProviderSearchRequest(query="q", limit=2, as_of=parse_as_of("2026-05-01")),
        )
    )

    call = session.calls[0]
    assert call["endpoint"] == "https://api.perplexity.ai/search"
    assert call["headers"]["Authorization"] == "Bearer pplx-key"
    assert call["json"]["search_before_date_filter"] == "5/2/2026"
    assert call["json"]["last_updated_before_filter"] == "5/2/2026"
    assert results[0]["text"] == "snippet"
