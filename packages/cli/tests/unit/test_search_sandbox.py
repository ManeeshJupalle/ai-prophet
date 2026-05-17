from __future__ import annotations

from datetime import UTC, datetime

from ai_prophet.search.sandbox import (
    filter_sandbox_results,
    parse_as_of,
    provider_exclusive_date_cutoff,
    provider_exclusive_us_date_cutoff,
    provider_inclusive_iso_cutoff,
)


def test_parse_as_of_date_string_includes_entire_day():
    cutoff = parse_as_of("2026-05-01")

    assert cutoff == datetime(2026, 5, 1, 23, 59, 59, 999999, tzinfo=UTC)


def test_sandbox_filter_rejects_future_and_missing_dates_by_default():
    cutoff = parse_as_of("2026-05-01")
    results = [
        {
            "url": "https://example.com/old",
            "title": "Old",
            "published_date": "2026-04-30",
        },
        {
            "url": "https://example.com/future",
            "title": "Future",
            "published_date": "2026-05-02",
        },
        {
            "url": "https://example.com/missing",
            "title": "Missing",
        },
    ]

    filtered = filter_sandbox_results(results, as_of=cutoff)

    assert [item["url"] for item in filtered.accepted] == ["https://example.com/old"]
    assert [item["url"] for item in filtered.rejected] == [
        "https://example.com/future",
        "https://example.com/missing",
    ]
    assert filtered.accepted[0]["sandbox_status"] == "accepted"
    assert filtered.rejected[0]["sandbox_reason"].startswith("published_date")
    assert filtered.rejected[1]["sandbox_reason"] == "missing provider date metadata"


def test_sandbox_filter_can_allow_missing_dates_when_configured():
    cutoff = parse_as_of("2026-05-01")
    filtered = filter_sandbox_results(
        [{"url": "https://example.com/missing", "title": "Missing"}],
        as_of=cutoff,
        missing_date_policy="allow",
    )

    assert len(filtered.accepted) == 1
    assert filtered.accepted[0]["sandbox_status"] == "accepted_missing_date"
    assert filtered.warnings


def test_sandbox_filter_rejects_future_crawled_date():
    cutoff = parse_as_of("2026-05-01")
    filtered = filter_sandbox_results(
        [
            {
                "url": "https://example.com/crawled-late",
                "title": "Crawled late",
                "published_date": "2026-04-01",
                "crawled_date": "2026-05-02",
            }
        ],
        as_of=cutoff,
    )

    assert len(filtered.accepted) == 0
    assert filtered.rejected[0]["sandbox_reason"].startswith("crawled_date")


def test_provider_cutoff_formatters_include_as_of_date():
    cutoff = parse_as_of("2026-05-01")

    assert provider_inclusive_iso_cutoff(cutoff) == "2026-05-01T23:59:59.999Z"
    assert provider_exclusive_date_cutoff(cutoff) == "2026-05-02"
    assert provider_exclusive_us_date_cutoff(cutoff) == "5/2/2026"
