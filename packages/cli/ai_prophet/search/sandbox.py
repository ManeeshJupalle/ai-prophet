"""Sandbox cutoff filtering for local search results."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Literal

MissingDatePolicy = Literal["reject", "allow"]

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class SandboxFilterResult:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    warnings: list[str]


def parse_as_of(value: str | date | datetime | None) -> datetime | None:
    """Parse an as-of cutoff.

    Date-only values mean the end of that UTC day, so ``2026-05-01`` includes
    sources dated on May 1 but rejects May 2 and later.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, date):
        return datetime.combine(value, time.max, tzinfo=UTC)

    raw = value.strip()
    if not raw:
        return None
    if _DATE_ONLY_RE.fullmatch(raw):
        parsed_date = date.fromisoformat(raw)
        return datetime.combine(parsed_date, time.max, tzinfo=UTC)
    return _ensure_utc(_parse_datetime(raw))


def parse_result_datetime(value: Any) -> datetime | None:
    """Best-effort parse for provider publication/update date fields."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, date):
        return datetime.combine(value, time.max, tzinfo=UTC)
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    try:
        if _DATE_ONLY_RE.fullmatch(raw):
            parsed_date = date.fromisoformat(raw)
            return datetime.combine(parsed_date, time.max, tzinfo=UTC)
        return _ensure_utc(_parse_datetime(raw))
    except ValueError:
        return None


def filter_sandbox_results(
    results: list[dict[str, Any]],
    *,
    as_of: datetime | None,
    missing_date_policy: MissingDatePolicy = "reject",
) -> SandboxFilterResult:
    """Apply an as-of cutoff to normalized search results."""
    if as_of is None:
        live_results = []
        for result in results:
            item = dict(result)
            item.setdefault("sandbox_status", "live")
            item.setdefault("sandbox_reason", None)
            live_results.append(item)
        return SandboxFilterResult(accepted=live_results, rejected=[], warnings=[])

    if missing_date_policy not in ("reject", "allow"):
        raise ValueError("missing_date_policy must be 'reject' or 'allow'")

    cutoff = _ensure_utc(as_of)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    warnings: list[str] = []

    for result in results:
        item = dict(result)
        published_at = parse_result_datetime(item.get("published_date"))
        updated_at = parse_result_datetime(item.get("updated_date"))
        crawled_at = parse_result_datetime(item.get("crawled_date"))

        if published_at is not None:
            item["published_date"] = _format_date(published_at)
        if updated_at is not None:
            item["updated_date"] = _format_date(updated_at)
        if crawled_at is not None:
            item["crawled_date"] = _format_date(crawled_at)

        reason = _sandbox_rejection_reason(
            published_at=published_at,
            updated_at=updated_at,
            crawled_at=crawled_at,
            cutoff=cutoff,
            missing_date_policy=missing_date_policy,
        )
        if reason is not None:
            item["sandbox_status"] = "rejected"
            item["sandbox_reason"] = reason
            rejected.append(item)
            continue

        if published_at is None and updated_at is None and crawled_at is None:
            warnings.append(
                f"Accepted result without provider date metadata: {item.get('url', '')}"
            )
            item["sandbox_status"] = "accepted_missing_date"
            item["sandbox_reason"] = "missing provider date metadata allowed by policy"
        else:
            item["sandbox_status"] = "accepted"
            item["sandbox_reason"] = None
        accepted.append(item)

    return SandboxFilterResult(accepted=accepted, rejected=rejected, warnings=warnings)


def provider_inclusive_iso_cutoff(as_of: datetime) -> str:
    """Return an ISO cutoff suitable for providers with date-time filters."""
    cutoff = _ensure_utc(as_of)
    return cutoff.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def provider_exclusive_date_cutoff(as_of: datetime) -> str:
    """Return next-day YYYY-MM-DD for provider filters phrased as 'before date'."""
    cutoff = _ensure_utc(as_of)
    next_day = cutoff.date() + timedelta(days=1)
    return next_day.isoformat()


def provider_exclusive_us_date_cutoff(as_of: datetime) -> str:
    """Return next-day M/D/YYYY for Perplexity's before-date filters."""
    cutoff = _ensure_utc(as_of)
    next_day = cutoff.date() + timedelta(days=1)
    return f"{next_day.month}/{next_day.day}/{next_day.year}"


def _sandbox_rejection_reason(
    *,
    published_at: datetime | None,
    updated_at: datetime | None,
    crawled_at: datetime | None,
    cutoff: datetime,
    missing_date_policy: MissingDatePolicy,
) -> str | None:
    if (
        published_at is None
        and updated_at is None
        and crawled_at is None
        and missing_date_policy == "reject"
    ):
        return "missing provider date metadata"
    if published_at is not None and published_at > cutoff:
        return f"published_date {published_at.date().isoformat()} is after as_of {cutoff.date().isoformat()}"
    if updated_at is not None and updated_at > cutoff:
        return f"updated_date {updated_at.date().isoformat()} is after as_of {cutoff.date().isoformat()}"
    if crawled_at is not None and crawled_at > cutoff:
        return f"crawled_date {crawled_at.date().isoformat()} is after as_of {cutoff.date().isoformat()}"
    return None


def _parse_datetime(raw: str) -> datetime:
    cleaned = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        pass

    for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return datetime.combine(parsed.date(), time.max)
        except ValueError:
            continue

    parsed_email_date = parsedate_to_datetime(raw)
    if parsed_email_date is None:
        raise ValueError(f"Unable to parse datetime: {raw}")
    return parsed_email_date


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _format_date(value: datetime) -> str:
    return _ensure_utc(value).date().isoformat()
