"""Public-market consensus signal for the ensemble.

Two lookup paths, tried in order by :func:`market_signal_estimate`:

1. **Kalshi by ticker** — the competition's events come from Kalshi, so the
   ``market_ticker`` on each event is a direct key into Kalshi's public API
   at ``/trade-api/v2/markets/{ticker}``. No auth required for reads. When
   we get a hit we know the prices we see are for the exact same question,
   so we trust this signal with ``strategy="market_price"`` and
   ``confidence=0.65``.

2. **Polymarket by fuzzy title match** — fallback when the Kalshi ticker
   lookup fails (404, rate-limit, etc.) OR the event isn't a Kalshi market
   at all. We pull a page of Polymarket markets and rank by Jaccard overlap
   of normalized title words. Returns ``strategy="market_consensus"`` so
   logs distinguish exact-ticker matches from fuzzy-title matches.

Both return :class:`Estimate` or ``None``. Why this exists: the official
scoring formula is ``(our_brier - market_brier) * completion_rate``.
Matching the market on events where we have no edge guarantees a near-zero
contribution from those events; beating the market on events where our
research finds alpha is then pure upside.

All operations are best-effort. Missing endpoint, network error, parse
error, unexpected JSON shape, missing price field, or no good title match
all return ``None`` — never an exception.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

from .strategies.base import Estimate, clamp_probability

logger = logging.getLogger(__name__)

KALSHI_TRADING_URL = "https://trading-api.kalshi.com/trade-api/v2/markets"
"""Main Kalshi trading API — requires ``KALSHI_API_KEY`` (sent as bearer)."""

KALSHI_ELECTIONS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
"""Public elections-only endpoint — no auth required, but limited inventory."""

POLYMARKET_URL = os.environ.get(
    "POLYMARKET_API_URL",
    "https://clob.polymarket.com/markets",
)
"""Public Polymarket CLOB markets list. Override via ``POLYMARKET_API_URL``."""

REQUEST_TIMEOUT = 6.0
"""Short timeout — we'd rather skip the signal than block the pipeline."""

MIN_JACCARD = 0.30
"""Minimum word-overlap score (after stopword removal) to accept a Polymarket match."""

MIN_OVERLAP = 3
"""Minimum raw word overlap before scoring — guards against tiny coincidences."""

MAX_MARKETS = 200
"""How many Polymarket markets to pull and scan. Endpoint paginates; this is one page."""

MARKET_CONSENSUS_CONFIDENCE = 0.65
"""Used by both Kalshi exact-ticker hits and Polymarket fuzzy-title hits."""

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
        "will", "is", "are", "was", "were", "be", "by", "with", "at",
        "this", "that", "these", "those", "have", "has", "had", "do",
        "does", "did", "as", "from", "it", "its",
    }
)


# ---------------------------------------------------------------------------
# Title matching
# ---------------------------------------------------------------------------


def _normalize(text: str) -> set[str]:
    """Lower-case, strip punctuation, drop stopwords and 1-char tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) >= 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Polymarket payload parsing
# ---------------------------------------------------------------------------


def _yes_price_from_market(market: dict[str, Any]) -> float | None:
    """Extract the YES outcome price from a Polymarket market record.

    Polymarket response shapes vary between endpoints and time; we try the
    three forms we've observed and return None if none match.
    """
    tokens = market.get("tokens")
    if isinstance(tokens, list):
        for tok in tokens:
            if not isinstance(tok, dict):
                continue
            outcome = str(tok.get("outcome", "")).strip().lower()
            if outcome == "yes":
                try:
                    return float(tok.get("price"))
                except (TypeError, ValueError):
                    return None

    prices = market.get("outcome_prices") or market.get("outcomePrices")
    if isinstance(prices, list) and prices:
        try:
            return float(prices[0])
        except (TypeError, ValueError):
            return None

    last = market.get("lastTradePrice") or market.get("last_trade_price")
    if last is not None:
        try:
            return float(last)
        except (TypeError, ValueError):
            return None

    return None


# ---------------------------------------------------------------------------
# Kalshi: direct ticker lookup (primary)
# ---------------------------------------------------------------------------


def _kalshi_yes_price_cents(market: dict[str, Any]) -> int | float | None:
    """Pick the most-current YES price (in cents) from a Kalshi market record.

    Order of preference:
      1. ``last_price`` — most recent trade, if it's a sensible 0 < p < 100.
      2. Midpoint of ``yes_bid`` / ``yes_ask`` — current order book.
      3. ``yes_ask`` alone — upper-bound when only one side is quoted.
      4. ``yes_bid`` alone — lower-bound.
    """
    last = market.get("last_price")
    if last is not None:
        try:
            v = float(last)
        except (TypeError, ValueError):
            v = None
        if v is not None and 0.0 < v < 100.0:
            return v

    bid = market.get("yes_bid")
    ask = market.get("yes_ask")
    try:
        bid_v = float(bid) if bid is not None else None
        ask_v = float(ask) if ask is not None else None
    except (TypeError, ValueError):
        bid_v = ask_v = None

    if bid_v is not None and ask_v is not None and bid_v > 0 and ask_v > 0:
        return (bid_v + ask_v) / 2.0
    if ask_v is not None and ask_v > 0:
        return ask_v
    if bid_v is not None and bid_v > 0:
        return bid_v
    return None


def kalshi_price_estimate(
    market_ticker: str | None,
    title: str | None = None,
) -> Estimate | None:
    """Look up the current Kalshi market price by ticker and return an Estimate.

    Hits ``KALSHI_API_URL/{market_ticker}``. Returns ``None`` on any failure
    (empty ticker, network error, non-200, malformed payload, no usable
    price field). Prices are in cents on Kalshi; we divide by 100 to map to
    a probability.

    ``title`` is unused for the lookup itself (the ticker is exact) but is
    included in the rationale for traceability.
    """
    if not market_ticker:
        return None
    ticker = str(market_ticker).strip()
    if not ticker:
        return None

    # Read env at call time so a key added mid-process takes effect, and so
    # tests can override via monkeypatch. Explicit ``KALSHI_API_URL`` wins.
    api_key = os.environ.get("KALSHI_API_KEY", "").strip()
    base = os.environ.get("KALSHI_API_URL") or (
        KALSHI_TRADING_URL if api_key else KALSHI_ELECTIONS_URL
    )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

    url = f"{base.rstrip('/')}/{ticker}"
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT, headers=headers) as client:
            resp = client.get(url)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 — never block pipeline
        logger.info("kalshi lookup failed ticker=%s: %s", ticker, exc)
        return None

    # Kalshi wraps the record as {"market": {...}}; tolerate a bare dict too.
    if isinstance(payload, dict) and "market" in payload:
        market = payload.get("market")
    else:
        market = payload
    if not isinstance(market, dict):
        logger.info("kalshi payload not a market dict ticker=%s", ticker)
        return None

    yes_cents = _kalshi_yes_price_cents(market)
    if yes_cents is None:
        logger.info("kalshi market has no usable price ticker=%s", ticker)
        return None

    p_yes = clamp_probability(yes_cents / 100.0)
    cents_str = f"{yes_cents:.0f}c" if yes_cents == int(yes_cents) else f"{yes_cents:.1f}c"
    rationale = f"Kalshi market price: {cents_str} (as of query time)"
    if title:
        rationale += f" for '{title[:80]}'"
    return Estimate(
        p_yes=p_yes,
        rationale=rationale,
        strategy="market_price",
        confidence=MARKET_CONSENSUS_CONFIDENCE,
    )


# ---------------------------------------------------------------------------
# Polymarket: fuzzy title match (fallback)
# ---------------------------------------------------------------------------


def query_polymarket(title: str, *, max_markets: int = MAX_MARKETS) -> dict | None:
    """Return the best-matching Polymarket market dict for ``title``, or None."""
    if not title:
        return None

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.get(POLYMARKET_URL, params={"limit": max_markets})
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 — never block pipeline
        logger.info("polymarket request failed: %s", exc)
        return None

    # Some Polymarket endpoints wrap the list in {"data": [...]}; others
    # return a bare list.
    if isinstance(payload, dict):
        markets = payload.get("data") or payload.get("markets") or []
    elif isinstance(payload, list):
        markets = payload
    else:
        return None

    if not isinstance(markets, list) or not markets:
        return None

    event_words = _normalize(title)
    if not event_words:
        return None

    best: dict | None = None
    best_score = 0.0
    for m in markets:
        if not isinstance(m, dict):
            continue
        question = (
            m.get("question")
            or m.get("title")
            or m.get("market_slug")
            or ""
        )
        market_words = _normalize(str(question))
        if not market_words:
            continue
        if len(event_words & market_words) < MIN_OVERLAP:
            continue
        score = _jaccard(event_words, market_words)
        if score > best_score:
            best_score = score
            best = m

    if best is None or best_score < MIN_JACCARD:
        return None
    return best


def market_consensus_estimate(title: str) -> Estimate | None:
    """Look up the best-matching public market and return it as an Estimate.

    Returns ``None`` on every failure mode — network error, no match, missing
    price, unexpected payload shape.
    """
    market = query_polymarket(title)
    if market is None:
        return None

    yes_price = _yes_price_from_market(market)
    if yes_price is None:
        logger.info("polymarket match found but no usable price field")
        return None

    question = market.get("question") or market.get("title") or "(unknown)"
    return Estimate(
        p_yes=clamp_probability(yes_price),
        rationale=(
            f"Polymarket consensus: '{question}' trading at YES "
            f"{yes_price:.3f}"
        ),
        strategy="market_consensus",
        confidence=MARKET_CONSENSUS_CONFIDENCE,
    )


# ---------------------------------------------------------------------------
# Dispatcher: Kalshi first, then Polymarket fallback
# ---------------------------------------------------------------------------


def market_signal_estimate(
    market_ticker: str | None,
    title: str | None,
) -> Estimate | None:
    """Try Kalshi by ticker; fall back to Polymarket by fuzzy title.

    This is the single entrypoint the pipeline calls. Hierarchy:

    1. If ``market_ticker`` is set, try Kalshi's direct lookup. Exact match
       to the question we're scoring — most trustworthy.
    2. If Kalshi misses (404, network, malformed, no price) AND ``title`` is
       set, try Polymarket's fuzzy title match — useful when the event came
       from somewhere other than Kalshi, or when Kalshi is down.
    3. If both miss, return ``None``. The ensemble proceeds without a
       market-consensus signal.
    """
    if market_ticker:
        est = kalshi_price_estimate(market_ticker, title)
        if est is not None:
            return est
    if title:
        return market_consensus_estimate(title)
    return None


__all__ = [
    "MARKET_CONSENSUS_CONFIDENCE",
    "MIN_JACCARD",
    "MIN_OVERLAP",
    "kalshi_price_estimate",
    "market_consensus_estimate",
    "market_signal_estimate",
    "query_polymarket",
]
