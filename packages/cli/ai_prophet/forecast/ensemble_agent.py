"""Multi-strategy ensemble forecasting agent.

This is the entry point loaded by ``prophet forecast predict``:

.. code-block:: bash

    prophet forecast predict \\
        --events events.json \\
        --local ai_prophet.forecast.ensemble_agent

Three independent strategies run in parallel over the same web-researched
brief, then their probabilities are combined in log-odds space with
adaptive shrinkage. The module also exposes a FastAPI ``/predict`` endpoint
so the same agent can be served over HTTP via ``--agent-url``.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import BaseModel

from .cache import cache_enabled, cache_prediction, get_cached_prediction
from .ensemble import FinalPrediction, ensemble_predict
from .llm_utils import call_llm_json
from .researcher import research_event
from .strategies import (
    BaseRateStrategy,
    ContrarianStrategy,
    EvidenceWeightedStrategy,
)
from .strategies.base import (
    Estimate,
    clamp_confidence,
    clamp_probability,
    failed_estimate,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deliberation round
# ---------------------------------------------------------------------------

DELIBERATION_STRATEGY_NAME = "deliberation"
DELIBERATION_CONFIDENCE = 0.85
"""Confidence assigned to the deliberation Estimate. It has seen all three
analysts' rationales, so we trust it more than any single strategy — but not
so much that it overwhelms strong disagreement among the originals."""

_DELIBERATION_SYSTEM_PROMPT = """You are a meta-forecaster adjudicating among
three independent analysts' estimates of a binary prediction-market question.
Each analyst saw the same web research and is calibration-aware, but they
approach the problem from different angles (evidence-weighted, base-rate,
contrarian).

Your job is to ADJUDICATE, not average.

Procedure:
1. Identify which analyst made the strongest argument on THIS specific
   question. Different question structures favor different analysts:
   evidence-heavy questions favor the evidence analyst, novel-but-classifiable
   questions favor the base-rate analyst, consensus-prone questions favor the
   contrarian.
2. Spot any reasoning errors or hidden assumptions in the rationales.
3. Decide where the truth lies. If one analyst clearly dominates, weight your
   answer toward them. If the strongest analyst is still uncertain, hedge.
   Do NOT collapse to the simple mean — that defeats the purpose.

Calibration discipline:
- Probabilities below 0.10 or above 0.90 require an analyst making an
  overwhelming case that survives scrutiny.
- If the analysts disagree sharply AND none has a clearly stronger case,
  the right answer is closer to 0.5 — admit ignorance rather than pick a side.

Respond with ONLY a JSON object:
{"p_yes": <float 0.01-0.99>,
 "winner": "<strategy name with the strongest case, or 'none' if all weak>",
 "rationale": "<2-3 sentences: which case won and why; flag mistakes the
 others made>"}

No prose, no markdown fences, no commentary."""


def _deliberation_enabled() -> bool:
    """Read ``ENABLE_DELIBERATION`` from the env; default ``True``.

    Explicit disables: ``false``, ``0``, ``no``, ``off``, empty string.
    Anything else (including missing) → enabled.
    """
    raw = os.environ.get("ENABLE_DELIBERATION", "true").strip().lower()
    return raw not in {"false", "0", "no", "off", ""}


def _build_deliberation_prompt(
    event: EventRequest, estimates: list[Estimate]
) -> str:
    lines = [f"Event: {event.title}"]
    if event.outcomes and len(event.outcomes) >= 2:
        lines.append(
            f"OUTCOMES: YES = {event.outcomes[0]}, NO = {event.outcomes[1]}"
        )
    if event.category:
        lines.append(f"Category: {event.category}")
    if event.rules:
        lines.append(f"Resolution rules: {event.rules}")
    lines.append("")
    lines.append("Three independent estimates from the analysts:")
    lines.append("")
    for i, est in enumerate(estimates, 1):
        lines.append(
            f"{i}) {est.strategy}  p_yes={est.p_yes:.3f}  "
            f"confidence={est.confidence:.2f}"
        )
        lines.append(f"   Rationale: {est.rationale}")
        lines.append("")
    lines.append(
        "Adjudicate. Which analyst is most convincing on THIS specific "
        "question? Where does the final probability land? Return the JSON "
        "object only."
    )
    return "\n".join(lines)


def _deliberate(
    event: EventRequest, estimates: list[Estimate]
) -> Estimate | None:
    """Run one meta-LLM pass over the strategies' estimates.

    Returns a fourth :class:`Estimate` with ``strategy="deliberation"`` and
    confidence :data:`DELIBERATION_CONFIDENCE`, or ``None`` if the call fails
    or the LLM returned an unparseable payload. The caller treats ``None`` as
    "skip the deliberation round; proceed with the three originals."
    """
    if not estimates:
        return None
    user_prompt = _build_deliberation_prompt(event, estimates)
    try:
        data = call_llm_json(
            _DELIBERATION_SYSTEM_PROMPT,
            user_prompt,
            tier="reasoning",
            temperature=0.2,
            max_tokens=500,
        )
    except Exception as exc:  # noqa: BLE001 — never break the pipeline
        logger.warning("deliberation LLM call failed: %s", exc)
        return None

    try:
        p = clamp_probability(float(data["p_yes"]))
        rationale = str(data.get("rationale", "")).strip() or "(no rationale)"
        winner = str(data.get("winner", "")).strip()
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("deliberation bad payload: %s", exc)
        return None

    if winner:
        rationale = f"[winner: {winner}] {rationale}"
    return Estimate(
        p_yes=p,
        rationale=rationale,
        strategy=DELIBERATION_STRATEGY_NAME,
        confidence=clamp_confidence(DELIBERATION_CONFIDENCE),
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models (mirror the CLI predict contract)
# ---------------------------------------------------------------------------


class EventRequest(BaseModel):
    event_ticker: str | None = None
    market_ticker: str | None = None
    title: str
    subtitle: str | None = None
    description: str | None = None
    category: str | None = None
    rules: str | None = None
    close_time: str | None = None
    outcomes: list[str] | None = None
    resolved_outcome: Any | None = None


class PredictionResponse(BaseModel):
    p_yes: float
    rationale: str


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------


def _build_strategies() -> list[Any]:
    return [
        EvidenceWeightedStrategy(),
        BaseRateStrategy(),
        ContrarianStrategy(),
    ]


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _coerce_event(event: dict[str, Any]) -> EventRequest:
    close = event.get("close_time")
    if close is not None and not isinstance(close, str):
        close = str(close)
    raw_outcomes = event.get("outcomes")
    outcomes = list(raw_outcomes) if isinstance(raw_outcomes, list) else None
    return EventRequest(
        event_ticker=event.get("event_ticker"),
        market_ticker=event.get("market_ticker"),
        title=str(event.get("title") or "").strip() or "(untitled event)",
        subtitle=event.get("subtitle"),
        description=event.get("description"),
        category=event.get("category"),
        rules=event.get("rules"),
        close_time=close,
        outcomes=outcomes,
        resolved_outcome=event.get("resolved_outcome"),
    )


def _resolved_outcome_value(raw: Any) -> str | None:
    """Normalize ``resolved_outcome`` to a single outcome string, or None.

    Accepts the two shapes seen in the wild:
      * a bare string (live-event format)
      * a dict with a ``"value"`` key that is either a string or a list
        of strings (the dataset-registry shape).
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        return stripped or None
    if isinstance(raw, dict):
        value = raw.get("value")
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str):
                return first.strip() or None
    return None


def _run_strategy(strategy: Any, event: EventRequest, research: str) -> Estimate:
    try:
        return strategy.estimate(
            title=event.title,
            description=event.description,
            category=event.category,
            rules=event.rules,
            close_time=event.close_time,
            research=research,
            outcomes=event.outcomes,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("strategy %s crashed: %s", getattr(strategy, "name", "?"), exc)
        return failed_estimate(getattr(strategy, "name", "unknown"), str(exc))


def forecast_event(event: EventRequest) -> FinalPrediction:
    """Run the full ensemble pipeline for one event and return its prediction."""
    overall_start = time.perf_counter()

    # Phase 1: research
    research_start = time.perf_counter()
    research = research_event(
        title=event.title,
        description=event.description,
        category=event.category,
        rules=event.rules,
    )
    logger.info(
        "phase=research ticker=%s chars=%d elapsed=%.2fs",
        event.market_ticker,
        len(research),
        time.perf_counter() - research_start,
    )

    # Phase 2: parallel strategies
    strat_start = time.perf_counter()
    strategies = _build_strategies()
    estimates: list[Estimate] = []
    with ThreadPoolExecutor(max_workers=len(strategies)) as pool:
        futures = {
            pool.submit(_run_strategy, s, event, research): s for s in strategies
        }
        for fut in as_completed(futures):
            estimates.append(fut.result())
    logger.info(
        "phase=strategies ticker=%s n=%d elapsed=%.2fs",
        event.market_ticker,
        len(estimates),
        time.perf_counter() - strat_start,
    )
    for est in estimates:
        logger.info(
            "  strategy=%s p=%.3f conf=%.2f",
            est.strategy,
            est.p_yes,
            est.confidence,
        )

    # Phase 2.5: optional deliberation round — one meta-LLM call adjudicates
    # over the three analysts' estimates and produces a fourth estimate.
    if _deliberation_enabled():
        delib_start = time.perf_counter()
        delib = _deliberate(event, estimates)
        delib_elapsed = time.perf_counter() - delib_start
        if delib is not None:
            estimates.append(delib)
            logger.info(
                "phase=deliberation ticker=%s p=%.3f conf=%.2f elapsed=%.2fs",
                event.market_ticker,
                delib.p_yes,
                delib.confidence,
                delib_elapsed,
            )
        else:
            logger.info(
                "phase=deliberation ticker=%s skipped (call failed) elapsed=%.2fs",
                event.market_ticker,
                delib_elapsed,
            )
    else:
        logger.debug("phase=deliberation disabled via ENABLE_DELIBERATION")

    # Phase 3: ensemble + calibration
    ens_start = time.perf_counter()
    final = ensemble_predict(estimates)
    logger.info(
        "phase=ensemble ticker=%s p_yes=%.3f agreement=%.2f shrinkage=%.2f elapsed=%.2fs",
        event.market_ticker,
        final.p_yes,
        final.agreement,
        final.shrinkage,
        time.perf_counter() - ens_start,
    )
    logger.info(
        "phase=total ticker=%s elapsed=%.2fs",
        event.market_ticker,
        time.perf_counter() - overall_start,
    )
    return final


# ---------------------------------------------------------------------------
# Local entry point (used by `prophet forecast predict --local`)
# ---------------------------------------------------------------------------


def _prediction_delay_seconds() -> float:
    """Read PREDICTION_DELAY from the environment, defaulting to 5 seconds.

    Negative or unparseable values fall back to the default.
    """
    raw = os.environ.get("PREDICTION_DELAY", "5")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 5.0
    return max(0.0, value)


def _resolved_shortcut(event: dict[str, Any]) -> dict | None:
    """Return a finished prediction if the event is already resolved.

    When ``resolved_outcome`` is populated and ``outcomes[0]`` / ``outcomes[1]``
    name the YES and NO sides, the answer is known — there is no reason to
    burn LLM calls. Returns ``None`` if the event isn't conclusively resolved
    against the known outcomes.

    p_yes is clamped to the ``Prediction`` schema's ``[0.01, 0.99]`` bounds
    rather than 1.0 / 0.0 so the CLI's downstream validation accepts it.
    """
    resolved = _resolved_outcome_value(event.get("resolved_outcome"))
    if resolved is None:
        return None
    raw_outcomes = event.get("outcomes")
    if not isinstance(raw_outcomes, list) or len(raw_outcomes) < 2:
        return None

    yes_side = str(raw_outcomes[0]).strip()
    no_side = str(raw_outcomes[1]).strip()
    if resolved == yes_side:
        return {
            "p_yes": 0.99,
            "rationale": (
                f"Event already resolved: {resolved!r} matches outcomes[0]. "
                "Skipped research and strategies."
            ),
        }
    if resolved == no_side:
        return {
            "p_yes": 0.01,
            "rationale": (
                f"Event already resolved: {resolved!r} matches outcomes[1]. "
                "Skipped research and strategies."
            ),
        }
    return None


def predict(event: dict) -> dict:
    """CLI-facing prediction function.

    Accepts an event dict matching :class:`EventRequest` and returns the
    ``{"p_yes": float, "rationale": str}`` payload expected by
    ``prophet forecast predict``. Failures are swallowed and yield a safe
    fallback of ``p_yes=0.5``.

    Events whose ``resolved_outcome`` already matches one of the two
    ``outcomes`` entries are short-circuited: we return the known answer
    directly and skip research, strategies, and pacing.

    If caching is enabled (``ENABLE_CACHE=true``, the default) and a fresh
    entry exists for the event's ``market_ticker``, the cached payload is
    returned immediately — no research, no strategies, no pacing sleep.
    Successful predictions are cached for ``CACHE_TTL_HOURS`` hours (default
    6) so repeated polls from the evaluation harness reuse the same work.

    Otherwise, after each call this function sleeps for ``PREDICTION_DELAY``
    seconds (default 5, overridable via the ``PREDICTION_DELAY`` env var) so
    that callers iterating over many events stay within provider rate limits.
    """
    ticker = event.get("market_ticker")

    shortcut = _resolved_shortcut(event)
    if shortcut is not None:
        logger.info(
            "predict.shortcut ticker=%s p_yes=%.2f",
            ticker,
            shortcut["p_yes"],
        )
        return shortcut

    if cache_enabled():
        cached = get_cached_prediction(ticker)
        if cached is not None:
            logger.info(
                "predict.cache_hit ticker=%s p_yes=%.3f expires_at=%s",
                ticker,
                cached["p_yes"],
                cached.get("expires_at"),
            )
            # Cache hit means zero LLM work was done — skip pacing too.
            return {
                "p_yes": cached["p_yes"],
                "rationale": cached["rationale"],
            }

    succeeded = False
    try:
        event_req = _coerce_event(event)
        final = forecast_event(event_req)
        result = {"p_yes": final.p_yes, "rationale": final.rationale}
        # Only treat as success if at least one strategy survived the
        # confidence floor — an all-failed ensemble returns 0.5, which we
        # don't want to lock into the cache for 6 hours.
        succeeded = bool(final.estimates)
    except Exception as exc:  # noqa: BLE001 — never crash the CLI
        logger.exception("predict() failed: %s", exc)
        result = {
            "p_yes": 0.5,
            "rationale": f"Ensemble agent failed: {exc}. Defaulting to 0.5.",
        }

    if succeeded and cache_enabled():
        cache_prediction(ticker, result["p_yes"], result["rationale"])

    delay = _prediction_delay_seconds()
    if delay > 0:
        logger.info("predict.pacing sleeping=%.1fs", delay)
        time.sleep(delay)
    return result


# ---------------------------------------------------------------------------
# FastAPI server (used by `prophet forecast predict --agent-url`)
# ---------------------------------------------------------------------------


def _build_app() -> Any:
    """Lazily construct the FastAPI app so importing this module is cheap."""
    from fastapi import FastAPI

    fastapi_app = FastAPI(title="Ensemble Forecast Agent")

    @fastapi_app.post("/predict", response_model=PredictionResponse)
    async def predict_endpoint(event: EventRequest) -> PredictionResponse:
        logger.info(
            "endpoint.predict ticker=%s title=%s",
            event.market_ticker,
            event.title,
        )
        final = forecast_event(event)
        return PredictionResponse(p_yes=final.p_yes, rationale=final.rationale)

    return fastapi_app


# Module-level app instance for `uvicorn ai_prophet.forecast.ensemble_agent:app`.
app = _build_app()


def main() -> None:
    """Run the FastAPI server (development mode)."""
    import uvicorn

    host = os.environ.get("ENSEMBLE_HOST", "0.0.0.0")
    port = int(os.environ.get("ENSEMBLE_PORT", "8000"))
    uvicorn.run(
        "ai_prophet.forecast.ensemble_agent:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
