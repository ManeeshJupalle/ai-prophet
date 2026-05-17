"""Tests for the ensemble forecasting math and pipeline integration."""

from __future__ import annotations

import json
import math

import pytest
from ai_prophet.forecast.ensemble import (
    DEFAULT_SHRINKAGE,
    calibrate,
    compute_agreement,
    confidence_weighted_ensemble,
    ensemble_predict,
    inv_logit,
    logit,
)
from ai_prophet.forecast.strategies.base import Estimate


def _est(p: float, conf: float = 0.7, name: str = "s") -> Estimate:
    return Estimate(p_yes=p, rationale="", strategy=name, confidence=conf)


# ---------------------------------------------------------------------------
# logit / inv_logit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.05, 0.2, 0.5, 0.73, 0.9, 0.98])
def test_logit_inv_logit_roundtrip(p: float) -> None:
    assert inv_logit(logit(p)) == pytest.approx(p, abs=1e-9)


def test_logit_clamps_to_legal_range() -> None:
    # logit(0) would be -inf; we clamp to 0.01 first.
    assert logit(0.0) == pytest.approx(math.log(0.01 / 0.99), abs=1e-9)
    assert logit(1.0) == pytest.approx(math.log(0.99 / 0.01), abs=1e-9)


def test_inv_logit_handles_large_magnitude() -> None:
    # Numerically stable for both signs.
    assert inv_logit(1000.0) == pytest.approx(1.0, abs=1e-9)
    assert inv_logit(-1000.0) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def test_calibrate_zero_shrinkage_is_identity() -> None:
    for p in (0.05, 0.4, 0.9):
        assert calibrate(p, 0.0) == pytest.approx(p, abs=1e-9)


def test_calibrate_full_shrinkage_collapses_to_half() -> None:
    for p in (0.05, 0.4, 0.9):
        assert calibrate(p, 1.0) == pytest.approx(0.5, abs=1e-9)


def test_calibrate_specific_values() -> None:
    # Spec: calibrate(0.9, 0.2) ≈ 0.82  and  calibrate(0.5, anything) = 0.5
    assert calibrate(0.9, 0.2) == pytest.approx(0.82, abs=1e-9)
    assert calibrate(0.5, 0.0) == pytest.approx(0.5, abs=1e-9)
    assert calibrate(0.5, 0.3) == pytest.approx(0.5, abs=1e-9)
    assert calibrate(0.5, 1.0) == pytest.approx(0.5, abs=1e-9)


def test_calibrate_clamps_shrinkage() -> None:
    # Out-of-range shrinkage values should be clamped to [0, 1].
    assert calibrate(0.9, -0.5) == pytest.approx(0.9, abs=1e-9)
    assert calibrate(0.9, 2.0) == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# confidence_weighted_ensemble
# ---------------------------------------------------------------------------


def test_confidence_weighted_ensemble_equal_weights_gives_logit_mean() -> None:
    estimates = [_est(0.3, 0.5), _est(0.7, 0.5)]
    expected = inv_logit((logit(0.3) + logit(0.7)) / 2)
    assert confidence_weighted_ensemble(estimates) == pytest.approx(expected, abs=1e-9)


def test_confidence_weighted_ensemble_skews_to_higher_confidence() -> None:
    estimates = [_est(0.2, 0.1), _est(0.8, 0.9)]
    result = confidence_weighted_ensemble(estimates)
    # Should land much closer to 0.8 than to 0.2.
    assert result > 0.65


def test_confidence_weighted_ensemble_empty_input_returns_half() -> None:
    assert confidence_weighted_ensemble([]) == 0.5


def test_confidence_weighted_ensemble_all_zero_weights_returns_half() -> None:
    estimates = [_est(0.8, 0.0), _est(0.2, 0.0)]
    assert confidence_weighted_ensemble(estimates) == 0.5


# ---------------------------------------------------------------------------
# compute_agreement
# ---------------------------------------------------------------------------


def test_compute_agreement_identical_estimates_is_one() -> None:
    estimates = [_est(0.75, 0.6, "a"), _est(0.75, 0.6, "b"), _est(0.75, 0.6, "c")]
    assert compute_agreement(estimates) == pytest.approx(1.0, abs=1e-9)


def test_compute_agreement_falls_with_spread() -> None:
    tight = [_est(0.55, 0.6, "a"), _est(0.60, 0.6, "b"), _est(0.65, 0.6, "c")]
    wide = [_est(0.20, 0.6, "a"), _est(0.50, 0.6, "b"), _est(0.85, 0.6, "c")]
    assert compute_agreement(tight) > compute_agreement(wide)


def test_compute_agreement_single_estimate_is_one() -> None:
    assert compute_agreement([_est(0.3)]) == 1.0


def test_compute_agreement_in_unit_interval() -> None:
    estimates = [_est(0.01, 0.6, "a"), _est(0.99, 0.6, "b")]
    a = compute_agreement(estimates)
    assert 0.0 <= a <= 1.0


# ---------------------------------------------------------------------------
# ensemble_predict
# ---------------------------------------------------------------------------


def test_ensemble_predict_high_agreement_low_shrinkage() -> None:
    estimates = [
        _est(0.74, 0.7, "evidence_weighted"),
        _est(0.76, 0.7, "base_rate"),
        _est(0.75, 0.7, "contrarian"),
    ]
    result = ensemble_predict(estimates)
    assert result.agreement > 0.95
    # Shrinkage is at most half of DEFAULT when strategies fully agree.
    assert result.shrinkage <= DEFAULT_SHRINKAGE * 0.6
    # Final probability should stay near the consensus.
    assert abs(result.p_yes - 0.75) < 0.05


def test_ensemble_predict_low_agreement_higher_shrinkage() -> None:
    agree = [
        _est(0.80, 0.7, "a"),
        _est(0.79, 0.7, "b"),
        _est(0.81, 0.7, "c"),
    ]
    disagree = [
        _est(0.30, 0.7, "a"),
        _est(0.55, 0.7, "b"),
        _est(0.85, 0.7, "c"),
    ]
    high = ensemble_predict(agree)
    low = ensemble_predict(disagree)
    assert low.shrinkage > high.shrinkage
    assert low.agreement < high.agreement


def test_ensemble_predict_all_half_stays_half() -> None:
    estimates = [
        _est(0.5, 0.8, "a"),
        _est(0.5, 0.6, "b"),
        _est(0.5, 0.9, "c"),
    ]
    result = ensemble_predict(estimates)
    assert result.p_yes == pytest.approx(0.5, abs=1e-9)
    assert result.agreement == pytest.approx(1.0, abs=1e-9)


def test_ensemble_predict_single_estimate() -> None:
    result = ensemble_predict([_est(0.7, 0.8, "solo")])
    assert 0.01 <= result.p_yes <= 0.99
    # With one estimate, agreement is 1.0 and shrinkage halves.
    assert result.agreement == pytest.approx(1.0, abs=1e-9)
    assert result.shrinkage == pytest.approx(DEFAULT_SHRINKAGE * 0.5, abs=1e-9)


def test_ensemble_predict_all_failed_falls_back_to_half() -> None:
    estimates = [
        _est(0.5, 0.1, "a"),
        _est(0.2, 0.05, "b"),
        _est(0.9, 0.1, "c"),
    ]
    result = ensemble_predict(estimates)
    assert result.p_yes == 0.5
    assert result.estimates == []
    assert "failed" in result.rationale.lower()


def test_ensemble_predict_empty_input_falls_back_to_half() -> None:
    result = ensemble_predict([])
    assert result.p_yes == 0.5
    assert result.estimates == []


def test_ensemble_predict_clamps_to_legal_range() -> None:
    # Extreme inputs should still stay within Prediction's [0.01, 0.99] bounds.
    extreme = [
        _est(0.99, 1.0, "a"),
        _est(0.99, 1.0, "b"),
        _est(0.99, 1.0, "c"),
    ]
    result = ensemble_predict(extreme)
    assert 0.01 <= result.p_yes <= 0.99


def test_ensemble_predict_filters_low_confidence_estimates() -> None:
    # The 0.1-confidence outlier should be filtered out, leaving the two
    # high-confidence estimates to dominate.
    estimates = [
        _est(0.80, 0.8, "a"),
        _est(0.78, 0.8, "b"),
        _est(0.05, 0.1, "outlier"),
    ]
    result = ensemble_predict(estimates)
    strategies = {e.strategy for e in result.estimates}
    assert "outlier" not in strategies
    assert result.p_yes > 0.6


# ---------------------------------------------------------------------------
# End-to-end smoke test for the predict() entrypoint
# ---------------------------------------------------------------------------


def test_predict_returns_valid_payload_with_no_keys(monkeypatch) -> None:
    """When no API keys are configured, predict() must still return a valid
    response — the strategies will fail, and the ensemble falls back to 0.5.
    """
    from ai_prophet.forecast import ensemble_agent

    # Ensure no provider keys are available.
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    # Pacing sleep would slow the suite — disable it for this test.
    monkeypatch.setenv("PREDICTION_DELAY", "0")
    # Don't touch the project-root cache file.
    monkeypatch.setenv("ENABLE_CACHE", "false")
    # Prevent the researcher from making real HTTP calls.
    monkeypatch.setattr(
        "ai_prophet.forecast.ensemble_agent.research_event",
        lambda **_kw: "",
    )

    result = ensemble_agent.predict(
        {
            "event_ticker": "TEST",
            "market_ticker": "TEST-MKT",
            "title": "Will it rain tomorrow in Chicago?",
            "category": "weather",
            "close_time": "2026-12-31T23:59:59Z",
        }
    )
    assert set(result.keys()) == {"p_yes", "rationale"}
    assert 0.01 <= result["p_yes"] <= 0.99
    assert isinstance(result["rationale"], str)
    assert result["rationale"]


# ---------------------------------------------------------------------------
# Rate-limit retry + pacing
# ---------------------------------------------------------------------------


def test_call_with_rate_limit_retry_retries_once_on_429(monkeypatch) -> None:
    """A 429 from a provider triggers exactly one retry after a delay."""
    import httpx
    from ai_prophet.forecast import llm_utils

    sleeps: list[float] = []
    monkeypatch.setattr(llm_utils.time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    def fake_provider(system, user, temperature, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            response = httpx.Response(429, request=httpx.Request("POST", "http://x"))
            raise httpx.HTTPStatusError("rate limited", request=response.request, response=response)
        return "ok"

    result = llm_utils._call_with_rate_limit_retry(
        "groq", fake_provider, "sys", "usr", 0.2, 100
    )
    assert result == "ok"
    assert calls["n"] == 2
    assert sleeps == [llm_utils.RATE_LIMIT_RETRY_DELAY]


def test_call_with_rate_limit_retry_propagates_non_429(monkeypatch) -> None:
    """Non-429 errors are not retried — they fall straight through to dispatch."""
    from ai_prophet.forecast import llm_utils

    sleeps: list[float] = []
    monkeypatch.setattr(llm_utils.time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    def fake_provider(system, user, temperature, max_tokens):
        calls["n"] += 1
        raise RuntimeError("not a rate limit")

    with pytest.raises(RuntimeError):
        llm_utils._call_with_rate_limit_retry(
            "groq", fake_provider, "sys", "usr", 0.2, 100
        )
    assert calls["n"] == 1
    assert sleeps == []


def test_call_with_rate_limit_retry_gives_up_after_one_retry(monkeypatch) -> None:
    """Two consecutive 429s let the exception propagate to dispatch."""
    import httpx
    from ai_prophet.forecast import llm_utils

    monkeypatch.setattr(llm_utils.time, "sleep", lambda s: None)

    def always_429(system, user, temperature, max_tokens):
        response = httpx.Response(429, request=httpx.Request("POST", "http://x"))
        raise httpx.HTTPStatusError("rate limited", request=response.request, response=response)

    with pytest.raises(httpx.HTTPStatusError):
        llm_utils._call_with_rate_limit_retry(
            "groq", always_429, "sys", "usr", 0.2, 100
        )


def test_predict_sleeps_for_prediction_delay(monkeypatch) -> None:
    """predict() sleeps for PREDICTION_DELAY seconds after each event."""
    from ai_prophet.forecast import ensemble_agent

    monkeypatch.setenv("PREDICTION_DELAY", "3")
    monkeypatch.setenv("ENABLE_CACHE", "false")
    monkeypatch.setattr(
        "ai_prophet.forecast.ensemble_agent.research_event",
        lambda **_kw: "",
    )
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    sleeps: list[float] = []
    monkeypatch.setattr(ensemble_agent.time, "sleep", lambda s: sleeps.append(s))

    ensemble_agent.predict({"title": "x"})
    assert sleeps == [3.0]


def test_predict_delay_disabled_when_zero(monkeypatch) -> None:
    """PREDICTION_DELAY=0 means no sleep at all."""
    from ai_prophet.forecast import ensemble_agent

    monkeypatch.setenv("PREDICTION_DELAY", "0")
    monkeypatch.setenv("ENABLE_CACHE", "false")
    monkeypatch.setattr(
        "ai_prophet.forecast.ensemble_agent.research_event",
        lambda **_kw: "",
    )
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    sleeps: list[float] = []
    monkeypatch.setattr(ensemble_agent.time, "sleep", lambda s: sleeps.append(s))

    ensemble_agent.predict({"title": "x"})
    assert sleeps == []


def test_predict_delay_handles_garbage_env_value(monkeypatch) -> None:
    """An unparseable PREDICTION_DELAY falls back to the 5s default."""
    from ai_prophet.forecast import ensemble_agent

    monkeypatch.setenv("PREDICTION_DELAY", "not-a-number")
    monkeypatch.setenv("ENABLE_CACHE", "false")
    monkeypatch.setattr(
        "ai_prophet.forecast.ensemble_agent.research_event",
        lambda **_kw: "",
    )
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    sleeps: list[float] = []
    monkeypatch.setattr(ensemble_agent.time, "sleep", lambda s: sleeps.append(s))

    ensemble_agent.predict({"title": "x"})
    assert sleeps == [5.0]


# ---------------------------------------------------------------------------
# resolved_outcome shortcut + outcomes plumbing
# ---------------------------------------------------------------------------


def test_predict_shortcuts_on_resolved_yes(monkeypatch) -> None:
    """A resolved event matching outcomes[0] returns p_yes=0.99 with no LLM calls."""
    from ai_prophet.forecast import ensemble_agent

    def _explode(*_args, **_kwargs):
        raise AssertionError("resolved events must not hit the research/strategy path")

    monkeypatch.setattr(
        "ai_prophet.forecast.ensemble_agent.research_event", _explode
    )
    monkeypatch.setattr(
        "ai_prophet.forecast.ensemble_agent.forecast_event", _explode
    )

    result = ensemble_agent.predict(
        {
            "market_ticker": "TEST",
            "title": "Did A win?",
            "outcomes": ["A", "B"],
            "resolved_outcome": "A",
        }
    )
    assert result["p_yes"] == 0.99
    assert "resolved" in result["rationale"].lower()


def test_predict_shortcuts_on_resolved_no(monkeypatch) -> None:
    """A resolved event matching outcomes[1] returns p_yes=0.01."""
    from ai_prophet.forecast import ensemble_agent

    def _explode(*_args, **_kwargs):
        raise AssertionError("resolved events must not hit the research/strategy path")

    monkeypatch.setattr(
        "ai_prophet.forecast.ensemble_agent.research_event", _explode
    )
    monkeypatch.setattr(
        "ai_prophet.forecast.ensemble_agent.forecast_event", _explode
    )

    result = ensemble_agent.predict(
        {
            "market_ticker": "TEST",
            "title": "Did A win?",
            "outcomes": ["A", "B"],
            "resolved_outcome": "B",
        }
    )
    assert result["p_yes"] == 0.01


def test_predict_shortcut_accepts_dict_shaped_resolved_outcome(monkeypatch) -> None:
    """Datasets ship ``resolved_outcome`` as ``{"value": [name]}`` — handle that."""
    from ai_prophet.forecast import ensemble_agent

    monkeypatch.setattr(
        "ai_prophet.forecast.ensemble_agent.research_event",
        lambda **_kw: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    result = ensemble_agent.predict(
        {
            "market_ticker": "TEST",
            "title": "Match",
            "outcomes": ["Alice", "Bob"],
            "resolved_outcome": {"value": ["Bob"], "source": "X"},
        }
    )
    assert result["p_yes"] == 0.01


def test_predict_no_shortcut_when_resolved_outcome_unknown_value(monkeypatch) -> None:
    """If resolved_outcome doesn't match either side, fall through to full pipeline."""
    from ai_prophet.forecast import ensemble_agent

    monkeypatch.setenv("PREDICTION_DELAY", "0")
    monkeypatch.setenv("ENABLE_CACHE", "false")
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "ai_prophet.forecast.ensemble_agent.research_event",
        lambda **_kw: "",
    )

    result = ensemble_agent.predict(
        {
            "market_ticker": "TEST",
            "title": "Match",
            "outcomes": ["Alice", "Bob"],
            "resolved_outcome": "Carol",  # not in outcomes
        }
    )
    # Full pipeline ran with no API keys -> graceful fallback to 0.5
    assert result["p_yes"] == 0.5


def test_strategy_prompt_includes_outcomes_line() -> None:
    """All three strategies render the OUTCOMES line when outcomes are present."""
    from ai_prophet.forecast.strategies.base_rate import (
        _build_user_prompt as base_rate_prompt,
    )
    from ai_prophet.forecast.strategies.contrarian import (
        _build_user_prompt as contrarian_prompt,
    )
    from ai_prophet.forecast.strategies.evidence import (
        _build_user_prompt as evidence_prompt,
    )

    kwargs = {
        "title": "Match",
        "description": None,
        "category": "Sports",
        "rules": None,
        "close_time": None,
        "research": "",
        "outcomes": ["Lakers", "Thunder"],
    }
    for fn in (evidence_prompt, base_rate_prompt, contrarian_prompt):
        rendered = fn(**kwargs)
        assert "OUTCOMES: YES = Lakers, NO = Thunder" in rendered


def test_strategy_prompt_omits_outcomes_line_when_missing() -> None:
    """When outcomes is None or too short, no OUTCOMES line is added."""
    from ai_prophet.forecast.strategies.evidence import _build_user_prompt

    for outcomes in (None, [], ["only-one"]):
        rendered = _build_user_prompt(
            title="t",
            description=None,
            category=None,
            rules=None,
            close_time=None,
            research="",
            outcomes=outcomes,
        )
        assert "OUTCOMES:" not in rendered


# ---------------------------------------------------------------------------
# Deliberation round
# ---------------------------------------------------------------------------


def _fake_strategy_estimates() -> dict[str, Estimate]:
    return {
        "evidence_weighted": Estimate(
            p_yes=0.70, rationale="ew", strategy="evidence_weighted", confidence=0.70
        ),
        "base_rate": Estimate(
            p_yes=0.65, rationale="br", strategy="base_rate", confidence=0.60
        ),
        "contrarian": Estimate(
            p_yes=0.75, rationale="co", strategy="contrarian", confidence=0.70
        ),
    }


def test_deliberation_enabled_flag_parsing(monkeypatch) -> None:
    """ENABLE_DELIBERATION env var parses correctly."""
    from ai_prophet.forecast.ensemble_agent import _deliberation_enabled

    monkeypatch.delenv("ENABLE_DELIBERATION", raising=False)
    assert _deliberation_enabled() is True  # default

    for val in ("false", "FALSE", "0", "no", "off", ""):
        monkeypatch.setenv("ENABLE_DELIBERATION", val)
        assert _deliberation_enabled() is False, f"{val!r} should disable"

    for val in ("true", "TRUE", "1", "yes", "on", "anything"):
        monkeypatch.setenv("ENABLE_DELIBERATION", val)
        assert _deliberation_enabled() is True, f"{val!r} should enable"


def test_deliberation_called_and_added_to_ensemble(monkeypatch) -> None:
    """When enabled, _deliberate runs after strategies and its estimate is in the ensemble."""
    from ai_prophet.forecast import ensemble_agent

    monkeypatch.setenv("ENABLE_DELIBERATION", "true")
    monkeypatch.setattr(ensemble_agent, "research_event", lambda **_kw: "")

    table = _fake_strategy_estimates()
    monkeypatch.setattr(
        ensemble_agent,
        "_run_strategy",
        lambda strategy, event, research, temporal_ctx=None: table[strategy.name],
    )

    captured_calls: list[list[Estimate]] = []

    def fake_deliberate(event, estimates, temporal_ctx=None):
        captured_calls.append(list(estimates))
        return Estimate(
            p_yes=0.72, rationale="meta", strategy="deliberation", confidence=0.85
        )

    monkeypatch.setattr(ensemble_agent, "_deliberate", fake_deliberate)

    event = ensemble_agent.EventRequest(title="test event")
    final = ensemble_agent.forecast_event(event)

    # Deliberation was called exactly once
    assert len(captured_calls) == 1
    # ... with the three strategy estimates as input
    assert len(captured_calls[0]) == 3
    assert {e.strategy for e in captured_calls[0]} == {
        "evidence_weighted",
        "base_rate",
        "contrarian",
    }
    # Final ensemble includes all four estimates
    assert len(final.estimates) == 4
    assert "deliberation" in {e.strategy for e in final.estimates}


def test_deliberation_skipped_when_env_false(monkeypatch) -> None:
    """ENABLE_DELIBERATION=false skips the deliberation round entirely."""
    from ai_prophet.forecast import ensemble_agent

    monkeypatch.setenv("ENABLE_DELIBERATION", "false")
    monkeypatch.setattr(ensemble_agent, "research_event", lambda **_kw: "")

    table = _fake_strategy_estimates()
    monkeypatch.setattr(
        ensemble_agent,
        "_run_strategy",
        lambda strategy, event, research, temporal_ctx=None: table[strategy.name],
    )

    invocations: list[bool] = []

    def fake_deliberate(event, estimates):
        invocations.append(True)
        return None

    monkeypatch.setattr(ensemble_agent, "_deliberate", fake_deliberate)

    event = ensemble_agent.EventRequest(title="test event")
    final = ensemble_agent.forecast_event(event)

    assert invocations == []  # _deliberate never called
    assert len(final.estimates) == 3
    assert "deliberation" not in {e.strategy for e in final.estimates}


def test_deliberation_failure_falls_back_to_three_estimates(monkeypatch) -> None:
    """When _deliberate returns None, the ensemble proceeds on the 3 originals."""
    from ai_prophet.forecast import ensemble_agent

    monkeypatch.setenv("ENABLE_DELIBERATION", "true")
    monkeypatch.setattr(ensemble_agent, "research_event", lambda **_kw: "")

    table = _fake_strategy_estimates()
    monkeypatch.setattr(
        ensemble_agent,
        "_run_strategy",
        lambda strategy, event, research, temporal_ctx=None: table[strategy.name],
    )
    monkeypatch.setattr(
        ensemble_agent,
        "_deliberate",
        lambda event, estimates, temporal_ctx=None: None,
    )

    event = ensemble_agent.EventRequest(title="test event")
    final = ensemble_agent.forecast_event(event)

    assert len(final.estimates) == 3
    assert "deliberation" not in {e.strategy for e in final.estimates}


def test_deliberate_uses_reasoning_tier_and_includes_outcomes(monkeypatch) -> None:
    """_deliberate routes to the reasoning tier and surfaces outcomes in the prompt."""
    from ai_prophet.forecast import ensemble_agent

    captured: list[dict] = []

    def fake_call(system, user, *, tier, temperature, max_tokens):
        captured.append({"tier": tier, "system": system, "user": user})
        return {
            "p_yes": 0.6,
            "rationale": "stub",
            "winner": "evidence_weighted",
        }

    monkeypatch.setattr(ensemble_agent, "call_llm_json", fake_call)

    event = ensemble_agent.EventRequest(
        title="Will Lakers beat Thunder?",
        outcomes=["Lakers", "Thunder"],
    )
    estimates = [
        Estimate(p_yes=0.7, rationale="r1", strategy="evidence_weighted", confidence=0.7),
        Estimate(p_yes=0.5, rationale="r2", strategy="base_rate", confidence=0.6),
        Estimate(p_yes=0.4, rationale="r3", strategy="contrarian", confidence=0.5),
    ]
    delib = ensemble_agent._deliberate(event, estimates)

    assert delib is not None
    assert delib.strategy == "deliberation"
    assert delib.p_yes == pytest.approx(0.6)
    assert delib.confidence == pytest.approx(0.85)
    assert len(captured) == 1
    assert captured[0]["tier"] == "reasoning"
    # Prompt surfaces the OUTCOMES line and every strategy by name.
    user = captured[0]["user"]
    assert "OUTCOMES: YES = Lakers, NO = Thunder" in user
    for name in ("evidence_weighted", "base_rate", "contrarian"):
        assert name in user
    # Winner is captured in the rationale prefix.
    assert "winner: evidence_weighted" in delib.rationale


def test_deliberate_returns_none_on_llm_failure(monkeypatch) -> None:
    """An LLM exception during deliberation returns None, not a crash."""
    from ai_prophet.forecast import ensemble_agent
    from ai_prophet.forecast.llm_utils import LLMError

    def boom(*_args, **_kwargs):
        raise LLMError("simulated chain failure")

    monkeypatch.setattr(ensemble_agent, "call_llm_json", boom)

    event = ensemble_agent.EventRequest(title="t")
    estimates = [
        Estimate(p_yes=0.5, rationale="r", strategy="x", confidence=0.5),
    ]
    assert ensemble_agent._deliberate(event, estimates) is None


def test_deliberate_returns_none_on_bad_payload(monkeypatch) -> None:
    """An LLM response missing p_yes returns None."""
    from ai_prophet.forecast import ensemble_agent

    monkeypatch.setattr(
        ensemble_agent,
        "call_llm_json",
        lambda *_a, **_kw: {"rationale": "no p_yes"},
    )

    event = ensemble_agent.EventRequest(title="t")
    estimates = [
        Estimate(p_yes=0.5, rationale="r", strategy="x", confidence=0.5),
    ]
    assert ensemble_agent._deliberate(event, estimates) is None


# ---------------------------------------------------------------------------
# Prediction cache integration
# ---------------------------------------------------------------------------


def _isolate_cache(monkeypatch, tmp_path) -> str:
    """Point the cache at a tmp path so tests don't share state."""
    cache_file = str(tmp_path / "cache.json")
    monkeypatch.setenv("PREDICTION_CACHE_PATH", cache_file)
    return cache_file


def test_predict_cache_hit_skips_pipeline(monkeypatch, tmp_path) -> None:
    """A fresh cache entry short-circuits the entire pipeline."""
    from ai_prophet.forecast import cache as cache_mod
    from ai_prophet.forecast import ensemble_agent

    _isolate_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_CACHE", "true")
    monkeypatch.setenv("PREDICTION_DELAY", "0")

    cache_mod.cache_prediction("CACHED-TICKER", 0.61, "from cache")

    def explode(*_a, **_kw):
        raise AssertionError("pipeline must not run on cache hit")

    monkeypatch.setattr(ensemble_agent, "forecast_event", explode)
    monkeypatch.setattr(ensemble_agent, "research_event", explode)

    result = ensemble_agent.predict(
        {"market_ticker": "CACHED-TICKER", "title": "anything"}
    )
    assert result["p_yes"] == pytest.approx(0.61)
    assert result["rationale"] == "from cache"


def test_predict_cache_miss_runs_pipeline_and_caches(monkeypatch, tmp_path) -> None:
    """A cache miss runs the pipeline and writes the result back to disk."""
    from ai_prophet.forecast import cache as cache_mod
    from ai_prophet.forecast import ensemble_agent
    from ai_prophet.forecast.ensemble import FinalPrediction

    _isolate_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_CACHE", "true")
    monkeypatch.setenv("PREDICTION_DELAY", "0")

    fake_estimate = Estimate(
        p_yes=0.73, rationale="r", strategy="s", confidence=0.7
    )
    fake_final = FinalPrediction(
        p_yes=0.73,
        rationale="ensembled",
        raw_p_yes=0.73,
        agreement=1.0,
        shrinkage=0.05,
        estimates=[fake_estimate],
    )
    monkeypatch.setattr(
        ensemble_agent, "forecast_event", lambda _event: fake_final
    )

    assert cache_mod.get_cached_prediction("FRESH-TICKER") is None
    result = ensemble_agent.predict(
        {"market_ticker": "FRESH-TICKER", "title": "t"}
    )
    assert result["p_yes"] == pytest.approx(0.73)
    # Result was written back to the cache.
    cached = cache_mod.get_cached_prediction("FRESH-TICKER")
    assert cached is not None
    assert cached["p_yes"] == pytest.approx(0.73)
    assert cached["rationale"] == "ensembled"


def test_predict_expired_entry_is_refreshed(monkeypatch, tmp_path) -> None:
    """A cache entry past its expires_at triggers a fresh pipeline run."""
    import json
    from datetime import UTC, datetime, timedelta
    from pathlib import Path

    from ai_prophet.forecast import ensemble_agent
    from ai_prophet.forecast.ensemble import FinalPrediction

    cache_file = _isolate_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_CACHE", "true")
    monkeypatch.setenv("PREDICTION_DELAY", "0")

    # Pre-populate with an entry that expired an hour ago.
    past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    Path(cache_file).write_text(
        json.dumps(
            {
                "STALE-TICKER": {
                    "p_yes": 0.99,
                    "rationale": "ancient history",
                    "timestamp": past,
                    "expires_at": past,
                }
            }
        ),
        encoding="utf-8",
    )

    fresh_final = FinalPrediction(
        p_yes=0.34,
        rationale="freshly computed",
        raw_p_yes=0.34,
        agreement=1.0,
        shrinkage=0.05,
        estimates=[Estimate(p_yes=0.34, rationale="r", strategy="s", confidence=0.7)],
    )
    monkeypatch.setattr(
        ensemble_agent, "forecast_event", lambda _event: fresh_final
    )

    result = ensemble_agent.predict(
        {"market_ticker": "STALE-TICKER", "title": "t"}
    )
    # The expired entry was ignored; the fresh prediction returned.
    assert result["p_yes"] == pytest.approx(0.34)
    assert result["rationale"] == "freshly computed"


def test_predict_does_not_cache_when_pipeline_falls_back_to_half(
    monkeypatch, tmp_path
) -> None:
    """When every strategy fails, the 0.5 fallback must NOT be cached."""
    from ai_prophet.forecast import cache as cache_mod
    from ai_prophet.forecast import ensemble_agent
    from ai_prophet.forecast.ensemble import FinalPrediction

    _isolate_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_CACHE", "true")
    monkeypatch.setenv("PREDICTION_DELAY", "0")

    # An all-failed ensemble has empty estimates (post-filter).
    failed_final = FinalPrediction(
        p_yes=0.5,
        rationale="all strategies failed",
        raw_p_yes=0.5,
        agreement=0.0,
        shrinkage=0.15,
        estimates=[],
    )
    monkeypatch.setattr(
        ensemble_agent, "forecast_event", lambda _event: failed_final
    )

    result = ensemble_agent.predict(
        {"market_ticker": "DOOMED", "title": "t"}
    )
    assert result["p_yes"] == 0.5
    assert cache_mod.get_cached_prediction("DOOMED") is None


def test_predict_with_cache_disabled_skips_both_read_and_write(
    monkeypatch, tmp_path
) -> None:
    """ENABLE_CACHE=false bypasses the cache for both reads and writes."""
    from ai_prophet.forecast import cache as cache_mod
    from ai_prophet.forecast import ensemble_agent
    from ai_prophet.forecast.ensemble import FinalPrediction

    _isolate_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_CACHE", "false")
    monkeypatch.setenv("PREDICTION_DELAY", "0")

    # Pre-populate the cache with a value we should NOT see.
    cache_mod.cache_prediction("X", 0.99, "should be ignored")

    fresh_final = FinalPrediction(
        p_yes=0.42,
        rationale="pipeline ran",
        raw_p_yes=0.42,
        agreement=1.0,
        shrinkage=0.05,
        estimates=[Estimate(p_yes=0.42, rationale="r", strategy="s", confidence=0.7)],
    )
    monkeypatch.setattr(
        ensemble_agent, "forecast_event", lambda _event: fresh_final
    )

    result = ensemble_agent.predict({"market_ticker": "X", "title": "t"})
    # Pipeline ran (didn't return the cached 0.99).
    assert result["p_yes"] == pytest.approx(0.42)
    # Original cache entry is unchanged — predict() didn't overwrite it
    # because caching is off.
    cached = cache_mod.get_cached_prediction("X")
    assert cached is not None
    assert cached["p_yes"] == pytest.approx(0.99)


def test_predict_without_market_ticker_skips_cache(monkeypatch, tmp_path) -> None:
    """Events with no market_ticker can still be predicted; cache is a no-op."""
    from ai_prophet.forecast import ensemble_agent
    from ai_prophet.forecast.ensemble import FinalPrediction

    _isolate_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_CACHE", "true")
    monkeypatch.setenv("PREDICTION_DELAY", "0")

    fresh_final = FinalPrediction(
        p_yes=0.55,
        rationale="r",
        raw_p_yes=0.55,
        agreement=1.0,
        shrinkage=0.05,
        estimates=[Estimate(p_yes=0.55, rationale="r", strategy="s", confidence=0.7)],
    )
    monkeypatch.setattr(
        ensemble_agent, "forecast_event", lambda _event: fresh_final
    )

    result = ensemble_agent.predict({"title": "no ticker"})
    assert result["p_yes"] == pytest.approx(0.55)


def test_predict_cache_hit_skips_pacing_sleep(monkeypatch, tmp_path) -> None:
    """A cache hit does no work, so it must not invoke PREDICTION_DELAY."""
    from ai_prophet.forecast import cache as cache_mod
    from ai_prophet.forecast import ensemble_agent

    _isolate_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_CACHE", "true")
    monkeypatch.setenv("PREDICTION_DELAY", "5")

    cache_mod.cache_prediction("FAST", 0.65, "r")

    sleeps: list[float] = []
    monkeypatch.setattr(ensemble_agent.time, "sleep", lambda s: sleeps.append(s))

    ensemble_agent.predict({"market_ticker": "FAST", "title": "t"})
    assert sleeps == []


# ---------------------------------------------------------------------------
# Temporal factor integration with the ensemble
# ---------------------------------------------------------------------------


def test_ensemble_predict_with_temporal_factor_changes_shrinkage() -> None:
    """temporal_factor scales the base shrinkage before agreement adjustment."""
    # Strategies disagree (so agreement-based shrinkage doesn't dominate).
    estimates = [
        _est(0.20, 0.7, "a"),
        _est(0.55, 0.7, "b"),
        _est(0.85, 0.7, "c"),
    ]
    no_temporal = ensemble_predict(estimates, base_shrinkage=0.20)
    imminent = ensemble_predict(estimates, base_shrinkage=0.20, temporal_factor=1.0)
    far = ensemble_predict(estimates, base_shrinkage=0.20, temporal_factor=0.3)

    # Imminent → less shrinkage than the no-temporal baseline.
    assert imminent.shrinkage < no_temporal.shrinkage
    # Far → slightly less than baseline too (factor 0.3 still trims base),
    # but more shrinkage than the imminent case.
    assert far.shrinkage > imminent.shrinkage
    assert far.shrinkage <= no_temporal.shrinkage


def test_ensemble_predict_imminent_more_decisive_than_far() -> None:
    """For the same disagreeing estimates, imminent stays closer to raw_p."""
    # Estimates lean YES (mean > 0.5) so less shrinkage → result is further from 0.5.
    estimates = [
        _est(0.85, 0.7, "a"),
        _est(0.80, 0.7, "b"),
        _est(0.40, 0.7, "c"),
    ]
    imminent = ensemble_predict(estimates, base_shrinkage=0.20, temporal_factor=1.0)
    far = ensemble_predict(estimates, base_shrinkage=0.20, temporal_factor=0.3)

    # Imminent should be further from 0.5 than far (less shrinkage applied).
    assert abs(imminent.p_yes - 0.5) > abs(far.p_yes - 0.5)


def test_ensemble_predict_clamps_invalid_temporal_factor() -> None:
    """Out-of-range temporal_factor is clamped to [0, 1]."""
    estimates = [_est(0.8, 0.8, "x")]
    a = ensemble_predict(estimates, base_shrinkage=0.20, temporal_factor=2.0)
    b = ensemble_predict(estimates, base_shrinkage=0.20, temporal_factor=1.0)
    # Both should produce the same effective shrinkage.
    assert a.shrinkage == pytest.approx(b.shrinkage)

    c = ensemble_predict(estimates, base_shrinkage=0.20, temporal_factor=-0.5)
    d = ensemble_predict(estimates, base_shrinkage=0.20, temporal_factor=0.0)
    assert c.shrinkage == pytest.approx(d.shrinkage)


def test_ensemble_predict_default_temporal_matches_legacy_behaviour() -> None:
    """Omitting temporal_factor produces the same result as before this feature."""
    estimates = [
        _est(0.7, 0.6, "a"),
        _est(0.65, 0.6, "b"),
        _est(0.75, 0.6, "c"),
    ]
    without = ensemble_predict(estimates, base_shrinkage=0.15)
    # Manually compute legacy shrinkage = base * (1 - agreement * 0.5)
    assert without.shrinkage == pytest.approx(
        0.15 * (1.0 - without.agreement * 0.5), abs=1e-9
    )


def test_strategy_prompts_include_temporal_context_when_provided() -> None:
    """All three strategies inject the temporal_context line into the user prompt."""
    from ai_prophet.forecast.strategies.base_rate import (
        _build_user_prompt as base_rate_prompt,
    )
    from ai_prophet.forecast.strategies.contrarian import (
        _build_user_prompt as contrarian_prompt,
    )
    from ai_prophet.forecast.strategies.evidence import (
        _build_user_prompt as evidence_prompt,
    )

    ctx = "TIME HORIZON: this event closes in 6 hours. Be decisive."
    kwargs = {
        "title": "t",
        "description": None,
        "category": "Sports",
        "rules": None,
        "close_time": None,
        "research": "",
        "outcomes": ["A", "B"],
        "temporal_context": ctx,
    }
    for fn in (evidence_prompt, base_rate_prompt, contrarian_prompt):
        rendered = fn(**kwargs)
        assert ctx in rendered


def test_strategy_prompts_omit_temporal_context_when_none() -> None:
    """A None temporal_context leaves no TIME HORIZON line in the prompt."""
    from ai_prophet.forecast.strategies.evidence import _build_user_prompt

    rendered = _build_user_prompt(
        title="t",
        description=None,
        category=None,
        rules=None,
        close_time=None,
        research="",
        outcomes=None,
        temporal_context=None,
    )
    assert "TIME HORIZON" not in rendered


def test_health_endpoint_returns_ok() -> None:
    """GET /health is wired so Railway's liveness probe sees a 200."""
    from ai_prophet.forecast.ensemble_agent import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ensemble-forecast-agent"


def _stub_pipeline(monkeypatch, tmp_path, p_yes: float = 0.6234) -> None:
    """Common setup for FastAPI endpoint tests: isolate cache + stub the pipeline."""
    from ai_prophet.forecast import ensemble_agent
    from ai_prophet.forecast.ensemble import FinalPrediction

    monkeypatch.setenv("PREDICTION_CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setenv("ENABLE_CACHE", "true")
    monkeypatch.setenv("PREDICTION_DELAY", "0")

    fake_final = FinalPrediction(
        p_yes=p_yes,
        rationale="r",
        raw_p_yes=p_yes,
        agreement=1.0,
        shrinkage=0.05,
        estimates=[Estimate(p_yes=p_yes, rationale="r", strategy="s", confidence=0.7)],
    )
    monkeypatch.setattr(ensemble_agent, "forecast_event", lambda _e: fake_final)


def test_predict_endpoint_logs_request_and_response(
    caplog, monkeypatch, tmp_path
) -> None:
    """POST /predict writes a request-entry and a response-exit log line."""
    import logging

    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    _stub_pipeline(monkeypatch, tmp_path, p_yes=0.6234)

    client = TestClient(ensemble_agent.app)
    caplog.set_level(logging.INFO, logger="ai_prophet.forecast.ensemble_agent")

    long_title = (
        "Will Cleveland beat Detroit in NBA Eastern Conference Game 6 on May 15, 2026?"
    )
    resp = client.post(
        "/predict",
        json={
            "event_ticker": "EVT-1",
            "market_ticker": "MKT-1",
            "title": long_title,
            "category": "Sports",
            "close_time": "2026-05-31T23:59:59Z",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["p_yes"] == pytest.approx(0.6234)

    endpoint_msgs = [
        r.message for r in caplog.records if "endpoint.predict" in r.message
    ]
    request_lines = [m for m in endpoint_msgs if "request" in m]
    response_lines = [m for m in endpoint_msgs if "response" in m]
    assert request_lines, "missing endpoint.predict request log"
    assert response_lines, "missing endpoint.predict response log"

    req = request_lines[-1]
    assert "event_ticker=EVT-1" in req
    assert "market_ticker=MKT-1" in req
    # The title in the log is truncated to 60 chars.
    assert long_title[:60] in req
    assert long_title not in req  # full string wasn't logged

    rsp = response_lines[-1]
    assert "event_ticker=EVT-1" in rsp
    assert "market_ticker=MKT-1" in rsp
    assert "p_yes=0.6234" in rsp


# ---------------------------------------------------------------------------
# /predict — batch mode + /predictions alias
# ---------------------------------------------------------------------------


def test_predict_accepts_list_and_returns_list(monkeypatch, tmp_path) -> None:
    """POST /predict with a JSON list returns a list of predictions."""
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    _stub_pipeline(monkeypatch, tmp_path, p_yes=0.55)

    client = TestClient(ensemble_agent.app)
    payload = [
        {
            "event_ticker": "E1",
            "market_ticker": "M1",
            "title": "Event one?",
            "category": "Sports",
            "close_time": "2026-06-01T00:00:00Z",
        },
        {
            "event_ticker": "E2",
            "market_ticker": "M2",
            "title": "Event two?",
            "category": "Crypto",
            "close_time": "2026-06-01T00:00:00Z",
        },
    ]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 2
    for item in body:
        assert set(item.keys()) >= {"p_yes", "rationale"}
        assert item["p_yes"] == pytest.approx(0.55)


def test_predict_single_dict_still_returns_dict(monkeypatch, tmp_path) -> None:
    """Backward-compat: a single dict body returns a single dict."""
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    _stub_pipeline(monkeypatch, tmp_path, p_yes=0.42)

    client = TestClient(ensemble_agent.app)
    resp = client.post(
        "/predict",
        json={
            "event_ticker": "E",
            "market_ticker": "M",
            "title": "One event?",
            "category": "Sports",
            "close_time": "2026-06-01T00:00:00Z",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)
    assert body["p_yes"] == pytest.approx(0.42)


def test_predict_multi_outcome_returns_probabilities_array(
    monkeypatch, tmp_path
) -> None:
    """3+ outcome events return {probabilities: [{market, probability}], rationale}.

    The forecasting harness rejects responses missing a ``probabilities``
    array on distribution events, so the /predict route must emit the
    array shape (not a dict) and include every market the event listed.
    """
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    monkeypatch.setenv("PREDICTION_CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setenv("ENABLE_CACHE", "false")

    # Skip the network: stub research and the multi-outcome LLM call.
    monkeypatch.setattr(
        ensemble_agent, "research_event", lambda **_kw: "(stubbed research)"
    )
    monkeypatch.setattr(
        ensemble_agent,
        "_predict_multi_outcome",
        lambda title, markets, research: {
            "rationale": "stub rationale",
            "probabilities": {
                "Boston Celtics": 0.40,
                "Denver Nuggets": 0.25,
                "Minnesota Timberwolves": 0.20,
                "Indiana Pacers": 0.15,
            },
        },
    )

    client = TestClient(ensemble_agent.app)
    outcomes = [
        "Boston Celtics",
        "Denver Nuggets",
        "Minnesota Timberwolves",
        "Indiana Pacers",
    ]
    resp = client.post(
        "/predict",
        json={
            "event_ticker": "NBA-2026",
            "market_ticker": "NBA-CHAMP-2026",
            "title": "Who will win the 2026 NBA championship?",
            "category": "Sports",
            "close_time": "2026-06-30T00:00:00Z",
            "outcomes": outcomes,
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    # Shape: distribution response, not binary.
    assert "probabilities" in body
    assert "p_yes" not in body
    probs = body["probabilities"]
    assert isinstance(probs, list)
    assert len(probs) == 4

    # Every market present, with the right keys.
    returned_markets = {p["market"] for p in probs}
    assert returned_markets == set(outcomes)
    for p in probs:
        assert set(p.keys()) == {"market", "probability"}
        assert 0.0 <= p["probability"] <= 1.0

    # Sum-normalized to a valid distribution.
    total = sum(p["probability"] for p in probs)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_predict_multi_outcome_uses_cache_on_repeat(monkeypatch, tmp_path) -> None:
    """A repeat call with the same market_ticker should skip the LLM and serve cache."""
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    monkeypatch.setenv("PREDICTION_CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setenv("ENABLE_CACHE", "true")

    monkeypatch.setattr(
        ensemble_agent, "research_event", lambda **_kw: "(stubbed)"
    )

    call_count = {"n": 0}

    def fake_multi(title, markets, research):
        call_count["n"] += 1
        return {
            "rationale": f"call #{call_count['n']}",
            "probabilities": {m: 1.0 / len(markets) for m in markets},
        }

    monkeypatch.setattr(ensemble_agent, "_predict_multi_outcome", fake_multi)

    client = TestClient(ensemble_agent.app)
    payload = {
        "event_ticker": "NBA-2026",
        "market_ticker": "NBA-CHAMP-2026",
        "title": "Who will win?",
        "outcomes": ["A", "B", "C", "D"],
    }

    first = client.post("/predict", json=payload)
    second = client.post("/predict", json=payload)
    third = client.post("/predict", json=payload)

    assert first.status_code == second.status_code == third.status_code == 200
    # LLM ran exactly once across the three calls — subsequent calls served from cache.
    assert call_count["n"] == 1
    # Shape is preserved on cache hits.
    for resp in (first, second, third):
        body = resp.json()
        assert isinstance(body["probabilities"], list)
        assert len(body["probabilities"]) == 4
        assert all("market" in p and "probability" in p for p in body["probabilities"])


def test_predict_two_outcomes_still_returns_p_yes(monkeypatch, tmp_path) -> None:
    """Binary (2-outcome) events keep returning the legacy {p_yes, rationale} shape."""
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    _stub_pipeline(monkeypatch, tmp_path, p_yes=0.73)

    client = TestClient(ensemble_agent.app)
    resp = client.post(
        "/predict",
        json={
            "event_ticker": "E",
            "market_ticker": "M",
            "title": "Will X happen?",
            "category": "Other",
            "outcomes": ["Yes", "No"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "p_yes" in body
    assert "probabilities" not in body
    assert body["p_yes"] == pytest.approx(0.73)


def test_predictions_endpoint_aliases_predict(monkeypatch, tmp_path) -> None:
    """/predictions behaves identically to /predict in both modes."""
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    _stub_pipeline(monkeypatch, tmp_path, p_yes=0.31)

    client = TestClient(ensemble_agent.app)

    # Single mode
    single_resp = client.post(
        "/predictions",
        json={"event_ticker": "E", "market_ticker": "M", "title": "x"},
    )
    assert single_resp.status_code == 200
    assert single_resp.json()["p_yes"] == pytest.approx(0.31)
    assert isinstance(single_resp.json(), dict)

    # Batch mode
    batch_resp = client.post(
        "/predictions",
        json=[
            {"event_ticker": "E1", "market_ticker": "M1", "title": "a"},
            {"event_ticker": "E2", "market_ticker": "M2", "title": "b"},
        ],
    )
    assert batch_resp.status_code == 200
    body = batch_resp.json()
    assert isinstance(body, list)
    assert len(body) == 2


def test_predict_batch_with_empty_list_returns_empty_list(
    monkeypatch, tmp_path
) -> None:
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    _stub_pipeline(monkeypatch, tmp_path)

    client = TestClient(ensemble_agent.app)
    resp = client.post("/predict", json=[])
    assert resp.status_code == 200
    assert resp.json() == []


def test_predict_batch_skips_non_dict_items_gracefully(
    monkeypatch, tmp_path
) -> None:
    """If an item in the batch isn't a JSON object, return 0.5 for that slot."""
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    _stub_pipeline(monkeypatch, tmp_path, p_yes=0.71)

    client = TestClient(ensemble_agent.app)
    payload = [
        {"event_ticker": "E1", "market_ticker": "M1", "title": "valid"},
        "not-a-dict",  # malformed
        {"event_ticker": "E2", "market_ticker": "M2", "title": "valid2"},
    ]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    assert body[0]["p_yes"] == pytest.approx(0.71)
    assert body[1]["p_yes"] == 0.5  # graceful fallback for the bad item
    assert "JSON object" in body[1]["rationale"]
    assert body[2]["p_yes"] == pytest.approx(0.71)


def test_predict_returns_graceful_fallback_for_top_level_non_object(
    monkeypatch, tmp_path
) -> None:
    """A bare number or string body returns a single 0.5 prediction."""
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    _stub_pipeline(monkeypatch, tmp_path)

    client = TestClient(ensemble_agent.app)
    resp = client.post("/predict", json=42)
    assert resp.status_code == 200
    body = resp.json()
    assert body["p_yes"] == 0.5
    assert "object or array" in body["rationale"]


# ---------------------------------------------------------------------------
# /v1/chat/completions — OpenAI-compatible endpoint for the eval harness
# ---------------------------------------------------------------------------


_REFERENCE_SYSTEM_PROMPT = """You are an AI assistant specialized in analyzing
and predicting real-world events. You have deep expertise in predicting the
outcome of the event: "Indiana vs Memphis"

Note that this event occurs in the future. You will be given a list of sources
with their summaries, rankings, and expert comments. Based on these collected
sources, your goal is to extract meaningful insights and provide well-reasoned
predictions based on the given data.
You will be predicting the probability (as a float value from 0 to 1) of ONLY
the following possible outcomes:
- Memphis
- Indiana

IMPORTANT CONSTRAINTS:
1. You MUST ONLY provide probabilities for the exact possible outcomes listed above
2. Do NOT create or invent any additional outcomes
3. Use exactly the same outcome names as provided (case-sensitive)
4. Ensure all probabilities are between 0 and 1
"""

_REFERENCE_USER_PROMPT = "HERE IS THE GIVEN DATA: Source 1: Pacers beat Grizzlies 124-103 last meeting..."


def _stub_chat_pipeline(monkeypatch, tmp_path, p_yes: float = 0.55) -> None:
    """Shared setup: isolate cache + stub the full ensemble pipeline."""
    from ai_prophet.forecast import ensemble_agent
    from ai_prophet.forecast.ensemble import FinalPrediction

    monkeypatch.setenv("PREDICTION_CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setenv("ENABLE_CACHE", "true")
    monkeypatch.setenv("PREDICTION_DELAY", "0")

    fake_final = FinalPrediction(
        p_yes=p_yes,
        rationale="stub rationale",
        raw_p_yes=p_yes,
        agreement=1.0,
        shrinkage=0.05,
        estimates=[Estimate(p_yes=p_yes, rationale="r", strategy="s", confidence=0.7)],
    )
    monkeypatch.setattr(ensemble_agent, "forecast_event", lambda _e: fake_final)


def test_parse_chat_request_extracts_title_and_markets() -> None:
    """The reference system-prompt format parses correctly."""
    from ai_prophet.forecast.ensemble_agent import _parse_chat_request

    parsed = _parse_chat_request(
        [
            {"role": "system", "content": _REFERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": _REFERENCE_USER_PROMPT},
        ]
    )
    assert parsed["title"] == "Indiana vs Memphis"
    assert parsed["markets"] == ["Memphis", "Indiana"]
    assert "Source 1" in parsed["user"]


def test_parse_chat_request_handles_empty_messages() -> None:
    from ai_prophet.forecast.ensemble_agent import _parse_chat_request

    parsed = _parse_chat_request([])
    assert parsed["title"] == ""
    assert parsed["markets"] == []


def test_chat_completions_v1_binary_event(monkeypatch, tmp_path) -> None:
    """Two-outcome event: run through ensemble, split probabilities."""
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    _stub_chat_pipeline(monkeypatch, tmp_path, p_yes=0.72)

    client = TestClient(ensemble_agent.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [
                {"role": "system", "content": _REFERENCE_SYSTEM_PROMPT},
                {"role": "user", "content": _REFERENCE_USER_PROMPT},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    # OpenAI ChatCompletion shape.
    assert body["object"] == "chat.completion"
    assert body["model"] == "ensemble-agent"
    assert "id" in body
    assert "created" in body
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["role"] == "assistant"

    # The message content is a JSON string with rationale + probabilities.
    content = json.loads(body["choices"][0]["message"]["content"])
    assert "rationale" in content
    assert set(content["probabilities"].keys()) == {"Memphis", "Indiana"}
    # Memphis is outcomes[0] → gets the ensemble's p_yes; Indiana gets 1-p_yes.
    assert content["probabilities"]["Memphis"] == pytest.approx(0.72)
    assert content["probabilities"]["Indiana"] == pytest.approx(0.28)


def test_chat_completions_path_alias_without_v1_prefix(monkeypatch, tmp_path) -> None:
    """``/chat/completions`` is registered as well as ``/v1/chat/completions``."""
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    _stub_chat_pipeline(monkeypatch, tmp_path, p_yes=0.42)

    client = TestClient(ensemble_agent.app)
    resp = client.post(
        "/chat/completions",
        json={
            "model": "x",
            "messages": [
                {"role": "system", "content": _REFERENCE_SYSTEM_PROMPT},
                {"role": "user", "content": ""},
            ],
        },
    )
    assert resp.status_code == 200
    content = json.loads(resp.json()["choices"][0]["message"]["content"])
    assert content["probabilities"]["Memphis"] == pytest.approx(0.42)


def test_chat_completions_multi_outcome_routes_to_dedicated_handler(
    monkeypatch, tmp_path
) -> None:
    """3+ markets go through ``_predict_multi_outcome`` instead of the ensemble."""
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ENABLE_CACHE", "false")

    # The multi-outcome path calls ``call_llm_json`` directly via
    # ``_predict_multi_outcome``. Patch that function instead, so we don't
    # need to drive the LLM mock through the call_llm chain.
    def fake_multi(title, markets, research):
        n = len(markets)
        return {
            "rationale": f"multi-outcome stub for {n} markets",
            "probabilities": dict.fromkeys(markets, 1.0 / n),
        }

    monkeypatch.setattr(ensemble_agent, "_predict_multi_outcome", fake_multi)

    multi_system = """You will predict outcomes for: "RWA on Avalanche"
You will be predicting the probability of ONLY the following possible outcomes:
- Above $450
- Above $500
- Above $600
- Above $700
"""
    client = TestClient(ensemble_agent.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble",
            "messages": [
                {"role": "system", "content": multi_system},
                {"role": "user", "content": "sources..."},
            ],
        },
    )
    assert resp.status_code == 200
    content = json.loads(resp.json()["choices"][0]["message"]["content"])
    assert set(content["probabilities"].keys()) == {
        "Above $450",
        "Above $500",
        "Above $600",
        "Above $700",
    }
    # All four markets get the uniform probability from our stub.
    for p in content["probabilities"].values():
        assert p == pytest.approx(0.25)


def test_chat_completions_unparseable_prompt_returns_safe_payload(
    monkeypatch, tmp_path
) -> None:
    """If we can't parse markets, return a valid (if empty) OpenAI response."""
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    _stub_chat_pipeline(monkeypatch, tmp_path)

    client = TestClient(ensemble_agent.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble",
            "messages": [
                {"role": "system", "content": "Plain text, no event header, no bullets."},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    content = json.loads(body["choices"][0]["message"]["content"])
    # No markets parsed → empty probabilities dict, but the response itself
    # is structurally valid.
    assert content["probabilities"] == {}
    assert "Could not parse" in content["rationale"]


def test_normalize_probabilities_clamps_and_fills() -> None:
    """Missing markets get 1/N; values are clamped to [0.01, 0.99]."""
    from ai_prophet.forecast.ensemble_agent import _normalize_probabilities

    markets = ["A", "B", "C"]
    out = _normalize_probabilities({"A": 1.5, "B": -0.3}, markets)
    assert out["A"] == 0.99  # clamped
    assert out["B"] == 0.01  # clamped
    assert out["C"] == pytest.approx(1.0 / 3)  # default for missing
    assert set(out.keys()) == {"A", "B", "C"}  # no extras


def test_chat_completions_response_has_openai_shape(monkeypatch, tmp_path) -> None:
    """Every required ChatCompletion field is present."""
    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    _stub_chat_pipeline(monkeypatch, tmp_path, p_yes=0.5)

    client = TestClient(ensemble_agent.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [
                {"role": "system", "content": _REFERENCE_SYSTEM_PROMPT},
                {"role": "user", "content": "sources"},
            ],
        },
    )
    body = resp.json()
    for key in ("id", "object", "created", "model", "choices", "usage"):
        assert key in body, f"missing top-level key: {key}"
    assert body["id"].startswith("chatcmpl-")
    choice = body["choices"][0]
    for key in ("index", "message", "finish_reason"):
        assert key in choice
    for key in ("role", "content"):
        assert key in choice["message"]


# ---------------------------------------------------------------------------
# Returns to the existing tests
# ---------------------------------------------------------------------------


def test_predict_batch_logs_one_request_response_pair_per_item(
    caplog, monkeypatch, tmp_path
) -> None:
    """Every item in a batch produces its own request + response log line."""
    import logging

    from ai_prophet.forecast import ensemble_agent
    from fastapi.testclient import TestClient

    _stub_pipeline(monkeypatch, tmp_path, p_yes=0.5)

    client = TestClient(ensemble_agent.app)
    caplog.set_level(logging.INFO, logger="ai_prophet.forecast.ensemble_agent")

    payload = [
        {"event_ticker": "A", "market_ticker": "MA", "title": "alpha"},
        {"event_ticker": "B", "market_ticker": "MB", "title": "beta"},
        {"event_ticker": "C", "market_ticker": "MC", "title": "gamma"},
    ]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    request_lines = [
        r.message
        for r in caplog.records
        if "endpoint.predict request" in r.message
    ]
    response_lines = [
        r.message
        for r in caplog.records
        if "endpoint.predict response" in r.message
    ]
    assert len(request_lines) == 3
    assert len(response_lines) == 3
    # Each event_ticker shows up in exactly one request line.
    for tk in ("A", "B", "C"):
        assert sum(f"event_ticker={tk}" in m for m in request_lines) == 1


def test_forecast_event_threads_temporal_context_to_strategies(
    monkeypatch,
) -> None:
    """forecast_event computes temporal context and passes it to every strategy."""
    from ai_prophet.forecast import ensemble_agent

    monkeypatch.setenv("ENABLE_DELIBERATION", "false")
    monkeypatch.setattr(ensemble_agent, "research_event", lambda **_kw: "")

    captured: list[dict] = []

    def capturing_run(strategy, event, research, temporal_ctx=None):
        captured.append(
            {"strategy": strategy.name, "temporal_ctx": temporal_ctx}
        )
        return Estimate(
            p_yes=0.6,
            rationale="r",
            strategy=strategy.name,
            confidence=0.6,
        )

    monkeypatch.setattr(ensemble_agent, "_run_strategy", capturing_run)

    # Event closing in ~6 hours → imminent bucket.
    from datetime import UTC, datetime, timedelta

    close_iso = (datetime.now(UTC) + timedelta(hours=6)).isoformat()
    event = ensemble_agent.EventRequest(title="t", close_time=close_iso)
    ensemble_agent.forecast_event(event)

    # All three strategies got the same non-None temporal context.
    assert len(captured) == 3
    ctxs = {c["temporal_ctx"] for c in captured}
    assert len(ctxs) == 1  # all received the same string
    ctx = next(iter(ctxs))
    assert ctx is not None
    assert "TIME HORIZON" in ctx
    assert "imminent" in ctx.lower()
