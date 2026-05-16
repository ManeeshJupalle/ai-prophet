"""Tests for the ensemble forecasting math and pipeline integration."""

from __future__ import annotations

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
        lambda strategy, event, research: table[strategy.name],
    )

    captured_calls: list[list[Estimate]] = []

    def fake_deliberate(event, estimates):
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
        lambda strategy, event, research: table[strategy.name],
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
        lambda strategy, event, research: table[strategy.name],
    )
    monkeypatch.setattr(
        ensemble_agent, "_deliberate", lambda event, estimates: None
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
