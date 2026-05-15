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
