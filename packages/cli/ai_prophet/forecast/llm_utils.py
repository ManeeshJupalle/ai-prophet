"""Multi-provider LLM client with automatic fallback.

Supports three providers, configured via environment variables:

* ``GROQ_API_KEY``        — Groq (free, fast Llama 3.3 70B)
* ``OPENROUTER_API_KEY``  — OpenRouter (gives access to many premium models)
* ``ANTHROPIC_API_KEY``   — Anthropic direct (Claude Sonnet)

The :func:`call_llm` entry point picks an ordered chain of providers based
on the requested ``tier``:

* ``tier="research"``  — Groq → OpenRouter → Anthropic (cheap, fast queries)
* ``tier="reasoning"`` — OpenRouter → Anthropic → Groq (higher quality)

Failures (missing key, network error, bad status, malformed body) are caught
and the next provider in the chain is tried. If every provider fails the
final exception is re-raised.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Provider configuration -----------------------------------------------------

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT_SECONDS", "45"))

_PROVIDER_CHAINS: dict[str, list[str]] = {
    "research": ["groq", "openrouter", "anthropic"],
    "reasoning": ["openrouter", "anthropic", "groq"],
}


_dotenv_loaded = False


def _ensure_env_loaded() -> None:
    """Load variables from ``.env`` once per process, if dotenv is available."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001 — best effort
        pass
    _dotenv_loaded = True


# Provider implementations ---------------------------------------------------


def _call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    extra_headers: dict[str, str] | None = None,
) -> str:
    """Hit any OpenAI-compatible ``/chat/completions`` endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Empty choices from {base_url}")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Empty content from {base_url}")
    return content


def _call_groq(system: str, user: str, temperature: float, max_tokens: int) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return _call_openai_compatible(
        base_url=GROQ_BASE_URL,
        api_key=api_key,
        model=GROQ_MODEL,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=DEFAULT_TIMEOUT,
    )


def _call_openrouter(system: str, user: str, temperature: float, max_tokens: int) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    extra_headers = {
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://github.com/ai-prophet"),
        "X-Title": os.environ.get("OPENROUTER_TITLE", "ai-prophet ensemble agent"),
    }
    return _call_openai_compatible(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        model=OPENROUTER_MODEL,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=DEFAULT_TIMEOUT,
        extra_headers=extra_headers,
    )


def _call_anthropic(system: str, user: str, temperature: float, max_tokens: int) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover — anthropic is in deps
        raise RuntimeError("anthropic package not installed") from exc

    client = anthropic.Anthropic(api_key=api_key, timeout=DEFAULT_TIMEOUT)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if not response.content:
        raise RuntimeError("Empty content from Anthropic")
    text = response.content[0].text
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Empty text from Anthropic")
    return text


_DISPATCH = {
    "groq": _call_groq,
    "openrouter": _call_openrouter,
    "anthropic": _call_anthropic,
}


# Public API -----------------------------------------------------------------


class LLMError(RuntimeError):
    """Raised when every provider in the chain has failed."""


def call_llm(
    system: str,
    user: str,
    *,
    tier: str = "research",
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> str:
    """Call an LLM, automatically falling back across providers.

    Args:
        system: System prompt.
        user: User prompt.
        tier: ``"research"`` (Groq-first) or ``"reasoning"`` (OpenRouter-first).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        The raw text completion.

    Raises:
        LLMError: If every provider in the chosen chain fails.
    """
    _ensure_env_loaded()

    chain = _PROVIDER_CHAINS.get(tier)
    if chain is None:
        raise ValueError(f"Unknown tier: {tier!r}")

    last_error: Exception | None = None
    for provider in chain:
        fn = _DISPATCH[provider]
        try:
            logger.debug("llm.call provider=%s tier=%s", provider, tier)
            return fn(system, user, temperature, max_tokens)
        except Exception as exc:  # noqa: BLE001
            logger.info("llm.call provider=%s failed: %s", provider, exc)
            last_error = exc
            continue

    raise LLMError(
        f"All providers failed for tier={tier!r}; last error: {last_error}"
    ) from last_error


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _extract_json_object(text: str) -> str:
    """Best-effort: pull the first balanced JSON object out of ``text``."""
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text


def call_llm_json(
    system: str,
    user: str,
    *,
    tier: str = "research",
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> dict[str, Any]:
    """Call an LLM and parse the response as JSON.

    Strips markdown code fences and tolerates extra prose around a single JSON
    object before parsing.
    """
    raw = call_llm(
        system,
        user,
        tier=tier,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    candidate = _strip_fences(raw)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        salvaged = _extract_json_object(candidate)
        return json.loads(salvaged)
