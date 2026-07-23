"""
NarrateKPI — LLM Synthesis Engine (Module 2)

Connects to an external LLM API (OpenAI, DeepSeek, or Google Gemini) to
transform the ``summary_raw_json`` from the Math Engine into a structured
Markdown client report.

Key design decisions:
  • Uses ``httpx`` for HTTP calls (lightweight, no heavy SDK dependency).
  • Reads API key from environment variables: ``OPENAI_API_KEY``,
    ``DEEPSEEK_API_KEY``, or ``GEMINI_API_KEY``.
  • When no API key is present, switches to **dry-run mode** and produces a
    realistic template-based report (``prompts.generate_mock_report``) so
    the pipeline can be tested without burning credits.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

from prompts import SYSTEM_PROMPT, build_user_prompt, generate_mock_report


# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

# Maps environment-variable → (provider label, base_url, model).
_PROVIDERS: Dict[str, tuple[str, str, str]] = {
    "OPENAI_API_KEY": ("openai", "https://api.openai.com/v1", "gpt-4o-mini"),
    "DEEPSEEK_API_KEY": ("deepseek", "https://api.deepseek.com", "deepseek-chat"),
    "GEMINI_API_KEY": ("gemini", "https://generativelanguage.googleapis.com/v1beta", "gemini-2.0-flash"),
}

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.3


# ──────────────────────────────────────────────────────────────────────
#  Provider detection
# ──────────────────────────────────────────────────────────────────────

def detect_provider() -> Optional[tuple[str, str, str, str]]:
    """Check environment variables for a supported API key.

    Returns
    -------
    (provider_label, base_url, model, api_key) or ``None`` if none found.
    """
    for env_var, (label, base_url, model) in _PROVIDERS.items():
        api_key = os.environ.get(env_var)
        if api_key:
            return (label, base_url, model, api_key)
    return None


# ──────────────────────────────────────────────────────────────────────
#  API call functions
# ──────────────────────────────────────────────────────────────────────

def _call_openai_compatible(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint via ``httpx``."""
    try:
        import httpx
    except ImportError:
        print(
            "[NarrateKPI] ⚠️  httpx is not installed. Install it with: pip install httpx",
            file=sys.stderr,
        )
        raise

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    # Parse response — handle both OpenAI and DeepSeek response formats.
    try:
        # OpenAI / DeepSeek format.
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        # Try Gemini wrapper response format.
        content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if not content:
            raise ValueError(
                f"Unexpected API response format: {json.dumps(data, indent=2)[:500]}"
            )

    return content.strip()


def _call_gemini(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Call Google Gemini API (different endpoint structure)."""
    try:
        import httpx
    except ImportError:
        print(
            "[NarrateKPI] ⚠️  httpx is not installed. Install it with: pip install httpx",
            file=sys.stderr,
        )
        raise

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise ValueError(
            f"Unexpected Gemini response format: {json.dumps(data, indent=2)[:500]}"
        )

    return content.strip()


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────

def generate_report(
    summary_raw_json: Dict[str, Any],
    *,
    model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Generate a structured Markdown report from the Math Engine's output.

    Automatically detects the LLM provider from environment variables.
    Falls back to a mock template-based report when no API key is set.

    Parameters
    ----------
    summary_raw_json:
        The ``summary_raw_json`` produced by ``math_engine.run_math_engine()``.
    model:
        Override the default model for the detected provider.
    max_tokens:
        Maximum tokens for the LLM response.
    temperature:
        Sampling temperature (lower = more deterministic).

    Returns
    -------
    str
        Markdown report ready for client delivery.
    """
    provider_info = detect_provider()

    if provider_info is None:
        # ── Dry-run / mock mode ────────────────────────────────────
        print(
            "[NarrateKPI] 🔄 No API key found — running in dry-run mode "
            "(mock report generated locally).",
        )
        return generate_mock_report(summary_raw_json)

    # ── Live LLM mode ──────────────────────────────────────────────
    label, base_url, default_model, api_key = provider_info
    active_model = model or default_model
    system_prompt = SYSTEM_PROMPT
    user_prompt = build_user_prompt(summary_raw_json)

    print(f"[NarrateKPI] 🤖 Calling {label} ({active_model})...")

    if label == "gemini":
        content = _call_gemini(
            api_key=api_key,
            model=active_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        content = _call_openai_compatible(
            api_key=api_key,
            base_url=base_url,
            model=active_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    print(f"[NarrateKPI] ✅ Report generated successfully.")
    return content
