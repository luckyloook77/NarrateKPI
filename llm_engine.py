"""
NarrateKPI — LLM Synthesis Engine (Module 2)

Connects to an external LLM API (NVIDIA, OpenAI, DeepSeek, or Google Gemini)
to transform the ``summary_raw_json`` from the Math Engine into a structured
Markdown client report.

Provider priority (first key found wins):
  1. ``NVIDIA_API_KEY``   — OpenAI SDK → https://integrate.api.nvidia.com/v1
  2. ``OPENAI_API_KEY``   — httpx       → https://api.openai.com/v1
  3. ``DEEPSEEK_API_KEY`` — httpx       → https://api.deepseek.com
  4. ``GEMINI_API_KEY``   — httpx       → https://generativelanguage.googleapis.com

When no API key is present, switches to **dry-run mode** and produces a
realistic template-based report so the pipeline can be tested without
burning credits.
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
# Ordered by priority — the first env var that is set wins.
_PROVIDERS: Dict[str, tuple[str, str, str]] = {
    # NVIDIA (uses openai SDK, base_url is for reference / health reports)
    "NVIDIA_API_KEY": ("nvidia", "https://integrate.api.nvidia.com/v1", "meta/llama-3.3-70b-instruct"),
    # Standard OpenAI-compatible providers (via httpx)
    "OPENAI_API_KEY": ("openai", "https://api.openai.com/v1", "gpt-4o-mini"),
    "DEEPSEEK_API_KEY": ("deepseek", "https://api.deepseek.com", "deepseek-chat"),
    # Google Gemini (different API structure)
    "GEMINI_API_KEY": ("gemini", "https://generativelanguage.googleapis.com/v1beta", "gemini-2.0-flash"),
}

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.3

# Allow model override per provider via environment variables.
_MODEL_OVERRIDES: Dict[str, str] = {
    "NVIDIA_API_KEY": "NVIDIA_MODEL_NAME",
}


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
            # Check for model override (e.g. NVIDIA_MODEL_NAME)
            override_var = _MODEL_OVERRIDES.get(env_var)
            if override_var:
                model = os.environ.get(override_var, model)
            return (label, base_url, model, api_key)
    return None


# ──────────────────────────────────────────────────────────────────────
#  NVIDIA API call (via openai SDK)
# ──────────────────────────────────────────────────────────────────────

def _call_nvidia(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Call NVIDIA's inference endpoint using the standard ``openai`` SDK.

    Uses the OpenAI-compatible API at ``https://integrate.api.nvidia.com/v1``
    with the provided NVIDIA API key.

    Falls back to an ``httpx``-based raw call if the ``openai`` package
    is not installed.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print(
            "[NarrateKPI] ⚠️  openai SDK not installed. "
            "Falling back to httpx for NVIDIA API call. "
            "Install it with: pip install openai",
            file=sys.stderr,
        )
        return _call_openai_compatible(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("NVIDIA API returned empty response content")

    return content.strip()


# ──────────────────────────────────────────────────────────────────────
#  OpenAI-compatible API call (via httpx)
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
    """Call an OpenAI-compatible chat completions endpoint via ``httpx``.

    Works for OpenAI, DeepSeek, and any provider that follows the
    ``/chat/completions`` convention.
    """
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


# ──────────────────────────────────────────────────────────────────────
#  Gemini API call (via httpx)
# ──────────────────────────────────────────────────────────────────────

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

    Automatically detects the LLM provider from environment variables
    (NVIDIA → OpenAI → DeepSeek → Gemini).  Falls back to a mock
    template-based report when no API key is set.

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

    try:
        if label == "gemini":
            content = _call_gemini(
                api_key=api_key,
                model=active_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif label == "nvidia":
            content = _call_nvidia(
                api_key=api_key,
                model=active_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            # OpenAI-compatible (OpenAI, DeepSeek, etc.) via httpx
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
    except Exception as e:
        print(
            f"[NarrateKPI] ⚠️  LLM API call failed ({e}). "
            f"Falling back to dry-run mock report to keep pipeline running.",
            file=sys.stderr,
        )
        return generate_mock_report(summary_raw_json)
