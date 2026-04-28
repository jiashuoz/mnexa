"""LLM client factory.

Model name implies provider:
    claude-*    -> Anthropic
    gemini-*    -> Google Gemini  (v0 default)
    gpt-*, o*   -> OpenAI

Override via MNEXA_PROVIDER env var. Override model via MNEXA_MODEL.
"""

from __future__ import annotations

import os

from mnexa.llm.base import Completion, LLMClient, Usage

DEFAULT_MODEL = "gemini-2.5-pro"


def get_client(model: str | None = None, provider: str | None = None) -> LLMClient:
    model = model or os.environ.get("MNEXA_MODEL") or DEFAULT_MODEL
    provider = provider or os.environ.get("MNEXA_PROVIDER") or _infer_provider(model)

    if provider == "gemini":
        from mnexa.llm.gemini import GeminiClient

        return GeminiClient(model=model)

    raise RuntimeError(
        f"unsupported provider {provider!r} for model {model!r}. "
        f"v0 ships Gemini only; install with extras for other providers."
    )


def _infer_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    raise RuntimeError(
        f"could not infer provider from model {model!r}. "
        f"Set MNEXA_PROVIDER=gemini|anthropic|openai explicitly."
    )


__all__ = ["DEFAULT_MODEL", "Completion", "LLMClient", "Usage", "get_client"]
