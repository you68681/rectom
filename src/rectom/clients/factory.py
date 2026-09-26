from __future__ import annotations

import os
from pathlib import Path

from rectom.clients.token_logger import TokenUsageLogger
from rectom.clients.trace_logger import LLMTraceLogger


def _build_token_logger() -> TokenUsageLogger | None:
    """Enable token logging only when an output path is explicitly supplied."""
    raw_path = os.environ.get("TOKEN_USAGE_PATH", "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return TokenUsageLogger(str(path))


def _build_trace_logger() -> LLMTraceLogger | None:
    """Enable full prompt/response tracing only when explicitly requested."""
    raw_path = os.environ.get("LLM_TRACE_PATH", "").strip()
    if not raw_path:
        return None
    return LLMTraceLogger(Path(raw_path).expanduser())


def _open_source_sampling(
    provider: str,
    model: str,
) -> tuple[float | None, float | None, int | None]:
    """Return the paper's sampling settings for supported open models."""
    normalized = model.lower()
    if provider == "qwen" or "qwen" in normalized:
        return 1.0, 0.95, 20
    if provider == "gemma" or "gemma" in normalized:
        return 1.0, 0.95, 64
    return None, None, None


def build_client(provider: str, model: str):
    provider = provider.lower()
    token_logger = _build_token_logger()
    trace_logger = _build_trace_logger()

    if provider in {"openai", "vllm", "openai_compatible", "qwen", "gemma"}:
        from rectom.clients.openai_client import OpenAIChatClient

        temperature, top_p, top_k = _open_source_sampling(provider, model)
        return OpenAIChatClient(
            model=model,
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            token_logger=token_logger,
            trace_logger=trace_logger,
            reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT") or None,
            temperature_override=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    if provider == "gemini":
        from rectom.clients.gemini_client import GeminiChatClient

        return GeminiChatClient(
            model=model,
            api_key=os.environ.get("GEMINI_API_KEY"),
            token_logger=token_logger,
            trace_logger=trace_logger,
        )

    raise ValueError(f"Unsupported provider: {provider}")
