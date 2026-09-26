from __future__ import annotations

import os
import time
from typing import Optional

from google import genai
from google.genai import types

from hitom_llm.clients.token_logger import TokenUsageLogger, TokenUsageRecord
from hitom_llm.clients.trace_logger import LLMTraceLogger, infer_call_type


class GeminiChatClient:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_retries: int = 10,
        retry_wait_seconds: float = 10.0,
        timeout_seconds: int = 120,
        token_logger: Optional[TokenUsageLogger] = None,
        trace_logger: Optional[LLMTraceLogger] = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds
        self.token_logger = token_logger
        self.trace_logger = trace_logger

        self.client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                timeout=timeout_seconds * 1000,
            )
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        call_type: str = "unknown",
    ) -> str:
        resolved_call_type = infer_call_type(prompt) if call_type == "unknown" else call_type
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                    ),
                )

                if response.text is None:
                    time.sleep(self.retry_wait_seconds)
                    continue

                if self.token_logger is not None and getattr(response, "usage_metadata", None) is not None:
                    usage = response.usage_metadata

                    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
                    total_tokens = getattr(usage, "total_token_count", 0) or 0

                    # Gemini thinking token 字段在不同 SDK/模型中可能不同，因此这里做兼容处理。
                    reasoning_tokens = (
                        getattr(usage, "thoughts_token_count", 0)
                        or getattr(usage, "thinking_token_count", 0)
                        or 0
                    )

                    self.token_logger.log(
                        TokenUsageRecord(
                            timestamp=time.time(),
                            provider="gemini",
                            model=self.model,
                            call_type=resolved_call_type,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            reasoning_tokens=reasoning_tokens,
                            success=True,
                        )
                    )

                if self.trace_logger is not None:
                    self.trace_logger.log(
                        provider="gemini",
                        model=self.model,
                        call_type=resolved_call_type,
                        prompt=prompt,
                        attempt=attempt,
                        generation_parameters={"max_output_tokens": max_tokens},
                        response=response.text or "",
                    )

                return response.text or ""

            except Exception as e:
                last_error = e
                if self.trace_logger is not None:
                    self.trace_logger.log(
                        provider="gemini",
                        model=self.model,
                        call_type=resolved_call_type,
                        prompt=prompt,
                        attempt=attempt,
                        generation_parameters={"max_output_tokens": max_tokens},
                        error=f"{type(e).__name__}: {e}",
                    )
                if attempt < self.max_retries:
                    print(
                        f"[GeminiChatClient] Error on attempt {attempt}/{self.max_retries}: {e}. "
                        f"Retrying in {self.retry_wait_seconds}s..."
                    )
                    time.sleep(self.retry_wait_seconds)
                else:
                    print(
                        f"[GeminiChatClient] Error on final attempt "
                        f"{attempt}/{self.max_retries}: {e}"
                    )
                    raise

        raise last_error
