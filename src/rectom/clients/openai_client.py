import os
import time
from typing import Optional

from openai import OpenAI, APIConnectionError

from rectom.clients.token_logger import TokenUsageLogger, TokenUsageRecord
from rectom.clients.trace_logger import LLMTraceLogger, infer_call_type


class OpenAIChatClient:
    def __init__(
        self,
        model: str,
        timeout: float = 120.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 5,
        retry_wait_seconds: float = 10.0,
        token_logger: Optional[TokenUsageLogger] = None,
        trace_logger: Optional[LLMTraceLogger] = None,
        reasoning_effort: Optional[str] = None,
        temperature_override: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds
        self.token_logger = token_logger
        self.trace_logger = trace_logger
        self.reasoning_effort = reasoning_effort
        self.temperature_override = temperature_override
        self.top_p = top_p
        self.top_k = top_k

        self.client = OpenAI(
            timeout=timeout,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        call_type: str = "unknown",
    ) -> str:
        resolved_call_type = infer_call_type(prompt) if call_type == "unknown" else call_type
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Proprietary models use provider defaults when no override is set,
        # matching the paper. Open-source experiment settings are injected by
        # the factory and take precedence over per-call legacy arguments.
        if self.temperature_override is not None:
            kwargs["temperature"] = self.temperature_override
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["extra_body"] = {"top_k": self.top_k}
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        if self.model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                response_text = resp.choices[0].message.content or ""

                if self.token_logger is not None and getattr(resp, "usage", None) is not None:
                    usage = resp.usage

                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                    total_tokens = getattr(usage, "total_tokens", 0) or 0

                    cached_tokens = 0
                    reasoning_tokens = 0

                    prompt_details = getattr(usage, "prompt_tokens_details", None)
                    if prompt_details is not None:
                        cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0

                    completion_details = getattr(usage, "completion_tokens_details", None)
                    if completion_details is not None:
                        reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0

                    self.token_logger.log(
                        TokenUsageRecord(
                            timestamp=time.time(),
                            provider="openai",
                            model=self.model,
                            call_type=resolved_call_type,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            cached_tokens=cached_tokens,
                            reasoning_tokens=reasoning_tokens,
                            success=True,
                        )
                    )

                if self.trace_logger is not None:
                    self.trace_logger.log(
                        provider="openai",
                        model=self.model,
                        call_type=resolved_call_type,
                        prompt=prompt,
                        attempt=attempt,
                        generation_parameters={
                            key: value
                            for key, value in kwargs.items()
                            if key != "messages"
                        },
                        response=response_text,
                    )

                return response_text

            except APIConnectionError as e:
                last_error = e
                if self.trace_logger is not None:
                    self.trace_logger.log(
                        provider="openai",
                        model=self.model,
                        call_type=resolved_call_type,
                        prompt=prompt,
                        attempt=attempt,
                        generation_parameters={
                            key: value
                            for key, value in kwargs.items()
                            if key != "messages"
                        },
                        error=f"{type(e).__name__}: {e}",
                    )
                if attempt < self.max_retries:
                    print(
                        f"[OpenAIChatClient] APIConnectionError on attempt {attempt}/{self.max_retries}. "
                        f"Retrying in {self.retry_wait_seconds}s..."
                    )
                    time.sleep(self.retry_wait_seconds)
                else:
                    print(
                        f"[OpenAIChatClient] APIConnectionError on final attempt "
                        f"{attempt}/{self.max_retries}."
                    )
                    raise

            except Exception as e:
                if self.trace_logger is not None:
                    self.trace_logger.log(
                        provider="openai",
                        model=self.model,
                        call_type=resolved_call_type,
                        prompt=prompt,
                        attempt=attempt,
                        generation_parameters={
                            key: value
                            for key, value in kwargs.items()
                            if key != "messages"
                        },
                        error=f"{type(e).__name__}: {e}",
                    )
                raise

        raise last_error
