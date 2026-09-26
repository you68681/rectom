import os
import time
from typing import Optional

from openai import OpenAI, APIConnectionError


class OpenAIChatClient:
    def __init__(
        self,
        model: str,
        timeout: float = 120.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 5,
        retry_wait_seconds: float = 10.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds
        self.client = OpenAI(
            timeout=timeout,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.0) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        if self.model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""

            except APIConnectionError as e:
                last_error = e
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

        raise last_error
