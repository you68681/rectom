from __future__ import annotations

import os
import time
from typing import Optional

from google import genai
from google.genai import types

class GeminiChatClient:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_retries: int = 10,
        retry_wait_seconds: float = 10.0,
        timeout_seconds: int = 120,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds

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
    ) -> str:
        last_error = None
        max_tokens += 8192 * 2
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        thinking_config=types.ThinkingConfig(thinking_level="low")
                    ),
                )
                if  response.text==None:
                    time.sleep(self.retry_wait_seconds)
                    continue
                return response.text or ""

            except Exception as e:
                last_error = e
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