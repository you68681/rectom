from __future__ import annotations

import os

from hitom_llm.clients.openai_client import OpenAIChatClient

# key=os.environ.get("OPENAI_API_KEY")
# def build_client(provider: str, model: str):
#     provider = provider.lower()
#
#     if provider in {"openai", "vllm", "openai_compatible", "qwen"}:
#         # vLLM / Qwen API can both be used through an OpenAI-compatible endpoint.
#         # Configure OPENAI_BASE_URL and OPENAI_API_KEY as needed.
#         # return OpenAIChatClient(
#         #     model=model,
#         #     base_url=os.environ.get("OPENAI_BASE_URL"),
#         #     api_key=os.environ.get("OPENAI_API_KEY"),
#         # )
#         return OpenAIChatClient(
#             model=model,
#             base_url=os.environ.get("OPENAI_BASE_URL"),
#             api_key=key,
#         )
#
#     raise ValueError(f"Unsupported provider: {provider}")


import os

from hitom_llm.clients.openai_client import OpenAIChatClient
from hitom_llm.clients.gemini_client import GeminiChatClient

open_ai_key = os.environ.get("OPENAI_API_KEY")

# open_ai_key = os.environ.get("OPENAI_API_KEY")
gemini_key = os.environ.get("GEMINI_API_KEY")


def build_client(provider: str, model: str):
    provider = provider.lower()

    if provider in {"openai", "vllm", "openai_compatible", "qwen"}:
        return OpenAIChatClient(
            model=model,
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=open_ai_key,
        )

    if provider == "gemini":
        return GeminiChatClient(
            model=model,
            api_key=gemini_key,
        )

    raise ValueError(f"Unsupported provider: {provider}")
