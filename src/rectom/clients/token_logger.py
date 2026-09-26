# rectom/clients/token_logger.py

from __future__ import annotations

import csv
import os
import time
import threading
from dataclasses import dataclass, asdict
from typing import Optional


_TOKEN_LOG_LOCK = threading.Lock()


@dataclass
class TokenUsageRecord:
    timestamp: float
    provider: str
    model: str
    call_type: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    success: bool = True
    error: str = ""


class TokenUsageLogger:
    def __init__(self, log_path: str = "token_usage.csv"):
        self.log_path = log_path
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cached_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_calls = 0

        with _TOKEN_LOG_LOCK:
            if not os.path.exists(self.log_path):
                with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(TokenUsageRecord.__dataclass_fields__.keys()))
                    writer.writeheader()

    def log(self, record: TokenUsageRecord):
        with _TOKEN_LOG_LOCK:
            self.total_calls += 1
            self.total_prompt_tokens += record.prompt_tokens
            self.total_completion_tokens += record.completion_tokens
            self.total_tokens += record.total_tokens
            self.total_cached_tokens += record.cached_tokens
            self.total_reasoning_tokens += record.reasoning_tokens

            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(TokenUsageRecord.__dataclass_fields__.keys()))
                writer.writerow(asdict(record))

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.total_cached_tokens,
            "reasoning_tokens": self.total_reasoning_tokens,
        }
