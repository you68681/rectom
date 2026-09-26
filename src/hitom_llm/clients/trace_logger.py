from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


def infer_call_type(prompt: str) -> str:
    """Give legacy generate() calls a readable stage label in trace files."""
    lowered = prompt.lower()
    patterns = (
        ("initial_participants", "who is already participating"),
        ("question_reduction_repair", "repair an invalid zero-order question"),
        ("question_reduction", "reducing a nested theory-of-mind question"),
        ("delta_repair", "repairing one invalid story-step delta"),
        ("delta_repair", "repair an invalid"),
        ("delta_generation", "extracting step-wise state deltas"),
        ("participation_delta", "participation-state"),
        ("communication_delta", "communication-content"),
        ("action_observation_repair", "repairing an invalid action observation basis sequence"),
        ("action_observation_mask", "observe an aligned action predicate"),
        ("state_observation_repair", "repairing an invalid observation basis sequence"),
        ("state_observation_mask", "present in each aligned source state"),
        ("observation_mask", "observation mask"),
        ("action_application", "apply"),
        ("answer", "predicted_answer"),
    )
    for label, marker in patterns:
        if marker in lowered:
            return label
    return "unknown"


class LLMTraceLogger:
    """Thread-safe JSONL logger for reproducible model-call inspection.

    The caller supplies only request/response content and public generation
    parameters. Credentials and client configuration are deliberately not
    accepted, so API keys cannot be written accidentally.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        *,
        provider: str,
        model: str,
        call_type: str,
        prompt: str,
        attempt: int,
        generation_parameters: Dict[str, Any],
        response: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        resolved_call_type = (
            infer_call_type(prompt) if call_type == "unknown" else call_type
        )
        record = {
            "timestamp_unix": time.time(),
            "provider": provider,
            "model": model,
            "call_type": resolved_call_type,
            "attempt": attempt,
            "generation_parameters": generation_parameters,
            "prompt": prompt,
            "response": response,
            "error": error,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
