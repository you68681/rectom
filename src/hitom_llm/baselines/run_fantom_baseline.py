from __future__ import annotations

import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]
PROJECT_ROOT = CURRENT_FILE.parents[3]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hitom_llm.clients.factory import build_client
from hitom_llm.baselines.fantom_runner import run_fantom_baseline


class ClientAdapter:
    def __init__(self, provider: str, model_name: str, temperature: float = 0.0):
        self.provider = provider
        self._model_name = model_name
        self.temperature = temperature
        self.client = build_client(provider=provider, model=model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        return self.client.generate(
            prompt=prompt,
            max_tokens=2048,
            temperature=self.temperature,
        )


def main() -> None:
    # ===== hard-coded debug config =====
    data_path = PROJECT_ROOT / "fantom.json"
    output_dir = PROJECT_ROOT / "results_fantom_sitom"

    method = "simulation"
    provider = "openai"
    model_name = "gpt-5.4-nano"

    sim_provider = None
    sim_model_name = None

    temperature = 0.0
    sample_limit = 400
    category = "all"
    verbose = True
    # ================================

    llm_client = ClientAdapter(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
    )

    sim_llm_client = None
    if sim_model_name is not None:
        sim_llm_client = ClientAdapter(
            provider=sim_provider or provider,
            model_name=sim_model_name,
            temperature=temperature,
        )

    summary = run_fantom_baseline(
        data_path=data_path,
        output_dir=output_dir,
        llm_client=llm_client,
        sim_llm_client=sim_llm_client,
        method=method,
        sample_limit=sample_limit,
        category=category,
        verbose=verbose,
    )

    print("Run finished")
    print(summary)


if __name__ == "__main__":
    main()