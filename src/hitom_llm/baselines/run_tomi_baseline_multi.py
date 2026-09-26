from __future__ import annotations

import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]
PROJECT_ROOT = CURRENT_FILE.parents[3]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hitom_llm.clients.factory import build_client
from hitom_llm.baselines.tomi_runner_multi import run_tomi_baseline


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
    data_path = PROJECT_ROOT / "Hi-ToM_data.json"

    method = "simulation"
    # provider = "openai"
    # model_name = "gpt-5.4"

    provider = "gemini"
    model_name = "gemini-3-flash-preview"

    sim_provider = None
    sim_model_name = None

    temperature = 0.0
    sample_limit = 100
    category = "all"
    verbose = True

    num_workers = 4

    # ============================================================
    # Mode 1: completely new run
    resume_only_unsolved = False
    input_dir = None
    output_dir = PROJECT_ROOT / "results_hitom_simtom_fixed_gemini_3_token"
    #
    # Mode 2: resume only unsolved samples
    #   resume_only_unsolved = True
    #   input_dir = PROJECT_ROOT / "results_hitom_sitom_test"
    #   output_dir = PROJECT_ROOT / "results_hitom_sitom_resume"
    # ============================================================

    # resume_only_unsolved = True
    #
    # input_dir = PROJECT_ROOT / "results_hitom_simtom_fixed"
    # output_dir = PROJECT_ROOT / "results_hitom_simtom_fixed_gemini_3"

    # If you want a completely new run, use this instead:
    # resume_only_unsolved = False
    # input_dir = None
    # output_dir = PROJECT_ROOT / "results_hitom_sitom_new"

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

    summary = run_tomi_baseline(
        data_path=data_path,
        output_dir=output_dir,
        llm_client=llm_client,
        input_dir=input_dir,
        sim_llm_client=sim_llm_client,
        method=method,
        sample_limit=sample_limit,
        category=category,
        verbose=verbose,
        num_workers=num_workers,
        resume_only_unsolved=resume_only_unsolved,
        provider=provider,
        model_name=model_name,
        sim_provider=sim_provider,
        sim_model_name=sim_model_name,
        temperature=temperature,
    )

    print("Run finished")
    print(summary)


if __name__ == "__main__":
    main()