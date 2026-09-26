from __future__ import annotations

import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]
PROJECT_ROOT = CURRENT_FILE.parents[3]

print("CURRENT_FILE =", CURRENT_FILE)
print("PROJECT_ROOT =", PROJECT_ROOT)
print("SRC_DIR =", SRC_DIR)
print("SRC_DIR exists =", SRC_DIR.exists())

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("sys.path[0] =", sys.path[0])
print("sys.path[:5] =", sys.path[:5])

from hitom_llm.baselines.bigtom_runner_multi import run_bigtom_baseline
from hitom_llm.clients.factory import build_client


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
    csv_path = "/Users/clei1/Downloads/hitom_llm_project/conditions_original/0_forward_belief_false_belief/stories.csv"

    method = "simulation"
    provider = "openai"
    model_name = "gpt-5.4"

    # provider = "gemini"
    # model_name = "gemini-3-flash-preview"

    sim_provider = None
    sim_model_name = None

    temperature = 0.0
    sample_limit = 50
    seed = 0
    perspective_gold = False
    verbose = True

    num_workers = 8

    # ============================================================
    # Mode 1: completely new run
    resume_only_unsolved = False
    input_dir = None
    output_dir = PROJECT_ROOT / "results_bigtom_false_belief_simtom_fixed_token_5.4"
    #
    # Mode 2: resume only unsolved samples
    #   resume_only_unsolved = True
    #   input_dir = PROJECT_ROOT / "results_bigtom_true_belief_simtom_again_again"
    #   output_dir = PROJECT_ROOT / "results_bigtom_true_belief_simtom_resume"
    # ============================================================

    # resume_only_unsolved = True
    #
    # input_dir = PROJECT_ROOT / "results_bigtom_true_belief_simtom_again_again"
    # output_dir = PROJECT_ROOT / "results_bigtom_true_belief_simtom_resume"

    # If you want a completely new run, use this instead:
    # resume_only_unsolved = False
    # input_dir = None
    # output_dir = PROJECT_ROOT / "results_bigtom_new"

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

    summary = run_bigtom_baseline(
        csv_path=csv_path,
        output_dir=output_dir,
        llm_client=llm_client,
        input_dir=input_dir,
        sim_llm_client=sim_llm_client,
        method=method,
        sample_limit=sample_limit,
        seed=seed,
        perspective_gold=perspective_gold,
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