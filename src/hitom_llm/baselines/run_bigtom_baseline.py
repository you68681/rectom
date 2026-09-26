import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]   # .../hitom_llm_project/src
PROJECT_ROOT = CURRENT_FILE.parents[3]  # .../hitom_llm_project

print("CURRENT_FILE =", CURRENT_FILE)
print("PROJECT_ROOT =", PROJECT_ROOT)
print("SRC_DIR =", SRC_DIR)
print("SRC_DIR exists =", SRC_DIR.exists())

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("sys.path[0] =", sys.path[0])
print("sys.path[:5] =", sys.path[:5])

from hitom_llm.baselines.bigtom_runner import run_bigtom_baseline
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
    # ===== hard-coded debug config =====
    csv_path ="/Users/clei1/Downloads/hitom_llm_project/conditions_original/0_forward_belief_true_belief/stories.csv"
    output_dir = PROJECT_ROOT / "results_bigtom_true_belief_simtom_again_again"
    method = "simulation"
    provider = "openai"
    model_name = "gpt-5.4-nano"
    sim_provider = None
    sim_model_name = None
    temperature = 0.0
    sample_limit = 1000
    seed = 0
    perspective_gold = False
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

    summary = run_bigtom_baseline(
        csv_path=csv_path,
        output_dir=output_dir,
        llm_client=llm_client,
        sim_llm_client=sim_llm_client,
        method=method,
        sample_limit=sample_limit,
        seed=seed,
        perspective_gold=perspective_gold,
        verbose=verbose,
    )

    print("Run finished")
    print(summary)


if __name__ == "__main__":
    main()