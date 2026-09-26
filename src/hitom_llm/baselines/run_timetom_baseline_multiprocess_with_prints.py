
from __future__ import annotations

import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]
PROJECT_ROOT = CURRENT_FILE.parents[3]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hitom_llm.clients.factory import build_client
from hitom_llm.baselines.timetom_runner_multiprocess_with_prints import run_timetom_baseline


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
    dataset_name = "bigtom"  # hitom | bigtom | fantom
    data_path = PROJECT_ROOT / "/Users/clei1/Downloads/hitom_llm_project/conditions_original/0_forward_belief_false_belief/stories.csv"
    output_dir = PROJECT_ROOT / "results_pycharm_timetom_bigtom_false_belief_again"

    method = "timetom"
    provider = "openai"
    model_name = "gpt-5.4-nano"

    temperature = 0.0
    sample_limit = 400
    verbose = True
    fantom_use_short_context = True

    process_workers = 8
    mp_start_method = "spawn"
    mp_chunksize = 1

    llm_client = ClientAdapter(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
    )

    summary = run_timetom_baseline(
        dataset_name=dataset_name,
        data_path=data_path,
        output_dir=output_dir,
        llm_client=llm_client,
        method=method,
        sample_limit=sample_limit,
        verbose=verbose,
        fantom_use_short_context=fantom_use_short_context,
        process_workers=process_workers,
        mp_start_method=mp_start_method,
        mp_chunksize=mp_chunksize,
    )

    print("Run finished")
    print(summary)


if __name__ == "__main__":
    main()
