from __future__ import annotations

import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]
PROJECT_ROOT = CURRENT_FILE.parents[3]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hitom_llm.clients.factory import build_client
from hitom_llm.baselines.timetom_runner_multiprocess_with_prints_with_unsolved import (
    run_timetom_baseline,
)


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

    # data_path = Path(
    #     "/Users/clei1/Downloads/hitom_llm_project/Hi-ToM_data.json"
    # )

    # data_path = Path(
    #     "/Users/clei1/Downloads/hitom_llm_project/fantom.json"
    # )

    data_path = Path(
        "/Users/clei1/Downloads/hitom_llm_project/conditions/0_forward_belief_true_belief/stories.csv"
    )

    method = "timetom"

    provider = "openai"
    model_name = "gpt-5.4"

    # provider = "gemini"
    # model_name = "gemini-3-flash-preview"

    temperature = 0.0
    sample_limit = 50
    verbose = True
    fantom_use_short_context = True

    process_workers = 4
    mp_start_method = "spawn"
    mp_chunksize = 1

    # ============================================================
    # Mode 1: completely new run
    # This will delete old output files in output_dir and start fresh.
    # ============================================================
    resume_only_unsolved = True
    input_dir = None
    output_dir = PROJECT_ROOT / "results_pycharm_timetom_bigtom_true_belief_token_5.4"

    # ============================================================
    # Mode 2: continue from previous half-finished run in the SAME folder
    # Use this if the previous run was interrupted and you want to continue.
    # It will read:
    #   output_dir / f"{dataset_name}_{method}_results.jsonl"
    # It will skip samples with correct=True and error=None.
    # ============================================================
    # resume_only_unsolved = True
    # input_dir = None
    # output_dir = PROJECT_ROOT / "results_pycharm_timetom_hitom_gemini_3"

    # ============================================================
    # Mode 3: read previous results from one folder, write resumed results
    # to another folder.
    # ============================================================
    # resume_only_unsolved = True
    # input_dir = PROJECT_ROOT / "results_pycharm_timetom_hitom_gemini_3"
    # output_dir = PROJECT_ROOT / "results_pycharm_timetom_hitom_gemini_3_resume"

    llm_client = ClientAdapter(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
    )

    summary = run_timetom_baseline(
        dataset_name=dataset_name,
        data_path=data_path,
        input_dir=input_dir,
        output_dir=output_dir,
        llm_client=llm_client,
        method=method,
        sample_limit=sample_limit,
        verbose=verbose,
        fantom_use_short_context=fantom_use_short_context,
        process_workers=process_workers,
        mp_start_method=mp_start_method,
        mp_chunksize=mp_chunksize,
        resume_only_unsolved=resume_only_unsolved,
    )

    print("Run finished")
    print(summary)


if __name__ == "__main__":
    main()