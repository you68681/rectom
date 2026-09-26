from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from .bigtom_simulation import eval_question, one_big_prompt

GLOBAL_SEED=42
class LLMClient(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class BigToMSample:
    sample_id: int
    story: str
    question: str
    true_answer: str
    wrong_answer: str


@dataclass
class BigToMResult:
    sample_id: int
    story: str
    question_original: str
    question_with_choices: str
    true_answer: str
    wrong_answer: str
    shuffled_answers: List[str]
    answer_key: str
    negative_answer_key: str
    predicted_answer_raw: str
    graded_answer: str
    world_state: str
    agent_state: str


class WorkerClientAdapter:
    def __init__(self, provider: str, model_name: str, temperature: float = 0.0):
        from hitom_llm.clients.factory import build_client

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


def load_bigtom_csv(
    csv_path: str | Path,
    max_rows: Optional[int] = None,
) -> List[BigToMSample]:
    csv_path = Path(csv_path)
    samples: List[BigToMSample] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")

        for idx, row in enumerate(reader):
            if len(row) < 4:
                continue

            story = row[0].strip()
            question = row[1].strip()
            true_answer = row[2].strip()
            wrong_answer = row[3].strip()

            if not story or not question:
                continue

            samples.append(
                BigToMSample(
                    sample_id=idx,
                    story=story,
                    question=question,
                    true_answer=true_answer,
                    wrong_answer=wrong_answer,
                )
            )

            # if max_rows is not None and len(samples) >= max_rows:
            #     break

    if max_rows is not None:
        rng = random.Random(GLOBAL_SEED)
        samples = rng.sample(
            samples,
            k=min(max_rows, len(samples)),
        )

    return samples


def build_binary_question(
    question: str,
    true_answer: str,
    wrong_answer: str,
    rng: random.Random,
):
    answers = [true_answer, wrong_answer]
    rng.shuffle(answers)

    question_with_choices = (
        f"{question}\n"
        f"Choose one of the following:\n"
        f"a){answers[0]}\n"
        f"b){answers[1]}"
    )

    if answers[0] == true_answer:
        answer_key = "a)"
        negative_answer_key = "b)"
    else:
        answer_key = "b)"
        negative_answer_key = "a)"

    return question_with_choices, answers, answer_key, negative_answer_key


def grade_prediction(
    predicted_answer: str,
    answer_key: str,
    negative_answer_key: str,
) -> str:
    predicted_answer_lower = predicted_answer.lower()

    if answer_key in predicted_answer_lower:
        return "True"

    if negative_answer_key in predicted_answer_lower:
        return "False"

    return "False"


def build_direct_prompt(story: str, question_with_choices: str) -> str:
    return f"""Story: {story}
Question: {question_with_choices}"""


def save_jsonl(path: str | Path, rows: List[BigToMResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def save_summary(path: str | Path, summary: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def load_existing_results(results_path: Path) -> Dict[int, BigToMResult]:
    existing: Dict[int, BigToMResult] = {}

    if not results_path.exists():
        print(f"[resume] input results file not found: {results_path}")
        return existing

    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)

                result = BigToMResult(
                    sample_id=int(row["sample_id"]),
                    story=row.get("story", ""),
                    question_original=row.get("question_original", ""),
                    question_with_choices=row.get("question_with_choices", ""),
                    true_answer=row.get("true_answer", ""),
                    wrong_answer=row.get("wrong_answer", ""),
                    shuffled_answers=row.get("shuffled_answers", []),
                    answer_key=row.get("answer_key", ""),
                    negative_answer_key=row.get("negative_answer_key", ""),
                    predicted_answer_raw=row.get("predicted_answer_raw", ""),
                    graded_answer=row.get("graded_answer", "Error"),
                    world_state=row.get("world_state", ""),
                    agent_state=row.get("agent_state", ""),
                )

                existing[result.sample_id] = result

            except Exception as e:
                print(f"[resume] skip invalid line: {type(e).__name__}: {e}")
                continue

    return existing


def should_rerun_sample(
    sample: BigToMSample,
    existing_results: Dict[int, BigToMResult],
    resume_only_unsolved: bool,
) -> bool:
    if not resume_only_unsolved:
        return True

    old = existing_results.get(sample.sample_id)

    if old is None:
        return True

    if old.graded_answer == "True":
        return False

    return True


def run_one_sample(
    sample_dict: dict,
    method: str,
    provider: str,
    model_name: str,
    sim_provider: Optional[str],
    sim_model_name: Optional[str],
    temperature: float,
    seed: int,
    perspective_gold: bool,
) -> BigToMResult:
    sample = BigToMSample(**sample_dict)

    llm_client = WorkerClientAdapter(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
    )

    sim_llm_client = None
    if sim_model_name is not None:
        sim_llm_client = WorkerClientAdapter(
            provider=sim_provider or provider,
            model_name=sim_model_name,
            temperature=temperature,
        )

    # Important:
    # Use sample-specific seed so binary answer order is deterministic
    # regardless of multiprocessing scheduling.
    rng = random.Random(seed + sample.sample_id)

    question_with_choices, shuffled_answers, answer_key, negative_answer_key = build_binary_question(
        question=sample.question,
        true_answer=sample.true_answer,
        wrong_answer=sample.wrong_answer,
        rng=rng,
    )

    try:
        if method == "simulation":
            predicted_answer, world = eval_question(
                llm=llm_client,
                context=sample.story,
                question=question_with_choices,
                knows_change=False,
                perspective_gold=perspective_gold,
                sim_model=sim_llm_client,
            )
            world_state = world.get_world_state()
            agent_state = world.get_agent_state()

        elif method == "onePromptSimulation":
            predicted_answer, full_response = one_big_prompt(
                llm=llm_client,
                story=sample.story,
                question=question_with_choices,
            )
            world_state = full_response
            agent_state = "N/A"

            if len(predicted_answer.strip()) == 0:
                predicted_answer = "Refused to respond"

        elif method == "direct":
            prompt = build_direct_prompt(sample.story, question_with_choices)
            predicted_answer = llm_client.generate(prompt)
            world_state = "N/A"
            agent_state = "N/A"

        else:
            raise ValueError(f"Unsupported method: {method}")

        graded_answer = grade_prediction(
            predicted_answer=predicted_answer,
            answer_key=answer_key,
            negative_answer_key=negative_answer_key,
        )

    except Exception as e:
        predicted_answer = f"ERROR: {type(e).__name__}: {e}"
        graded_answer = "Error"
        world_state = "ERROR"
        agent_state = "ERROR"

    return BigToMResult(
        sample_id=sample.sample_id,
        story=sample.story,
        question_original=sample.question,
        question_with_choices=question_with_choices,
        true_answer=sample.true_answer,
        wrong_answer=sample.wrong_answer,
        shuffled_answers=shuffled_answers,
        answer_key=answer_key,
        negative_answer_key=negative_answer_key,
        predicted_answer_raw=predicted_answer,
        graded_answer=graded_answer,
        world_state=world_state,
        agent_state=agent_state,
    )


def run_bigtom_baseline(
    csv_path: str | Path,
    output_dir: str | Path,
    llm_client: LLMClient,
    input_dir: Optional[str | Path] = None,
    method: str = "simulation",
    sim_llm_client: Optional[LLMClient] = None,
    sample_limit: Optional[int] = None,
    seed: int = 0,
    perspective_gold: bool = False,
    verbose: bool = False,
    num_workers: int = 1,
    resume_only_unsolved: bool = False,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    sim_provider: Optional[str] = None,
    sim_model_name: Optional[str] = None,
    temperature: float = 0.0,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"bigtom_{method}_results.jsonl"
    summary_path = output_dir / f"bigtom_{method}_summary.json"

    input_results_path: Optional[Path] = None
    existing_results: Dict[int, BigToMResult] = {}

    if resume_only_unsolved:
        if input_dir is None:
            input_dir = output_dir

        input_dir = Path(input_dir)
        input_results_path = input_dir / f"bigtom_{method}_results.jsonl"
        existing_results = load_existing_results(input_results_path)

    samples = load_bigtom_csv(csv_path, max_rows=sample_limit)

    samples_to_run = [
        sample
        for sample in samples
        if should_rerun_sample(
            sample=sample,
            existing_results=existing_results,
            resume_only_unsolved=resume_only_unsolved,
        )
    ]

    print("\n------------------------")
    print("    EVALUATING BIGTOM    ")
    print("------------------------")
    print(f"MODEL: {llm_client.model_name}")
    if sim_llm_client is not None:
        print(f"SIM MODEL: {sim_llm_client.model_name}")
    print(f"CSV: {csv_path}")
    print(f"METHOD: {method}")
    print(f"N = {sample_limit if sample_limit is not None else 'all'}")
    print(f"NUM WORKERS = {num_workers}")
    print(f"RESUME ONLY UNSOLVED = {resume_only_unsolved}")
    print(f"INPUT RESULTS: {input_results_path if input_results_path else 'N/A'}")
    print(f"OUTPUT RESULTS: {results_path}")
    print(f"OUTPUT SUMMARY: {summary_path}")
    print(f"EXISTING RESULTS = {len(existing_results)}")
    print(f"SAMPLES TO RUN = {len(samples_to_run)}")
    print("------------------------\n")

    actual_provider = provider or getattr(llm_client, "provider", None)
    actual_model_name = model_name or llm_client.model_name

    if actual_provider is None:
        raise ValueError(
            "provider is required. Pass provider=provider into run_bigtom_baseline()."
        )

    new_results: Dict[int, BigToMResult] = {}

    if num_workers <= 1:
        for idx, sample in enumerate(samples_to_run, start=1):
            result = run_one_sample(
                sample_dict=asdict(sample),
                method=method,
                provider=actual_provider,
                model_name=actual_model_name,
                sim_provider=sim_provider,
                sim_model_name=sim_model_name,
                temperature=temperature,
                seed=seed,
                perspective_gold=perspective_gold,
            )

            new_results[result.sample_id] = result

            if verbose:
                print(
                    f"[{idx}/{len(samples_to_run)}] "
                    f"sample_id={result.sample_id} "
                    f"graded={result.graded_answer}"
                )

    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_sample_id = {}

            for sample in samples_to_run:
                future = executor.submit(
                    run_one_sample,
                    asdict(sample),
                    method,
                    actual_provider,
                    actual_model_name,
                    sim_provider,
                    sim_model_name,
                    temperature,
                    seed,
                    perspective_gold,
                )
                future_to_sample_id[future] = sample.sample_id

            finished = 0
            total = len(future_to_sample_id)

            for future in as_completed(future_to_sample_id):
                sample_id = future_to_sample_id[future]
                finished += 1

                try:
                    result = future.result()
                except Exception as e:
                    sample = next(s for s in samples_to_run if s.sample_id == sample_id)

                    rng = random.Random(seed + sample.sample_id)
                    question_with_choices, shuffled_answers, answer_key, negative_answer_key = build_binary_question(
                        question=sample.question,
                        true_answer=sample.true_answer,
                        wrong_answer=sample.wrong_answer,
                        rng=rng,
                    )

                    result = BigToMResult(
                        sample_id=sample.sample_id,
                        story=sample.story,
                        question_original=sample.question,
                        question_with_choices=question_with_choices,
                        true_answer=sample.true_answer,
                        wrong_answer=sample.wrong_answer,
                        shuffled_answers=shuffled_answers,
                        answer_key=answer_key,
                        negative_answer_key=negative_answer_key,
                        predicted_answer_raw=f"ERROR: {type(e).__name__}: {e}",
                        graded_answer="Error",
                        world_state="ERROR",
                        agent_state="ERROR",
                    )

                new_results[result.sample_id] = result

                if verbose:
                    print(
                        f"[{finished}/{total}] "
                        f"sample_id={result.sample_id} "
                        f"graded={result.graded_answer}"
                    )

    if resume_only_unsolved:
        merged_results: Dict[int, BigToMResult] = dict(existing_results)
        merged_results.update(new_results)
    else:
        merged_results = dict(new_results)

    final_results: List[BigToMResult] = []
    for sample in samples:
        result = merged_results.get(sample.sample_id)
        if result is not None:
            final_results.append(result)

    graded_answers: List[str] = []
    num_errors = 0

    for result in final_results:
        if result.graded_answer == "True":
            graded_answers.append("True")
        elif result.graded_answer == "False":
            graded_answers.append("False")
        else:
            num_errors += 1

    accuracy = 0.0
    if graded_answers:
        accuracy = graded_answers.count("True") / len(graded_answers)

    save_jsonl(results_path, final_results)

    summary = {
        "csv_path": str(csv_path),
        "input_results_path": str(input_results_path) if input_results_path else None,
        "method": method,
        "model": llm_client.model_name,
        "sim_model": sim_llm_client.model_name if sim_llm_client else None,
        "num_samples": len(final_results),
        "num_scored": len(graded_answers),
        "num_correct": graded_answers.count("True"),
        "num_errors": num_errors,
        "accuracy": accuracy,
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "num_workers": num_workers,
        "resume_only_unsolved": resume_only_unsolved,
        "num_existing_results": len(existing_results),
        "num_rerun": len(samples_to_run),
    }

    save_summary(summary_path, summary)

    print("\n------------------------")
    print("         RESULTS        ")
    print("------------------------")
    print(f"MODEL: {llm_client.model_name}")
    if sim_llm_client is not None:
        print(f"SIM MODEL: {sim_llm_client.model_name}")
    print(f"METHOD: {method}")
    print(f"NUM SAMPLES: {len(final_results)}")
    print(f"NUM SCORED: {len(graded_answers)}")
    print(f"NUM ERRORS: {num_errors}")
    print(f"RERUN SAMPLES: {len(samples_to_run)}")
    print(f"ACCURACY: {accuracy:.2%}")
    print(f"RESULTS PATH: {results_path}")
    print(f"SUMMARY PATH: {summary_path}")
    print("------------------------\n")

    return summary