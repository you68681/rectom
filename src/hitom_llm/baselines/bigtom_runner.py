# src/hitom_llm/baselines/bigtom_runner.py

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Protocol

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


def load_bigtom_csv(csv_path: str | Path, max_rows: Optional[int] = None) -> List[BigToMSample]:
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


def build_binary_question(question: str, true_answer: str, wrong_answer: str, rng: random.Random):
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


def run_bigtom_baseline(
    csv_path: str | Path,
    output_dir: str | Path,
    llm_client: LLMClient,
    method: str = "simulation",
    sim_llm_client: Optional[LLMClient] = None,
    sample_limit: Optional[int] = None,
    seed: int = 0,
    perspective_gold: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Run BigToM baseline on a semicolon-separated CSV with 4 columns:
    story ; question ; true_answer ; wrong_answer
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    samples = load_bigtom_csv(csv_path, max_rows=sample_limit)

    results: List[BigToMResult] = []
    graded_answers: List[str] = []
    num_errors = 0

    for sample in samples:
        try:
            question_with_choices, shuffled_answers, answer_key, negative_answer_key = build_binary_question(
                question=sample.question,
                true_answer=sample.true_answer,
                wrong_answer=sample.wrong_answer,
                rng=rng,
            )

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
            num_errors += 1
            predicted_answer = f"ERROR: {type(e).__name__}: {e}"
            graded_answer = "Error"
            world_state = "ERROR"
            agent_state = "ERROR"

            question_with_choices, shuffled_answers, answer_key, negative_answer_key = build_binary_question(
                question=sample.question,
                true_answer=sample.true_answer,
                wrong_answer=sample.wrong_answer,
                rng=rng,
            )

            print(
                f"[sample_id={sample.sample_id}] ERROR: "
                f"{type(e).__name__}: {e}"
            )

        if graded_answer == "True":
            graded_answers.append("True")
        elif graded_answer == "False":
            graded_answers.append("False")

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
            predicted_answer_raw=predicted_answer,
            graded_answer=graded_answer,
            world_state=world_state,
            agent_state=agent_state,
        )
        results.append(result)

        if verbose:
            if graded_answers:
                running_acc = graded_answers.count("True") / len(graded_answers)
                print(
                    f"[sample_id={sample.sample_id}] graded={graded_answer} "
                    f"running_acc={running_acc:.2%}"
                )
            else:
                print(f"[sample_id={sample.sample_id}] graded={graded_answer}")

    accuracy = 0.0
    if graded_answers:
        accuracy = graded_answers.count("True") / len(graded_answers)

    results_path = output_dir / f"bigtom_{method}_results.jsonl"
    summary_path = output_dir / f"bigtom_{method}_summary.json"

    save_jsonl(results_path, results)
    summary = {
        "csv_path": str(csv_path),
        "method": method,
        "num_samples": len(results),
        "num_scored": len(graded_answers),
        "num_correct": graded_answers.count("True"),
        "num_errors": num_errors,
        "accuracy": accuracy,
        "results_path": str(results_path),
    }
    save_summary(summary_path, summary)

    print("\n------------------------")
    print("         RESULTS        ")
    print("------------------------")
    print(f"MODEL: {llm_client.model_name}")
    if sim_llm_client is not None:
        print(f"SIM MODEL: {sim_llm_client.model_name}")
    print(f"METHOD: {method}")
    print(f"NUM SAMPLES: {len(results)}")
    print(f"NUM SCORED: {len(graded_answers)}")
    print(f"NUM ERRORS: {num_errors}")
    print(f"ACCURACY: {accuracy:.2%}")
    print("------------------------\n")

    return summary