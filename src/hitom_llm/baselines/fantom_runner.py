# src/hitom_llm/tomi/tomi_runner.py

from __future__ import annotations

import json
import re
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from . import fantom_prompts
from .fantom_simulation import eval_question, one_big_prompt

from hitom_llm.utils import load_json

class LLMClient(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def generate(self, prompt: str) -> str:
        ...



@dataclass
class FanTomResult:
    sample_id: int
    story: str
    question: str
    label: str
    prediction_raw: str
    prediction_clean: str
    perspective: str
    question_order:int
    method: str
    correct: Optional[bool]
    error: Optional[str]

@dataclass
class StorySample:
    sample_id: int
    story: str
    question: str
    choices: str
    answer: str
    prompting_type: Optional[str] = None
    question_order: Optional[int] = None
    deception: Optional[bool] = None
    prediction: Optional[str] = None
    reasoning_summary: Optional[str] = None
    is_correct: Optional[bool] = None


def split_story_into_steps_from_newline(story: str) -> List[str]:
    if not isinstance(story, str):
        return []

    story = story.strip()
    if not story:
        return []

    steps: List[str] = []
    for line in story.splitlines():
        line = line.strip()
        if not line:
            continue
        steps.append(line)

    return steps


def rebuild_story_from_steps_newline(steps: List[str]) -> str:
    return "\n".join(steps)


def build_choices(correct_answer: str, opposite_answer: str) -> tuple[str, str]:
    values = [correct_answer, opposite_answer]
    random.shuffle(values)

    labels = ["A", "B"]
    options = list(zip(labels, values))

    options_sorted = sorted(options, key=lambda x: x[0])

    choices = " ".join(f"{label}. {text}" for label, text in options_sorted)

    correct_label = next(label for label, text in options if text == correct_answer)

    return choices, correct_label


def load_samples_fantom_json(data_path: Path, sample_limit: int | None = None) -> List[StorySample]:
    raw = load_json(data_path)

    if isinstance(raw, dict) and "data" in raw:
        rows = raw["data"]
    elif isinstance(raw, list):
        rows = raw
    else:
        raise ValueError("Unsupported JSON structure for newtom dataset.")

    samples: List[StorySample] = []
    sample_id = 0
    totally_questions=0

    for row in rows:
        story = row.get("short_context", "").strip()
        if not story:
            continue


        parsed_steps = split_story_into_steps_from_newline(story)
        if len(parsed_steps)>=18:
            continue
        rebuilt_story = rebuild_story_from_steps_newline(parsed_steps)

        belief_qas = row.get("beliefQAs", [])

        totally_questions+=1

        for qa in belief_qas:
            question = qa.get("question", "").strip()
            correct_answer = qa.get("correct_answer", "").strip()
            wrong_answer = qa.get("wrong_answer", "").strip()
            question_order=qa.get("tom_type","").strip()

            if not question or not correct_answer or not wrong_answer:
                continue

            choices, correct_label = build_choices(correct_answer, wrong_answer)

            samples.append(
                StorySample(
                    sample_id=sample_id,
                    story=rebuilt_story,
                    question=question,
                    choices=choices,
                    answer=correct_label,
                    prompting_type="fantom",
                    question_order=question_order,
                    deception=None,
                )
            )
            sample_id += 1

            if sample_limit is not None and len(samples) >= sample_limit:
                return samples
    print(f"totally_questions: {totally_questions}")
    return samples



def strip_reasoning_keep_final_answer(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    m = re.search(r"Answer\s*:\s*([A-Z])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.match(r"^\s*([A-Z])\s*[.)]", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"[A-Z]", text, flags=re.IGNORECASE)
    if m:
        return m.group(0).upper()

    return text


def build_prompts(sample: StorySample, sim_model_name: Optional[str]) -> Dict[str, str]:

    choices=sample.choices

    baseline_prompt = fantom_prompts.baselinePrompt.format(
        story=sample.story,
        question=sample.question,
        choices=choices,
    )

    question_prompt = fantom_prompts.questionPrompt.format(
        story=sample.story,
        question=sample.question,
        choices=choices,
    )
    if sim_model_name is not None and "llama" in sim_model_name.lower():
        baseline_prompt = fantom_prompts.llamaprompt.format(
            story=sample.story,
            question=sample.question,
            choices=choices,
        )

    return {
        "baseline": baseline_prompt,
        "question": question_prompt,
    }



def is_correct(label: str, prediction: str) -> bool:
    return label.lower() in prediction.lower()


def run_fantom_baseline(
    data_path: str | Path,
    output_dir: str | Path,
    llm_client: LLMClient,
    method: str = "baseline",
    sim_llm_client: Optional[LLMClient] = None,
    sample_limit: Optional[int] = None,
    category: str = "all",
    verbose: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples_fantom_json(data_path, sample_limit=sample_limit)

    results: List[FanTomResult] = []
    correct_num = 0
    total_num = 0
    bad = 0

    category_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    print("\n------------------------")
    print("    EVALUATING TOMI      ")
    print("------------------------")
    print(f"MODEL: {llm_client.model_name}")
    if sim_llm_client is not None:
        print(f"SIM MODEL: {sim_llm_client.model_name}")
    print(f"DATA: {data_path}")
    print(f"METHOD: {method}")
    print(f"CATEGORY: {category}")
    print(f"N = {sample_limit if sample_limit is not None else 'all'}")
    print("------------------------\n")

    allowed_categories = None
    if category != "all":
        allowed_categories = {c.strip() for c in category.split(",") if c.strip()}

    for sample in samples:
        if allowed_categories is not None and sample.question_type not in allowed_categories:
            continue

        perspective = "N/A"
        prediction_raw = ""
        prediction_clean = ""
        error = None
        correct: Optional[bool] = None

        try:
            prompt_map = build_prompts(sample, sim_llm_client.model_name if sim_llm_client else llm_client.model_name)

            if method == "baseline":
                perspective = "N/A (Baseline)"
                prediction_raw = llm_client.generate(prompt_map["baseline"])

            elif method == "baselineRules":
                perspective = "N/A (Baseline)"
                prediction_raw = llm_client.generate(prompt_map["baselineRules"])

            elif method == "oneshot":
                perspective = "N/A (One Shot)"
                prediction_raw = llm_client.generate(prompt_map["oneshot"])

            elif method == "cot":
                perspective = llm_client.generate(prompt_map["cot"])
                if "Answer:" in perspective:
                    prediction_raw = perspective.split("Answer:", 1)[1].strip()
                else:
                    prediction_raw = perspective[-30:].strip()
                perspective = "Chain of Thought:\n" + perspective

            elif method == "cotRules":
                perspective = llm_client.generate(prompt_map["cotRules"])
                if "Answer:" in perspective:
                    prediction_raw = perspective.split("Answer:", 1)[1].strip()
                else:
                    prediction_raw = perspective[-30:].strip()
                perspective = "Chain of Thought:\n" + perspective

            elif method == "oneshotcot":
                perspective = llm_client.generate(prompt_map["oneshotcot"])
                parts = perspective.strip().split(".")
                prediction_raw = parts[-2].strip() if len(parts) >= 2 else perspective.strip()

            elif method == "onePromptSimulation":
                prediction_raw, perspective = one_big_prompt(
                    llm=llm_client,
                    story=sample.story,
                    question=prompt_map["question"],
                )

            elif method == "simulation":
                prediction_raw, perspective = eval_question(
                    llm=llm_client,
                    context=asdict(sample),
                    question=prompt_map["question"],
                    debug=verbose,
                    sim_model=sim_llm_client,
                )

            else:
                raise ValueError(f"Unsupported evaluation method: {method}")

            prediction_clean = strip_reasoning_keep_final_answer(prediction_raw)

            correct = is_correct(sample.answer, prediction_clean)
            if correct:
                correct_num += 1

            total_num += 1
            category_counts[sample.question_type]["total"] += 1
            if correct:
                category_counts[sample.question_type]["correct"] += 1

            if prediction_clean == "-1":
                bad += 1

        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            prediction_raw = f"ERROR: {error}"
            prediction_clean = prediction_raw
            perspective = "ERROR"
            print(f"[sample_id={sample.sample_id}] ERROR: {error}")

        result = FanTomResult(
            sample_id=sample.sample_id,
            story=sample.story,
            question=sample.question,
            label=sample.answer,
            prediction_raw=prediction_raw,
            prediction_clean=prediction_clean,
            perspective=perspective,
            question_order=sample.question_order,
            method=method,
            correct=correct,
            error=error,
        )
        results.append(result)

        if verbose and correct is not None and total_num > 0:
            running_acc = correct_num / total_num
            print(f"[sample_id={sample.sample_id}] correct={correct} running_acc={running_acc:.2%}")

    accuracy = 0.0 if total_num == 0 else correct_num / total_num

    category_percents = {}
    for qtype, counts in category_counts.items():
        if counts["total"] > 0:
            category_percents[qtype] = counts["correct"] / counts["total"]

    results_path = output_dir / f"tomi_{method}_results.jsonl"
    summary_path = output_dir / f"tomi_{method}_summary.json"

    with results_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    summary = {
        "data_path": str(data_path),
        "method": method,
        "model": llm_client.model_name,
        "sim_model": sim_llm_client.model_name if sim_llm_client else None,
        "num_results": len(results),
        "num_scored": total_num,
        "num_correct": correct_num,
        "bad_responses": bad,
        "accuracy": accuracy,
        "category_accuracy": category_percents,
        "results_path": str(results_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Accuracy: {accuracy * 100:.3f}%")
    print(f"Bad responses: {bad}")

    print("\n------------------------")
    print("         RESULTS        ")
    print("------------------------")
    print(f"MODEL: {llm_client.model_name}")
    print(f"METHOD: {method}")
    print(f"ACCURACY: {accuracy:.2%}")
    print("------------------------\n")

    return summary