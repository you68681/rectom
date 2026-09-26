from __future__ import annotations
import random
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from . import tomi_prompts
from .tomi_simulation import eval_question, one_big_prompt

from hitom_llm.utils import load_json

GLOBAL_SEED=42

class LLMClient(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class ToMISample:
    sample_id: int
    story: str
    question: str
    answer: str
    containers: List[str]
    story_type: str
    question_type: str


@dataclass
class ToMIResult:
    sample_id: int
    story: str
    question: str
    label: str
    prediction_raw: str
    prediction_clean: str
    perspective: str
    question_order: Optional[int]
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


def read_jsonl(path: str | Path, max_rows: Optional[int] = None) -> List[ToMISample]:
    path = Path(path)
    samples: List[ToMISample] = []

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            fields = json.loads(line)

            samples.append(
                ToMISample(
                    sample_id=idx,
                    story=fields["story"],
                    question=fields["question"],
                    answer=fields["answer"],
                    containers=fields["containers"],
                    story_type=fields["story_type"],
                    question_type=fields["question_type"],
                )
            )

            if max_rows is not None and len(samples) >= max_rows:
                break

    return samples


def load_samples_hitom(data_path: Path, sample_limit: int | None = None) -> List[StorySample]:
    raw = load_json(data_path)
    rows = raw["data"]

    samples: List[StorySample] = []

    for row in rows:
        if row.get("story_length") != 1:
            continue
        samples.append(
            StorySample(
                sample_id=int(row["sample_id"]),
                story=row["story"],
                question=row["question"],
                choices=row["choices"],
                answer=row["answer"],
                prompting_type=row.get("prompting_type"),
                question_order=row.get("question_order"),
                deception=row.get("deception"),
            )
        )

        # if sample_limit is not None and len(samples) >= sample_limit:
        #     break

    if sample_limit is not None:
        rng = random.Random(GLOBAL_SEED)
        samples = rng.sample(
            samples,
            k=min(sample_limit, len(samples)),
        )

    return samples


def strip_reasoning_keep_final_answer(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    m = re.search(r"(Answer\s*:\s*[^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"\b([A-O])\s*[\.\)]", text)
    if m:
        return m.group(1).strip()

    m = re.search(r"\bchoice\s+([A-O])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"\banswer\s+is\s+([A-O])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"\b([A-O])\b", text)
    if m:
        return m.group(1).strip()

    return text.strip()


def build_prompts(sample: StorySample, sim_model_name: Optional[str]) -> Dict[str, str]:
    choices = sample.choices

    baseline_prompt = tomi_prompts.baselinePrompt.format(
        story=sample.story,
        question=sample.question,
        choices=choices,
    )

    question_prompt = tomi_prompts.questionPrompt.format(
        story=sample.story,
        question=sample.question,
        choices=choices,
    )

    if sim_model_name is not None and "llama" in sim_model_name.lower():
        baseline_prompt = tomi_prompts.llamaprompt.format(
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


def load_existing_results(results_path: Path) -> Dict[int, ToMIResult]:
    existing: Dict[int, ToMIResult] = {}

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
                result = ToMIResult(
                    sample_id=int(row["sample_id"]),
                    story=row.get("story", ""),
                    question=row.get("question", ""),
                    label=row.get("label", ""),
                    prediction_raw=row.get("prediction_raw", ""),
                    prediction_clean=row.get("prediction_clean", ""),
                    perspective=row.get("perspective", ""),
                    question_order=row.get("question_order"),
                    method=row.get("method", ""),
                    correct=row.get("correct"),
                    error=row.get("error"),
                )
                existing[result.sample_id] = result
            except Exception as e:
                print(f"[resume] skip invalid line: {type(e).__name__}: {e}")
                continue

    return existing


def should_rerun_sample(
    sample: StorySample,
    existing_results: Dict[int, ToMIResult],
    resume_only_unsolved: bool,
) -> bool:
    if not resume_only_unsolved:
        return True

    old = existing_results.get(sample.sample_id)

    if old is None:
        return True

    if old.correct is True:
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
    verbose: bool,
) -> ToMIResult:
    sample = StorySample(**sample_dict)

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

    perspective = "N/A"
    prediction_raw = ""
    prediction_clean = ""
    error = None
    correct: Optional[bool] = None

    try:
        prompt_map = build_prompts(
            sample,
            sim_llm_client.model_name if sim_llm_client else llm_client.model_name,
        )

        if method == "baseline":
            perspective = "N/A (Baseline)"
            prediction_raw = llm_client.generate(prompt_map["baseline"])

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
            raise ValueError(f"Unsupported multiprocessing method: {method}")

        prediction_clean = strip_reasoning_keep_final_answer(prediction_raw)
        print(f"prediction_clean:{prediction_clean}")
        correct = is_correct(sample.answer, prediction_clean)

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        prediction_raw = f"ERROR: {error}"
        prediction_clean = prediction_raw
        perspective = "ERROR"

    return ToMIResult(
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


def run_tomi_baseline(
    data_path: str | Path,
    output_dir: str | Path,
    llm_client: LLMClient,
    input_dir: Optional[str | Path] = None,
    method: str = "baseline",
    sim_llm_client: Optional[LLMClient] = None,
    sample_limit: Optional[int] = None,
    category: str = "all",
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

    results_path = output_dir / f"tomi_{method}_results.jsonl"
    summary_path = output_dir / f"tomi_{method}_summary.json"

    input_results_path: Optional[Path] = None
    existing_results: Dict[int, ToMIResult] = {}

    if resume_only_unsolved:
        if input_dir is None:
            input_dir = output_dir

        input_dir = Path(input_dir)
        input_results_path = input_dir / f"tomi_{method}_results.jsonl"
        existing_results = load_existing_results(input_results_path)

    samples = load_samples_hitom(Path(data_path), sample_limit=sample_limit)

    allowed_categories = None
    if category != "all":
        allowed_categories = {c.strip() for c in category.split(",") if c.strip()}

    filtered_samples: List[StorySample] = []
    for sample in samples:
        if allowed_categories is not None:
            if sample.prompting_type not in allowed_categories:
                continue
        filtered_samples.append(sample)

    samples_to_run = [
        sample
        for sample in filtered_samples
        if should_rerun_sample(
            sample=sample,
            existing_results=existing_results,
            resume_only_unsolved=resume_only_unsolved,
        )
    ]

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
    print(f"NUM WORKERS = {num_workers}")
    print(f"RESUME ONLY UNSOLVED = {resume_only_unsolved}")
    print(f"INPUT RESULTS: {input_results_path if input_results_path else 'N/A'}")
    print(f"OUTPUT RESULTS: {results_path}")
    print(f"OUTPUT SUMMARY: {summary_path}")
    print(f"EXISTING RESULTS = {len(existing_results)}")
    print(f"SAMPLES TO RUN = {len(samples_to_run)}")
    print("------------------------\n")

    new_results: Dict[int, ToMIResult] = {}

    actual_provider = provider or getattr(llm_client, "provider", None)
    actual_model_name = model_name or llm_client.model_name

    if actual_provider is None:
        raise ValueError(
            "provider is required. Pass provider=provider into run_tomi_baseline()."
        )

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
                verbose=verbose,
            )

            new_results[result.sample_id] = result

            if verbose:
                print(
                    f"[{idx}/{len(samples_to_run)}] "
                    f"sample_id={result.sample_id} "
                    f"correct={result.correct} "
                    f"error={result.error}"
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
                    verbose,
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
                    error = f"{type(e).__name__}: {e}"
                    sample = next(s for s in samples_to_run if s.sample_id == sample_id)

                    result = ToMIResult(
                        sample_id=sample.sample_id,
                        story=sample.story,
                        question=sample.question,
                        label=sample.answer,
                        prediction_raw=f"ERROR: {error}",
                        prediction_clean=f"ERROR: {error}",
                        perspective="ERROR",
                        question_order=sample.question_order,
                        method=method,
                        correct=None,
                        error=error,
                    )

                new_results[result.sample_id] = result

                if verbose:
                    print(
                        f"[{finished}/{total}] "
                        f"sample_id={result.sample_id} "
                        f"correct={result.correct} "
                        f"error={result.error}"
                    )

    if resume_only_unsolved:
        merged_results: Dict[int, ToMIResult] = dict(existing_results)
        merged_results.update(new_results)
    else:
        merged_results = dict(new_results)

    final_results: List[ToMIResult] = []
    for sample in filtered_samples:
        result = merged_results.get(sample.sample_id)
        if result is not None:
            final_results.append(result)

    correct_num = 0
    total_num = 0
    bad = 0

    category_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    sample_by_id = {sample.sample_id: sample for sample in filtered_samples}

    for result in final_results:
        sample = sample_by_id.get(result.sample_id)

        if result.correct is not None:
            total_num += 1

            if result.correct:
                correct_num += 1

            if sample is not None:
                qtype = str(sample.question_order)
                category_counts[qtype]["total"] += 1
                if result.correct:
                    category_counts[qtype]["correct"] += 1

        if result.prediction_clean == "-1":
            bad += 1

    accuracy = 0.0 if total_num == 0 else correct_num / total_num

    category_percents = {}
    for qtype, counts in category_counts.items():
        if counts["total"] > 0:
            category_percents[qtype] = counts["correct"] / counts["total"]

    with results_path.open("w", encoding="utf-8") as f:
        for row in sorted(final_results, key=lambda x: x.sample_id):
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    summary = {
        "data_path": str(data_path),
        "input_results_path": str(input_results_path) if input_results_path else None,
        "method": method,
        "model": llm_client.model_name,
        "sim_model": sim_llm_client.model_name if sim_llm_client else None,
        "num_results": len(final_results),
        "num_scored": total_num,
        "num_correct": correct_num,
        "bad_responses": bad,
        "accuracy": accuracy,
        "category_accuracy": category_percents,
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "num_workers": num_workers,
        "resume_only_unsolved": resume_only_unsolved,
        "num_existing_results": len(existing_results),
        "num_rerun": len(samples_to_run),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Accuracy: {accuracy * 100:.3f}%")
    print(f"Bad responses: {bad}")
    print(f"Rerun samples: {len(samples_to_run)}")

    print("\n------------------------")
    print("         RESULTS        ")
    print("------------------------")
    print(f"MODEL: {llm_client.model_name}")
    print(f"METHOD: {method}")
    print(f"ACCURACY: {accuracy:.2%}")
    print(f"RESULTS PATH: {results_path}")
    print(f"SUMMARY PATH: {summary_path}")
    print("------------------------\n")

    return summary