from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

from . import timetom_prompts
from .timetom_utils import (
    StorySample,
    answer_letter_to_text,
    build_common_belief_text,
    classify_scenario,
    extract_characters,
    extract_choice_letter,
    infer_question_characters,
    infer_question_order,
    load_story_samples,
    reduce_higher_order_question_to_first_order,
    strip_reasoning_keep_final_answer,
)


class LLMClient(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class TimeToMResult:
    sample_id: int | str
    story: str
    question: str
    label: str
    label_text: str
    prediction_raw: str
    prediction_clean: str
    prediction_letter: Optional[str]
    prediction_text: Optional[str]
    temporal_story: str
    target_character: Optional[str]
    relevant_characters: List[str]
    tbsc_by_character: Dict[str, str]
    compressed_by_character: Dict[str, str]
    belief_solver_feedback: Optional[str]
    belief_solver_answer: Optional[str]
    reduced_question: Optional[str]
    question_order: int
    method: str
    correct: Optional[bool]
    error: Optional[str]


def construct_temporal_space(llm_client: LLMClient, sample: StorySample) -> str:
    scenario = classify_scenario(sample.dataset_name or "")
    if scenario == "dialogue":
        prompt = timetom_prompts.DIALOGUE_TEMPORAL_SPACE_PROMPT.format(
            dialogue=sample.story
        )
    else:
        prompt = timetom_prompts.READING_TEMPORAL_SPACE_PROMPT.format(
            story=sample.story
        )
    return llm_client.generate(prompt).strip()


def construct_tbscs(
    llm_client: LLMClient,
    temporal_story: str,
    characters: List[str],
    scenario: str,
) -> Dict[str, str]:
    tbscs: Dict[str, str] = {}

    for character in characters:
        if scenario == "dialogue":
            prompt = timetom_prompts.DIALOGUE_TBSC_PROMPT.format(
                dialogue=temporal_story,
                character=character,
            )
        else:
            prompt = timetom_prompts.READING_TBSC_PROMPT.format(
                story=temporal_story,
                character=character,
            )

        tbscs[character] = llm_client.generate(prompt).strip()

    return tbscs


def compress_beliefs(
    llm_client: LLMClient,
    tbscs: Dict[str, str],
    scenario: str,
) -> Dict[str, str]:
    compressed: Dict[str, str] = {}

    for character, tbsc in tbscs.items():
        if scenario == "dialogue":
            compressed[character] = tbsc
        else:
            prompt = timetom_prompts.READING_BELIEF_COMPRESSION_PROMPT.format(
                character=character,
                perspective=tbsc,
            )
            compressed[character] = llm_client.generate(prompt).strip()

    return compressed


def truth_answer(
    llm_client: LLMClient,
    sample: StorySample,
    temporal_story: str,
) -> str:
    prompt = f"""\
{temporal_story}

Based on the above information, answer the following question:
{sample.question}

Choices: {sample.choices}

Keep your answer concise, one sentence is enough.
You must choose one of the above choices.
"""
    return llm_client.generate(prompt)


def first_order_answer(
    llm_client: LLMClient,
    sample: StorySample,
    character: str,
    perspective: str,
    scenario: str,
) -> str:
    if scenario == "dialogue":
        prompt = timetom_prompts.DIALOGUE_FIRST_ORDER_QA_PROMPT.format(
            perspective=perspective,
            name=character,
            question=sample.question,
            choices=sample.choices,
        )
    else:
        prompt = timetom_prompts.READING_FIRST_ORDER_QA_PROMPT.format(
            perspective=perspective,
            name=character,
            question=sample.question,
            choices=sample.choices,
        )

    return llm_client.generate(prompt)


def higher_order_initial_answer(
    llm_client: LLMClient,
    sample: StorySample,
    character: str,
    perspective: str,
    scenario: str,
) -> str:
    if scenario == "dialogue":
        prompt = timetom_prompts.DIALOGUE_HIGHER_ORDER_QA_PROMPT.format(
            perspective=perspective,
            name=character,
            question=sample.question,
            choices=sample.choices,
        )
    else:
        prompt = timetom_prompts.READING_HIGHER_ORDER_QA_PROMPT.format(
            perspective=perspective,
            name=character,
            question=sample.question,
            choices=sample.choices,
        )

    return llm_client.generate(prompt)


def belief_solver_feedback_answer(
    llm_client: LLMClient,
    sample: StorySample,
    character: str,
    perspective: str,
    scenario: str,
    tbscs: Dict[str, str],
    initial_answer: str,
    participants: List[str],
) -> Tuple[str, str, str, str]:
    participant_tbscs = [tbscs[name] for name in participants if name in tbscs]
    common_belief = build_common_belief_text(participant_tbscs)

    print(f"common belief: {common_belief}")

    if len(participants) >= 2:
        print(f"full question: {sample.question}")
        reduced_question = reduce_higher_order_question_to_first_order(
            sample.question,
            participants,
        )
        print(f"reduced question: {reduced_question}")

        for one_character in participants:
            if one_character not in reduced_question:
                print(f"removed character: {one_character}")

        question_subject = participants[0]
        question_chain = participants[1:]
    else:
        reduced_question = sample.question
        question_subject = character
        question_chain = [character]

    belief_chain_text = " -> ".join(participants) if participants else character
    remaining_chain_text = (
        " -> ".join(question_chain) if question_chain else question_subject
    )

    if scenario == "dialogue":
        reduced_prompt = timetom_prompts.DIALOGUE_SOLVER_REDUCED_QA_PROMPT.format(
            common_belief=common_belief,
            reduced_question=reduced_question,
            choices=sample.choices,
        )
    else:
        reduced_prompt = timetom_prompts.READING_SOLVER_REDUCED_QA_PROMPT.format(
            common_belief=common_belief,
            reduced_question=reduced_question,
            choices=sample.choices,
        )

    answer2 = llm_client.generate(reduced_prompt).strip()

    if scenario == "dialogue":
        feedback_prompt = timetom_prompts.DIALOGUE_SOLVER_FEEDBACK_PROMPT.format(
            perspective=perspective,
            name=character,
            question=sample.question,
            choices=sample.choices,
            answer1=initial_answer,
            belief_chain_text=belief_chain_text,
            question_subject=question_subject,
            remaining_chain_text=remaining_chain_text,
            common_belief=common_belief,
            reduced_question=reduced_question,
            answer2=answer2,
        )
    else:
        feedback_prompt = timetom_prompts.READING_SOLVER_FEEDBACK_PROMPT.format(
            perspective=perspective,
            name=character,
            question=sample.question,
            choices=sample.choices,
            answer1=initial_answer,
            belief_chain_text=belief_chain_text,
            question_subject=question_subject,
            remaining_chain_text=remaining_chain_text,
            common_belief=common_belief,
            reduced_question=reduced_question,
            answer2=answer2,
        )

    final_answer = llm_client.generate(feedback_prompt)

    return final_answer, common_belief, answer2, reduced_question


def _run_one_sample(
    llm_client: LLMClient,
    sample: StorySample,
    method: str,
) -> TimeToMResult:
    temporal_story = ""
    target_character: Optional[str] = None
    relevant_characters: List[str] = []
    tbscs: Dict[str, str] = {}
    compressed: Dict[str, str] = {}
    belief_solver_feedback: Optional[str] = None
    belief_solver_answer: Optional[str] = None
    reduced_question: Optional[str] = None

    prediction_raw = ""
    prediction_clean = ""
    prediction_letter: Optional[str] = None
    prediction_text: Optional[str] = None
    error: Optional[str] = None
    correct: Optional[bool] = None
    question_order = -1

    try:
        scenario = classify_scenario(sample.dataset_name or "")
        question_order = infer_question_order(sample)
        characters = extract_characters(sample.story)

        temporal_story = construct_temporal_space(llm_client, sample)

        if question_order == 0:
            prediction_raw = truth_answer(
                llm_client=llm_client,
                sample=sample,
                temporal_story=temporal_story,
            )
        else:
            question_characters = infer_question_characters(
                sample.question,
                characters,
            )

            dataset_name_norm = str(sample.dataset_name or "").strip().lower()
            if dataset_name_norm in {"fantom", "fan-tom"}:
                question_characters = question_characters[:2]

            if question_characters:
                target_character = question_characters[0]
            else:
                target_character = characters[0] if characters else None

            if question_characters:
                relevant_characters = question_characters[:]
            elif target_character:
                relevant_characters = [target_character]
            else:
                relevant_characters = []

            if not relevant_characters and characters:
                relevant_characters = [characters[0]]

            tbscs = construct_tbscs(
                llm_client=llm_client,
                temporal_story=temporal_story,
                characters=relevant_characters,
                scenario=scenario,
            )

            compressed = compress_beliefs(
                llm_client=llm_client,
                tbscs=tbscs,
                scenario=scenario,
            )

            if question_order == 1:
                perspective = compressed.get(
                    target_character or "",
                    tbscs.get(target_character or "", ""),
                )

                print(f"order one perspective: {perspective}")

                prediction_raw = first_order_answer(
                    llm_client=llm_client,
                    sample=sample,
                    character=target_character or "Unknown",
                    perspective=perspective,
                    scenario=scenario,
                )
            else:
                full_perspective = tbscs.get(target_character or "", "")

                initial_answer = higher_order_initial_answer(
                    llm_client=llm_client,
                    sample=sample,
                    character=target_character or "Unknown",
                    perspective=full_perspective,
                    scenario=scenario,
                )

                if question_characters:
                    participants = question_characters[:]
                elif target_character:
                    participants = [target_character]
                else:
                    participants = []

                (
                    prediction_raw,
                    belief_solver_feedback,
                    belief_solver_answer,
                    reduced_question,
                ) = belief_solver_feedback_answer(
                    llm_client=llm_client,
                    sample=sample,
                    character=target_character or "Unknown",
                    perspective=full_perspective,
                    scenario=scenario,
                    tbscs=tbscs,
                    initial_answer=initial_answer,
                    participants=participants,
                )

        prediction_clean = strip_reasoning_keep_final_answer(prediction_raw)
        print(f"prediction_clean:{prediction_clean}")

        prediction_letter = extract_choice_letter(prediction_clean, sample.choices)
        print(f"prediction_letter: {prediction_letter}")

        if prediction_letter is not None:
            prediction_text = answer_letter_to_text(prediction_letter, sample.choices)
            correct = prediction_letter.upper() == sample.answer.upper()
        else:
            correct = False

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        prediction_raw = f"ERROR: {error}"
        prediction_clean = prediction_raw
        print(f"[sample_id={sample.sample_id}] ERROR: {error}")

    return TimeToMResult(
        sample_id=sample.sample_id,
        story=sample.story,
        question=sample.question,
        label=sample.answer,
        label_text=answer_letter_to_text(sample.answer, sample.choices),
        prediction_raw=prediction_raw,
        prediction_clean=prediction_clean,
        prediction_letter=prediction_letter,
        prediction_text=prediction_text,
        temporal_story=temporal_story,
        target_character=target_character,
        relevant_characters=relevant_characters,
        tbsc_by_character=tbscs,
        compressed_by_character=compressed,
        belief_solver_feedback=belief_solver_feedback,
        belief_solver_answer=belief_solver_answer,
        reduced_question=reduced_question,
        question_order=question_order if question_order >= 0 else infer_question_order(sample),
        method=method,
        correct=correct,
        error=error,
    )


_WORKER_CLIENT: Optional[LLMClient] = None


class _WorkerClient:
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


def _init_worker(provider: str, model_name: str, temperature: float) -> None:
    global _WORKER_CLIENT

    _WORKER_CLIENT = _WorkerClient(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
    )


def _process_sample_in_worker(
    args: Tuple[int, StorySample, str],
) -> Tuple[int, TimeToMResult]:
    idx, sample, method = args

    if _WORKER_CLIENT is None:
        raise RuntimeError("Worker client is not initialized.")

    return idx, _run_one_sample(_WORKER_CLIENT, sample, method=method)


def _extract_parallel_client_config(llm_client: LLMClient) -> Tuple[str, str, float]:
    provider = getattr(llm_client, "provider", None)
    model_name = getattr(llm_client, "model_name", None)
    temperature = getattr(llm_client, "temperature", 0.0)

    if provider is None or model_name is None:
        raise ValueError(
            "Multiprocessing requires llm_client to expose provider and model_name "
            "(and preferably temperature), like the current ClientAdapter does."
        )

    return str(provider), str(model_name), float(temperature)


def _load_existing_timetom_results(results_path: Path) -> Dict[int | str, TimeToMResult]:
    existing: Dict[int | str, TimeToMResult] = {}

    if not results_path.exists():
        print(f"[resume] input results file not found: {results_path}")
        return existing

    with results_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)

                result = TimeToMResult(
                    sample_id=row.get("sample_id"),
                    story=row.get("story", ""),
                    question=row.get("question", ""),
                    label=row.get("label", ""),
                    label_text=row.get("label_text", ""),
                    prediction_raw=row.get("prediction_raw", ""),
                    prediction_clean=row.get("prediction_clean", ""),
                    prediction_letter=row.get("prediction_letter"),
                    prediction_text=row.get("prediction_text"),
                    temporal_story=row.get("temporal_story", ""),
                    target_character=row.get("target_character"),
                    relevant_characters=row.get("relevant_characters", []),
                    tbsc_by_character=row.get("tbsc_by_character", {}),
                    compressed_by_character=row.get("compressed_by_character", {}),
                    belief_solver_feedback=row.get("belief_solver_feedback"),
                    belief_solver_answer=row.get("belief_solver_answer"),
                    reduced_question=row.get("reduced_question"),
                    question_order=int(row.get("question_order", -1)),
                    method=row.get("method", ""),
                    correct=row.get("correct"),
                    error=row.get("error"),
                )

                existing[result.sample_id] = result

            except Exception as e:
                print(
                    f"[resume] skip invalid result line {line_no}: "
                    f"{type(e).__name__}: {e}"
                )

    return existing


def _should_rerun_timetom_sample(
    sample: StorySample,
    existing_results: Dict[int | str, TimeToMResult],
    resume_only_unsolved: bool,
) -> bool:
    if not resume_only_unsolved:
        return True

    old = existing_results.get(sample.sample_id)

    # Resume mode:
    # If this sample_id already exists in the previous JSONL,
    # treat it as completed, regardless of correct/error.
    if old is not None:
        return False

    # Only run samples that have no previous result.
    return True

# def _should_rerun_timetom_sample(
#     sample: StorySample,
#     existing_results: Dict[int | str, TimeToMResult],
#     resume_only_unsolved: bool,
# ) -> bool:
#     if not resume_only_unsolved:
#         return True
#
#     old = existing_results.get(sample.sample_id)
#
#     if old is None:
#         return True
#
#     if old.correct is True and old.error is None:
#         return False
#
#     return True


def _merge_timetom_results_in_sample_order(
    samples: List[StorySample],
    existing_results: Dict[int | str, TimeToMResult],
    new_results: Dict[int | str, TimeToMResult],
    resume_only_unsolved: bool,
) -> List[TimeToMResult]:
    if resume_only_unsolved:
        merged: Dict[int | str, TimeToMResult] = dict(existing_results)
        merged.update(new_results)
    else:
        merged = dict(new_results)

    final_results: List[TimeToMResult] = []

    for sample in samples:
        result = merged.get(sample.sample_id)
        if result is not None:
            final_results.append(result)

    return final_results


def _append_one_result_jsonl(results_path: Path, result: TimeToMResult) -> None:
    """
    Save one finished sample immediately.

    This is important for long experiments:
    if the process is killed halfway, completed samples are already in JSONL.
    """
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_results_jsonl_atomic(
    results_path: Path,
    results: List[TimeToMResult],
) -> None:
    """
    Rewrite final compact JSONL in sample order.

    During resume, append mode can leave duplicate sample_id lines.
    The loader keeps the last one, but final rewrite makes the file clean.
    """
    tmp_path = results_path.with_suffix(results_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    tmp_path.replace(results_path)


def _build_summary(
    dataset_name: str,
    data_path: str | Path,
    input_results_path: Optional[Path],
    method: str,
    model_name: str,
    results: List[TimeToMResult],
    results_path: Path,
    summary_path: Path,
    process_workers: int,
    mp_start_method: str,
    mp_chunksize: int,
    resume_only_unsolved: bool,
    num_existing_results: int,
    num_rerun: int,
) -> dict:
    correct_num = sum(1 for r in results if r.correct)
    total_num = len(results)
    bad = sum(1 for r in results if r.prediction_letter is None)

    category_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for r in results:
        qkey = str(r.question_order)
        category_counts[qkey]["total"] += 1
        if r.correct:
            category_counts[qkey]["correct"] += 1

    accuracy = 0.0 if total_num == 0 else correct_num / total_num

    category_percents = {
        qtype: counts["correct"] / counts["total"]
        for qtype, counts in category_counts.items()
        if counts["total"] > 0
    }

    return {
        "dataset_name": dataset_name,
        "data_path": str(data_path),
        "input_results_path": str(input_results_path) if input_results_path else None,
        "method": method,
        "model": model_name,
        "num_results": len(results),
        "num_scored": total_num,
        "num_correct": correct_num,
        "bad_responses": bad,
        "accuracy": accuracy,
        "question_order_accuracy": category_percents,
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "process_workers": process_workers,
        "mp_start_method": mp_start_method,
        "mp_chunksize": mp_chunksize,
        "resume_only_unsolved": resume_only_unsolved,
        "num_existing_results": num_existing_results,
        "num_rerun": num_rerun,
    }


def _write_summary_atomic(summary_path: Path, summary: dict) -> None:
    tmp_path = summary_path.with_suffix(summary_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    tmp_path.replace(summary_path)


def run_timetom_baseline(
    dataset_name: str,
    data_path: str | Path,
    output_dir: str | Path,
    llm_client: LLMClient,
    input_dir: Optional[str | Path] = None,
    method: str = "timetom",
    sample_limit: Optional[int] = None,
    verbose: bool = False,
    fantom_use_short_context: bool = True,
    process_workers: int = 1,
    mp_start_method: str = "spawn",
    mp_chunksize: int = 1,
    resume_only_unsolved: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{dataset_name}_{method}_results.jsonl"
    summary_path = output_dir / f"{dataset_name}_{method}_summary.json"

    input_results_path: Optional[Path] = None
    existing_results: Dict[int | str, TimeToMResult] = {}

    if resume_only_unsolved:
        if input_dir is None:
            input_dir = output_dir

        input_dir = Path(input_dir)
        input_results_path = input_dir / f"{dataset_name}_{method}_results.jsonl"
        existing_results = _load_existing_timetom_results(input_results_path)

        # If resuming from another folder, copy existing results into this output file first.
        # If input_dir == output_dir, we keep appending to the same file.
        if input_results_path.resolve() != results_path.resolve():
            _write_results_jsonl_atomic(
                results_path=results_path,
                results=list(existing_results.values()),
            )
    else:
        # Fresh run: remove old files to avoid mixing previous experiments.
        if results_path.exists():
            results_path.unlink()
        if summary_path.exists():
            summary_path.unlink()

    samples = load_story_samples(
        dataset_name=dataset_name,
        data_path=data_path,
        sample_limit=sample_limit,
        fantom_use_short_context=fantom_use_short_context,
    )

    samples_to_run = [
        sample
        for sample in samples
        if _should_rerun_timetom_sample(
            sample=sample,
            existing_results=existing_results,
            resume_only_unsolved=resume_only_unsolved,
        )
    ]

    print("\n------------------------")
    print("   EVALUATING TIMETOM    ")
    print("------------------------")
    print(f"MODEL: {llm_client.model_name}")
    print(f"DATASET: {dataset_name}")
    print(f"DATA: {data_path}")
    print(f"METHOD: {method}")
    print(f"N = {sample_limit if sample_limit is not None else 'all'}")
    print(f"PROCESS_WORKERS: {process_workers}")
    print(f"RESUME ONLY UNSOLVED: {resume_only_unsolved}")
    print(f"INPUT RESULTS: {input_results_path if input_results_path else 'N/A'}")
    print(f"OUTPUT RESULTS: {results_path}")
    print(f"OUTPUT SUMMARY: {summary_path}")
    print(f"EXISTING RESULTS: {len(existing_results)}")
    print(f"SAMPLES TO RUN: {len(samples_to_run)}")
    print("------------------------\n")

    new_results: Dict[int | str, TimeToMResult] = {}

    def save_progress(result: TimeToMResult) -> None:
        new_results[result.sample_id] = result

        # Save this sample immediately.
        _append_one_result_jsonl(results_path, result)

        # Update summary immediately.
        current_results = _merge_timetom_results_in_sample_order(
            samples=samples,
            existing_results=existing_results,
            new_results=new_results,
            resume_only_unsolved=resume_only_unsolved,
        )

        current_summary = _build_summary(
            dataset_name=dataset_name,
            data_path=data_path,
            input_results_path=input_results_path,
            method=method,
            model_name=llm_client.model_name,
            results=current_results,
            results_path=results_path,
            summary_path=summary_path,
            process_workers=process_workers,
            mp_start_method=mp_start_method,
            mp_chunksize=mp_chunksize,
            resume_only_unsolved=resume_only_unsolved,
            num_existing_results=len(existing_results),
            num_rerun=len(samples_to_run),
        )

        _write_summary_atomic(summary_path, current_summary)

    if process_workers <= 1:
        for sample in samples_to_run:
            result = _run_one_sample(llm_client, sample, method=method)
            save_progress(result)

            if verbose and result.correct is not None:
                finished = list(new_results.values())
                scored = len(finished)
                correct = sum(1 for r in finished if r.correct)
                running_acc = 0.0 if scored == 0 else correct / scored
                print(
                    f"[sample_id={sample.sample_id}] "
                    f"correct={result.correct} "
                    f"running_acc={running_acc:.2%} "
                    f"saved={results_path}"
                )

    else:
        provider, model_name, temperature = _extract_parallel_client_config(llm_client)

        indexed_samples = [
            (idx, sample, method)
            for idx, sample in enumerate(samples_to_run)
        ]

        ordered_results: List[Optional[TimeToMResult]] = [None] * len(indexed_samples)
        mp_ctx = get_context(mp_start_method)

        with ProcessPoolExecutor(
            max_workers=process_workers,
            mp_context=mp_ctx,
            initializer=_init_worker,
            initargs=(provider, model_name, temperature),
        ) as executor:
            future_to_idx = {
                executor.submit(_process_sample_in_worker, args): args[0]
                for args in indexed_samples
            }

            # as_completed: whichever sample finishes first will be saved first.
            # This avoids waiting for earlier slow samples.
            for future in as_completed(future_to_idx):
                idx, result = future.result()

                ordered_results[idx] = result
                save_progress(result)

                if verbose and result.correct is not None:
                    finished = [r for r in ordered_results if r is not None]
                    scored = len(finished)
                    correct = sum(1 for r in finished if r and r.correct)
                    running_acc = 0.0 if scored == 0 else correct / scored
                    print(
                        f"[sample_id={result.sample_id}] "
                        f"correct={result.correct} "
                        f"running_acc={running_acc:.2%} "
                        f"saved={results_path}"
                    )

    results = _merge_timetom_results_in_sample_order(
        samples=samples,
        existing_results=existing_results,
        new_results=new_results,
        resume_only_unsolved=resume_only_unsolved,
    )

    # Final compact rewrite in sample order.
    _write_results_jsonl_atomic(results_path, results)

    summary = _build_summary(
        dataset_name=dataset_name,
        data_path=data_path,
        input_results_path=input_results_path,
        method=method,
        model_name=llm_client.model_name,
        results=results,
        results_path=results_path,
        summary_path=summary_path,
        process_workers=process_workers,
        mp_start_method=mp_start_method,
        mp_chunksize=mp_chunksize,
        resume_only_unsolved=resume_only_unsolved,
        num_existing_results=len(existing_results),
        num_rerun=len(samples_to_run),
    )

    _write_summary_atomic(summary_path, summary)

    print(f"Accuracy: {summary['accuracy'] * 100:.3f}%")
    print(f"Bad responses: {summary['bad_responses']}")
    print(f"Rerun samples: {len(samples_to_run)}")

    print("\n------------------------")
    print("         RESULTS        ")
    print("------------------------")
    print(f"MODEL: {llm_client.model_name}")
    print(f"METHOD: {method}")
    print(f"ACCURACY: {summary['accuracy']:.2%}")
    print(f"RESULTS PATH: {results_path}")
    print(f"SUMMARY PATH: {summary_path}")
    print("------------------------\n")

    return summary