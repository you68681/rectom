
from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
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
        prompt = timetom_prompts.DIALOGUE_TEMPORAL_SPACE_PROMPT.format(dialogue=sample.story)
    else:
        prompt = timetom_prompts.READING_TEMPORAL_SPACE_PROMPT.format(story=sample.story)
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
        reduced_question = reduce_higher_order_question_to_first_order(sample.question, participants)
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
    remaining_chain_text = " -> ".join(question_chain) if question_chain else question_subject

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
            question_characters = infer_question_characters(sample.question, characters)

            dataset_name_norm = str(sample.dataset_name or "").strip().lower()
            if dataset_name_norm in {"fantom", "fan-tom"}:
                question_characters = question_characters[:2]

            if question_characters:
                target_character = question_characters[0]
            else:
                target_character = characters[0] if characters else None

            relevant_characters = question_characters[:] if question_characters else (
                [target_character] if target_character else []
            )

            if not relevant_characters and characters:
                relevant_characters = [characters[0]]

            tbscs = construct_tbscs(llm_client, temporal_story, relevant_characters, scenario)
            compressed = compress_beliefs(llm_client, tbscs, scenario)

            if question_order == 1:
                perspective = compressed.get(target_character or "", tbscs.get(target_character or "", ""))
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
                participants = question_characters[:] if question_characters else ([target_character] if target_character else [])

                prediction_raw, belief_solver_feedback, belief_solver_answer, reduced_question = belief_solver_feedback_answer(
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


def _process_sample_in_worker(args: Tuple[int, StorySample, str]) -> Tuple[int, TimeToMResult]:
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


def run_timetom_baseline(
    dataset_name: str,
    data_path: str | Path,
    output_dir: str | Path,
    llm_client: LLMClient,
    method: str = "timetom",
    sample_limit: Optional[int] = None,
    verbose: bool = False,
    fantom_use_short_context: bool = True,
    process_workers: int = 1,
    mp_start_method: str = "spawn",
    mp_chunksize: int = 1,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_story_samples(
        dataset_name=dataset_name,
        data_path=data_path,
        sample_limit=sample_limit,
        fantom_use_short_context=fantom_use_short_context,
    )

    print("\\n------------------------")
    print("   EVALUATING TIMETOM    ")
    print("------------------------")
    print(f"MODEL: {llm_client.model_name}")
    print(f"DATASET: {dataset_name}")
    print(f"DATA: {data_path}")
    print(f"METHOD: {method}")
    print(f"N = {sample_limit if sample_limit is not None else 'all'}")
    print(f"PROCESS_WORKERS: {process_workers}")
    print("------------------------\\n")

    results: List[TimeToMResult] = []

    if process_workers <= 1:
        for sample in samples:
            result = _run_one_sample(llm_client, sample, method=method)
            results.append(result)

            if verbose and result.correct is not None:
                scored = len(results)
                correct = sum(1 for r in results if r.correct)
                running_acc = 0.0 if scored == 0 else correct / scored
                print(f"[sample_id={sample.sample_id}] correct={result.correct} running_acc={running_acc:.2%}")
    else:
        provider, model_name, temperature = _extract_parallel_client_config(llm_client)
        indexed_samples = [(idx, sample, method) for idx, sample in enumerate(samples)]
        ordered_results: List[Optional[TimeToMResult]] = [None] * len(indexed_samples)
        mp_ctx = get_context(mp_start_method)

        with ProcessPoolExecutor(
            max_workers=process_workers,
            mp_context=mp_ctx,
            initializer=_init_worker,
            initargs=(provider, model_name, temperature),
        ) as executor:
            for idx, result in executor.map(_process_sample_in_worker, indexed_samples, chunksize=mp_chunksize):
                ordered_results[idx] = result

                if verbose and result.correct is not None:
                    finished = [r for r in ordered_results if r is not None]
                    scored = len(finished)
                    correct = sum(1 for r in finished if r and r.correct)
                    running_acc = 0.0 if scored == 0 else correct / scored
                    print(f"[sample_id={result.sample_id}] correct={result.correct} running_acc={running_acc:.2%}")

        results = [r for r in ordered_results if r is not None]

    correct_num = sum(1 for r in results if r.correct)
    total_num = len(results)
    bad = sum(1 for r in results if r.prediction_letter is None)

    category_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
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

    results_path = output_dir / f"{dataset_name}_{method}_results.jsonl"
    summary_path = output_dir / f"{dataset_name}_{method}_summary.json"

    with results_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\\n")

    summary = {
        "dataset_name": dataset_name,
        "data_path": str(data_path),
        "method": method,
        "model": llm_client.model_name,
        "num_results": len(results),
        "num_scored": total_num,
        "num_correct": correct_num,
        "bad_responses": bad,
        "accuracy": accuracy,
        "question_order_accuracy": category_percents,
        "results_path": str(results_path),
        "process_workers": process_workers,
        "mp_start_method": mp_start_method,
        "mp_chunksize": mp_chunksize,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Accuracy: {accuracy * 100:.3f}%")
    print(f"Bad responses: {bad}")

    print("\\n------------------------")
    print("         RESULTS        ")
    print("------------------------")
    print(f"MODEL: {llm_client.model_name}")
    print(f"METHOD: {method}")
    print(f"ACCURACY: {accuracy:.2%}")
    print("------------------------\\n")

    return summary
