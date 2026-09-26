from __future__ import annotations

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
import random
from hitom_llm.clients.factory import build_client
from hitom_llm.models import StorySample
from hitom_llm.runtime.engine import SequentialHiToMEngine,SequentialBigToMEngine,SequentialFanToMEngine
from hitom_llm.utils import ensure_dir, load_json, split_story_into_steps, write_json, write_jsonl
from hitom_llm.runtime.baseline_engine import (
    SequentialHiToMCoTEngine,
    SequentialBigToMCoTEngine,
    SequentialFanToMCoTEngine,
)

def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

import re
from pathlib import Path
from typing import List


def should_drop_step(step_text: str) -> bool:
    """
    Drop any step whose content is an 'entered the waiting_room' event.
    This works no matter where that step appears in the story.
    """
    step = step_text.strip()
    return bool(re.match(r"^.* entered the waiting_room\.?$", step))


def rebuild_story_from_steps(steps: List[str]) -> str:
    """
    Rebuild the story string with fresh sequential numbering:
    1 ...
    2 ...
    3 ...
    """
    return " ".join(f"{i}. {step}\n" for i, step in enumerate(steps, start=1))

def strip_story_preamble(story: str) -> str:
    prefixes = [
        "Read the following story and answer the multiple-choice question. Please provide answer without explanations.\n",
        "Read the following story and answer the multiple-choice question. Think step-by-step. Provide the answer first, and then explain it.\nStory:\n",
        "Story:\n",
    ]
    for prefix in prefixes:
        if story.startswith(prefix):
            return story[len(prefix):]
    return story

def load_samples_hitom(data_path: Path, sample_limit: int | None = None) -> List[StorySample]:
    raw = load_json(data_path)
    rows = raw["data"]

    samples: List[StorySample] = []

    for row in rows:
        if row.get("story_length") != 1:
            continue

        cleaned_story = strip_story_preamble(row["story"])
        original_steps = split_story_into_steps(cleaned_story)
        filtered_steps = [step for step in original_steps if not should_drop_step(step)]
        filtered_story = rebuild_story_from_steps(filtered_steps)

        samples.append(
            StorySample(
                sample_id=int(row["sample_id"]),
                story=filtered_story,
                question=row["question"],
                choices=row["choices"],
                answer=row["answer"],
                parsed_steps=filtered_steps,
                prompting_type=row.get("prompting_type"),
                question_order=row.get("question_order"),
                deception=row.get("deception"),
            )
        )

        if sample_limit is not None and len(samples) >= sample_limit:
            break

    return samples

import pandas as pd
import re
from typing import List

def split_story_into_sentences_from_period(story: str) -> List[str]:
    if not isinstance(story, str):
        return []

    story = re.sub(r"\s+", " ", story).strip()
    if not story:
        return []

    parts = re.split(r"(?<=[.!?])\s+", story)

    steps: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = re.sub(r"[.!?]+$", "", part).strip()
        if part:
            steps.append(part)

    return steps


def rebuild_story_from_steps(steps: List[str]) -> str:
    return " ".join(f"{i}. {step}\n" for i, step in enumerate(steps, start=1))



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



MENTAL_PREFIX_PATTERNS = [
    r"believes\s+that\s+",
    r"believes\s+",
    r"knows\s+about\s+",
    r"knows\s+that\s+",
    r"knows\s+",
    r"thinks\s+that\s+",
    r"thinks\s+",
    r"expects\s+that\s+",
    r"expects\s+",
    r"understands\s+that\s+",
    r"understands\s+",
    r"assumes\s+that\s+",
    r"assumes\s+",
    r"realizes\s+that\s+",
    r"realizes\s+",
    r"suspects\s+that\s+",
    r"suspects\s+",
]

def strip_character_and_mental_verb(text: str, character: str | None = None) -> str:
    text = text.strip()

    if character:
        for pat in MENTAL_PREFIX_PATTERNS:
            full_pat = rf"^{re.escape(character)}\s+{pat}"
            new_text = re.sub(full_pat, "", text, count=1, flags=re.IGNORECASE).strip()
            if new_text != text:
                return lowercase_first(new_text)

    combined_pat = (
        r"^[A-Z][a-zA-Z'’-]*\s+"
        r"(?:believes(?:\s+that)?|knows(?:\s+about|\s+that)?|thinks(?:\s+that)?|"
        r"expects(?:\s+that)?|understands(?:\s+that)?|assumes(?:\s+that)?|"
        r"realizes(?:\s+that)?|suspects(?:\s+that)?)\s+"
    )
    new_text = re.sub(combined_pat, "", text, count=1).strip()
    return lowercase_first(new_text)

def lowercase_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]

def load_samples_bigtom(data_path: Path, sample_limit: int | None = None) -> List[StorySample]:
    df = pd.read_csv(
        data_path,
        sep=";",
        header=None,
        names=["story", "question", "correct_answer", "opposite_answer"],
        dtype=str,
        keep_default_na=False,
    )

    if sample_limit is not None:
        df = df.iloc[:sample_limit]

    samples: List[StorySample] = []

    for idx, row in df.iterrows():
        story = row["story"].strip()
        question = row["question"].strip()
        correct_answer = row["correct_answer"].strip()
        opposite_answer = row["opposite_answer"].strip()
        correct_answer=strip_character_and_mental_verb(correct_answer)
        opposite_answer = strip_character_and_mental_verb(opposite_answer)

        parsed_steps = split_story_into_sentences_from_period(story)
        rebuilt_story = rebuild_story_from_steps(parsed_steps)
        choices,correct_label = build_choices(correct_answer, opposite_answer)

        # print(f"sample_id: {int(idx)}; question: {question}")
        # print(f"sample_id: {int(idx)}; choices: {choices}")
        samples.append(
            StorySample(
                sample_id=int(idx),
                story=rebuilt_story,
                question=question,
                choices=choices,
                answer=correct_label,
                parsed_steps=parsed_steps,
                prompting_type="bigtom",
                question_order=None,
                deception=None,
            )
        )

    return samples


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
                    parsed_steps=parsed_steps,
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


def load_samples(
    data_path: Path,
    dataset_type: str = "hitom",
    sample_limit: int | None = None,
) -> List[StorySample]:
    if  "hitom" in dataset_type:
        return load_samples_hitom(data_path, sample_limit)
    elif "bigtom" in dataset_type:
        return load_samples_bigtom(data_path, sample_limit)
    elif "fantom" in dataset_type:
        return load_samples_fantom_json(data_path, sample_limit)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")





def load_unresolved_sample_ids(
    data_path: Path,
    previous_predictions_path: Path,
    dataset_type: str = "hitom",
    sample_limit: int | None = None,
) -> List[int]:
    """
    Return sample_ids that should be retried:
      1) samples missing from previous predictions.jsonl
      2) samples explicitly marked failed in previous records (if such fields exist)
    """
    samples = load_samples(data_path, dataset_type=dataset_type, sample_limit=sample_limit)
    all_sample_ids = [s.sample_id for s in samples]

    existing_records = read_jsonl(previous_predictions_path)
    processed_ids: Set[int] = set()
    failed_ids: Set[int] = set()

    for record in existing_records:
        sid = record.get("sample_id")
        if sid is None:
            continue
        sid = int(sid)
        processed_ids.add(sid)

        # Optional: if your record has explicit failure markers, include them
        if (
            record.get("status") == "failed"
            or record.get("is_correct") is False
            or record.get("error") not in (None, "", [])
        ):
            failed_ids.add(sid)

    missing_ids = {sid for sid in all_sample_ids if sid not in processed_ids}
    # unresolved_ids = sorted(missing_ids | failed_ids)
    unresolved_ids = failed_ids
    return unresolved_ids

def run_retry_unresolved(
    data_path,
    previous_output_dir,
    new_output_dir,
    provider: str,
    model: str,
    dataset_type: str = "hitom",
    sample_limit: int | None = None,
    debug: bool = False,
) -> Dict:
    data_path = Path(data_path)
    previous_output_dir = Path(previous_output_dir)
    new_output_dir = ensure_dir(new_output_dir)

    setup_logging(debug)
    client = build_client(provider=provider, model=model)
    if dataset_type=="hitom":
        engine = SequentialHiToMEngine(llm_client=client, debug=debug)
    if dataset_type=="bigtom":
        engine = SequentialBigToMEngine(llm_client=client, debug=debug)
    if dataset_type=="fantom":
        engine = SequentialFanToMEngine(llm_client=client, debug=debug)
    if dataset_type == "hitom_CoT":
        engine = SequentialHiToMCoTEngine(llm_client=client, debug=debug)
    if dataset_type == "bigtom_CoT":
        engine = SequentialBigToMCoTEngine(llm_client=client, debug=debug)
    elif dataset_type == "fantom_CoT":
        engine = SequentialFanToMCoTEngine(llm_client=client, debug=debug)

    previous_predictions_path = previous_output_dir / "predictions.jsonl"
    retry_sample_ids = set(
        load_unresolved_sample_ids(
            data_path=data_path,
            previous_predictions_path=previous_predictions_path,
            dataset_type=dataset_type,
            sample_limit=sample_limit,
        )
    )

    all_samples = load_samples(data_path, dataset_type=dataset_type, sample_limit=sample_limit)
    retry_samples = [s for s in all_samples if s.sample_id in retry_sample_ids]

    logger.info(
        "Found %s unresolved samples to retry from %s",
        len(retry_samples),
        previous_predictions_path,
    )

    predictions_path = new_output_dir / "predictions.jsonl"
    summary_path = new_output_dir / "summary.json"

    if predictions_path.exists():
        predictions_path.unlink()

    correct = 0
    processed_count = 0

    for idx, sample in enumerate(retry_samples, start=1):
        logger.info(
            "Retrying unresolved sample %s/%s (sample_id=%s)",
            idx,
            len(retry_samples),
            sample.sample_id,
        )

        try:
            sample = engine.process_sample(sample)
            record = sample.to_record()
        except Exception as e:
            logger.exception("Retry failed for sample_id=%s", sample.sample_id)
            record = {
                "sample_id": sample.sample_id,
                "is_correct": False,
                "status": "failed",
                "error": str(e),
            }

        append_jsonl(predictions_path, record)
        processed_count += 1

        if record.get("is_correct") is True:
            correct += 1

        summary = {
            "num_samples": len(retry_samples),
            "num_processed": processed_count,
            "num_remaining": len(retry_samples) - processed_count,
            "num_correct": correct,
            "accuracy": (correct / processed_count) if processed_count else 0.0,
            "provider": provider,
            "model": model,
            "status": "running" if processed_count < len(retry_samples) else "finished",
            "source_previous_output_dir": str(previous_output_dir),
        }
        write_json(summary_path, summary)

    final_summary = {
        "num_samples": len(retry_samples),
        "num_processed": processed_count,
        "num_remaining": len(retry_samples) - processed_count,
        "num_correct": correct,
        "accuracy": (correct / processed_count) if processed_count else 0.0,
        "provider": provider,
        "model": model,
        "status": "finished",
        "source_previous_output_dir": str(previous_output_dir),
    }
    write_json(summary_path, final_summary)

    return final_summary


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error: {e}")
                print("Problematic line preview:")
                print(line[:1000])
                raise
    return records


def load_processed_state(predictions_path: Path) -> tuple[Set[int], int, List[dict]]:
    """
    Read existing predictions.jsonl and recover:
      - processed sample_id set
      - num_correct
      - existing records
    """
    existing_records = read_jsonl(predictions_path)
    processed_ids: Set[int] = set()
    num_correct = 0

    for record in existing_records:
        sample_id = record.get("sample_id")
        if sample_id is not None:
            processed_ids.add(int(sample_id))

        if record.get("is_correct") is True:
            num_correct += 1

    return processed_ids, num_correct, existing_records


def run_pipeline(
    data_path,
    output_dir,
    provider: str,
    model: str,
    dataset_type: str = "hitom",
    sample_limit: int | None = None,
    debug: bool = False,
    resume: bool = True,
) -> Dict:
    data_path = Path(data_path)
    output_dir = ensure_dir(output_dir)

    setup_logging(debug)
    client = build_client(provider=provider, model=model)
    if dataset_type=="hitom":
        engine = SequentialHiToMEngine(llm_client=client, debug=debug)
    if dataset_type=="bigtom":
        engine = SequentialBigToMEngine(llm_client=client, debug=debug)
    if dataset_type=="fantom":
        engine = SequentialFanToMEngine(llm_client=client, debug=debug)
    if dataset_type == "hitom_CoT":
        engine = SequentialHiToMCoTEngine(llm_client=client, debug=debug)
    if dataset_type == "bigtom_CoT":
        engine = SequentialBigToMCoTEngine(llm_client=client, debug=debug)
    elif dataset_type == "fantom_CoT":
        engine = SequentialFanToMCoTEngine(llm_client=client, debug=debug)

    samples = load_samples(data_path, dataset_type=dataset_type, sample_limit=sample_limit)

    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    if resume:
        processed_ids, correct, existing_records = load_processed_state(predictions_path)
    else:
        processed_ids, correct, existing_records = set(), 0, []
        if predictions_path.exists():
            predictions_path.unlink()

    total_samples = len(samples)
    already_processed = sum(1 for s in samples if s.sample_id in processed_ids)

    logger.info(
        "Loaded %s existing predictions. %s/%s samples already processed.",
        len(existing_records),
        already_processed,
        total_samples,
    )

    # 先写一次初始 summary，方便一开始就看到当前状态
    summary = {
        "num_samples": total_samples,
        "num_processed": already_processed,
        "num_remaining": total_samples - already_processed,
        "num_correct": correct,
        "accuracy": (correct / already_processed) if already_processed else 0.0,
        "provider": provider,
        "model": model,
        "status": "running" if already_processed < total_samples else "finished",
        "resume_enabled": resume,
    }
    write_json(summary_path, summary)

    processed_count = already_processed

    for idx, sample in enumerate(samples, start=1):
        if sample.sample_id in processed_ids:
            logger.info(
                "Skipping sample %s/%s (sample_id=%s): already processed",
                idx,
                total_samples,
                sample.sample_id,
            )
            continue

        logger.info(
            "Processing sample %s/%s (sample_id=%s)",
            idx,
            total_samples,
            sample.sample_id,
        )

        sample = engine.process_sample(sample)
        record = sample.to_record()

        append_jsonl(predictions_path, record)

        processed_ids.add(sample.sample_id)
        processed_count += 1

        if sample.is_correct:
            correct += 1

        summary = {
            "num_samples": total_samples,
            "num_processed": processed_count,
            "num_remaining": total_samples - processed_count,
            "num_correct": correct,
            "accuracy": (correct / processed_count) if processed_count else 0.0,
            "provider": provider,
            "model": model,
            "status": "running" if processed_count < total_samples else "finished",
            "resume_enabled": resume,
        }
        write_json(summary_path, summary)

    # 最终再写一次 finished 状态
    final_summary = {
        "num_samples": total_samples,
        "num_processed": processed_count,
        "num_remaining": total_samples - processed_count,
        "num_correct": correct,
        "accuracy": (correct / processed_count) if processed_count else 0.0,
        "provider": provider,
        "model": model,
        "status": "finished" if processed_count == total_samples else "partial",
        "resume_enabled": resume,
    }
    write_json(summary_path, final_summary)

    return final_summary

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--data-path", required=True)
    run_parser.add_argument("--output-dir", default="results")
    run_parser.add_argument("--provider", default="openai")
    run_parser.add_argument("--model", default="gpt-5.4-nano")
    run_parser.add_argument("--sample-limit", type=int, default=None)
    run_parser.add_argument("--debug", action="store_true")
    run_parser.add_argument("--dataset-type", default="hitom", choices=["hitom", "bigtom", "fantom"])

    retry_parser = sub.add_parser("retry-unresolved")
    retry_parser.add_argument("--dataset-type", default="hitom", choices=["hitom", "bigtom", "fantom"])
    retry_parser.add_argument("--data-path", required=True)
    retry_parser.add_argument("--previous-output-dir", required=True)
    retry_parser.add_argument("--new-output-dir", required=True)
    retry_parser.add_argument("--provider", default="openai")
    retry_parser.add_argument("--model", default="gpt-5.4-nano")
    retry_parser.add_argument("--sample-limit", type=int, default=None)
    retry_parser.add_argument("--debug", action="store_true")

    return parser

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "run":
        summary = run_pipeline(
            data_path=args.data_path,
            output_dir=args.output_dir,
            provider=args.provider,
            model=args.model,
            dataset_type=args.dataset_type,
            sample_limit=args.sample_limit,
            debug=args.debug,
        )
        print(summary)

    elif args.command == "retry-unresolved":
        summary = run_retry_unresolved(
            data_path=args.data_path,
            previous_output_dir=args.previous_output_dir,
            new_output_dir=args.new_output_dir,
            provider=args.provider,
            model=args.model,
            dataset_type=args.dataset_type,
            sample_limit=args.sample_limit,
            debug=args.debug,
        )
        print(summary)


if __name__ == "__main__":
    main()
