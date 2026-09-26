from __future__ import annotations

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
from hitom_llm.clients.factory import build_client
from hitom_llm.models import StorySample
from hitom_llm.runtime.engine import SequentialHiToMEngine
from hitom_llm.utils import ensure_dir, load_json, split_story_into_steps, write_json, write_jsonl


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
    return " ".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))

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


def load_samples(data_path: Path, sample_limit: int | None = None) -> List[StorySample]:
    raw = load_json(data_path)
    rows = raw["data"]

    samples: List[StorySample] = []

    for row in rows:
        # Only keep samples with story_length == 1
        if row.get("story_length") != 1:
            continue
        # if not row.get("deception"):
        #     continue

        cleaned_story = strip_story_preamble(row["story"])
        original_steps = split_story_into_steps(cleaned_story)

        # original_steps = split_story_into_steps(row["story"])

        # Remove any waiting_room-enter step, no matter where it appears
        filtered_steps = [step for step in original_steps if not should_drop_step(step)]

        # Rebuild story text so numbering stays sequential
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



# def load_samples(data_path: Path, sample_limit: int | None = None) -> List[StorySample]:
#     raw = load_json(data_path)
#     rows = raw["data"]
#     if sample_limit is not None:
#         rows = rows[:sample_limit]
#
#     samples: List[StorySample] = []
#     for row in rows:
#         samples.append(
#             StorySample(
#                 sample_id=int(row["sample_id"]),
#                 story=row["story"],
#                 question=row["question"],
#                 choices=row["choices"],
#                 answer=row["answer"],
#                 parsed_steps=split_story_into_steps(row["story"]),
#                 prompting_type=row.get("prompting_type"),
#                 question_order=row.get("question_order"),
#                 deception=row.get("deception"),
#             )
#         )
#     return samples


# def run_pipeline(
#     data_path,
#     output_dir,
#     provider: str,
#     model: str,
#     sample_limit: int | None = None,
#     debug: bool = False,
# ) -> Dict:
#     data_path = Path(data_path)
#     output_dir = ensure_dir(output_dir)
#
#     setup_logging(debug)
#     client = build_client(provider=provider, model=model)
#     engine = SequentialHiToMEngine(llm_client=client, debug=debug)
#
#     samples = load_samples(data_path, sample_limit=sample_limit)
#     samples=samples[-3:-1]
#     predictions = []
#     correct = 0
#
#     for idx, sample in enumerate(samples, start=1):
#         logger.info("Processing sample %s/%s (sample_id=%s)", idx, len(samples), sample.sample_id)
#         sample = engine.process_sample(sample)
#         predictions.append(sample.to_record())
#         if sample.is_correct:
#             correct += 1
#
#     summary = {
#         "num_samples": len(samples),
#         "num_correct": correct,
#         "accuracy": (correct / len(samples)) if samples else 0.0,
#         "provider": provider,
#         "model": model,
#     }
#
#     write_jsonl(output_dir / "predictions.jsonl", predictions)
#     write_json(output_dir / "summary.json", summary)
#     return summary


def load_unresolved_sample_ids(
    data_path: Path,
    previous_predictions_path: Path,
    sample_limit: int | None = None,
) -> List[int]:
    """
    Return sample_ids that should be retried:
      1) samples missing from previous predictions.jsonl
      2) samples explicitly marked failed in previous records (if such fields exist)
    """
    samples = load_samples(data_path, sample_limit=sample_limit)
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
    sample_limit: int | None = None,
    debug: bool = False,
) -> Dict:
    data_path = Path(data_path)
    previous_output_dir = Path(previous_output_dir)
    new_output_dir = ensure_dir(new_output_dir)

    setup_logging(debug)
    client = build_client(provider=provider, model=model)
    engine = SequentialHiToMEngine(llm_client=client, debug=debug)

    previous_predictions_path = previous_output_dir / "predictions.jsonl"
    retry_sample_ids = set(
        load_unresolved_sample_ids(
            data_path=data_path,
            previous_predictions_path=previous_predictions_path,
            sample_limit=sample_limit,
        )
    )

    all_samples = load_samples(data_path, sample_limit=sample_limit)
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
            records.append(json.loads(line))
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
    sample_limit: int | None = None,
    debug: bool = False,
    resume: bool = True,
) -> Dict:
    data_path = Path(data_path)
    output_dir = ensure_dir(output_dir)

    setup_logging(debug)
    client = build_client(provider=provider, model=model)
    engine = SequentialHiToMEngine(llm_client=client, debug=debug)

    samples = load_samples(data_path, sample_limit=sample_limit)

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

    retry_parser = sub.add_parser("retry-unresolved")
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
            sample_limit=args.sample_limit,
            debug=args.debug,
        )
        print(summary)


if __name__ == "__main__":
    main()
