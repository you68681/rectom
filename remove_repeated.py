from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


INPUT_PATH = Path("/Users/clei1/Downloads/hitom_llm_project/results_pycharm_timetom_hitom_gemini_3/hitom_timetom_results.jsonl")
OUTPUT_PATH = Path("/Users/clei1/Downloads/hitom_llm_project/results_pycharm_timetom_hitom_gemini_3/hitom_timetom_results_dedup.jsonl")


def sample_id_sort_key(x: Any):
    try:
        return int(x)
    except Exception:
        return str(x)


def is_better_record(new: Dict[str, Any], old: Dict[str, Any]) -> bool:
    """
    Return True if `new` should replace `old`.

    Priority:
    1. correct == True
    2. error is None
    3. later record wins
    """
    new_correct = new.get("correct") is True
    old_correct = old.get("correct") is True

    if new_correct and not old_correct:
        return True

    if old_correct and not new_correct:
        return False

    new_no_error = new.get("error") is None
    old_no_error = old.get("error") is None

    if new_no_error and not old_no_error:
        return True

    if old_no_error and not new_no_error:
        return False

    # If quality is the same, keep the later one.
    return True


def dedup_jsonl(input_path: Path, output_path: Path) -> None:
    best_by_sample_id: Dict[Any, Dict[str, Any]] = {}

    total_lines = 0
    valid_lines = 0
    skipped_lines = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            total_lines += 1
            line = line.strip()

            if not line:
                skipped_lines += 1
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                skipped_lines += 1
                print(f"[skip] invalid JSON at line {line_no}: {e}")
                continue

            sample_id = row.get("sample_id")

            if sample_id is None:
                skipped_lines += 1
                print(f"[skip] missing sample_id at line {line_no}")
                continue

            valid_lines += 1

            old = best_by_sample_id.get(sample_id)

            if old is None or is_better_record(row, old):
                best_by_sample_id[sample_id] = row

    deduped_rows = [
        best_by_sample_id[sample_id]
        for sample_id in sorted(best_by_sample_id.keys(), key=sample_id_sort_key)
    ]

    with output_path.open("w", encoding="utf-8") as f:
        for row in deduped_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    num_correct = sum(1 for row in deduped_rows if row.get("correct") is True)
    num_bad = sum(1 for row in deduped_rows if row.get("prediction_letter") is None)

    print("Done")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total lines: {total_lines}")
    print(f"Valid lines: {valid_lines}")
    print(f"Skipped lines: {skipped_lines}")
    print(f"Unique sample_id: {len(deduped_rows)}")
    print(f"Correct: {num_correct}")
    print(f"Bad responses: {num_bad}")
    print(
        f"Accuracy: {0.0 if not deduped_rows else num_correct / len(deduped_rows):.2%}"
    )


if __name__ == "__main__":
    dedup_jsonl(INPUT_PATH, OUTPUT_PATH)