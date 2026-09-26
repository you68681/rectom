from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


INPUT_PATH = Path(
    "/Users/clei1/Downloads/hitom_llm_project/results_pycharm_timetom_hitom_gemini_3/hitom_timetom_results.jsonl"
)
OUTPUT_PATH = Path(
    "/Users/clei1/Downloads/hitom_llm_project/results_pycharm_timetom_hitom_gemini_3/hitom_timetom_results_clearn.jsonl"
)


BAD_KEYWORDS = [
    "PoolTimeout",
    "ConnectError",
]


def has_bad_error(row: Dict[str, Any], raw_line: str) -> bool:
    # 先检查整行，最保险
    for keyword in BAD_KEYWORDS:
        if keyword in raw_line:
            return True

    fields_to_check = [
        "error",
        "prediction_raw",
        "prediction_clean",
    ]

    for field in fields_to_check:
        value = row.get(field)
        if value is None:
            continue

        value_str = str(value)
        for keyword in BAD_KEYWORDS:
            if keyword in value_str:
                return True

    return False


def remove_bad_error_rows(input_path: Path, output_path: Path) -> None:
    total_lines = 0
    kept_lines = 0
    removed_lines = 0
    skipped_invalid = 0

    removed_by_keyword = {keyword: 0 for keyword in BAD_KEYWORDS}

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            total_lines += 1
            raw_line = line.rstrip("\n")

            if not raw_line.strip():
                continue

            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as e:
                skipped_invalid += 1
                print(f"[skip] invalid JSON at line {line_no}: {e}")
                continue

            matched_keywords = [
                keyword for keyword in BAD_KEYWORDS if keyword in raw_line
            ]

            if has_bad_error(row, raw_line):
                removed_lines += 1
                sample_id = row.get("sample_id", "UNKNOWN")

                if matched_keywords:
                    for keyword in matched_keywords:
                        removed_by_keyword[keyword] += 1
                    reason = ",".join(matched_keywords)
                else:
                    reason = "UNKNOWN_BAD_ERROR"

                print(f"[remove] line={line_no}, sample_id={sample_id}, reason={reason}")
                continue

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept_lines += 1

    print("Done")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total lines: {total_lines}")
    print(f"Kept lines: {kept_lines}")
    print(f"Removed bad-error lines: {removed_lines}")
    print(f"Skipped invalid JSON lines: {skipped_invalid}")
    print("Removed by keyword:")
    for keyword, count in removed_by_keyword.items():
        print(f"  {keyword}: {count}")


if __name__ == "__main__":
    remove_bad_error_rows(INPUT_PATH, OUTPUT_PATH)