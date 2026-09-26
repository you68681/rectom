import os
import re
import glob
import pandas as pd


def parse_token_filename(path: str) -> dict | None:
    """
    Parse metadata from token usage filename.

    Expected examples:
    - token_usage_hitom_5.4_100.csv
    - token_usage_hitom_5.4_CoT_100.csv
    - token_usage_fantom_gemini_50.csv
    - token_usage_fanton_gemini_CoT_100.csv
    - token_usage_bigtom_false_belief_gemini_50.csv
    - token_usage_bigtom_true_belief_gemini_50_CoT.csv
    """

    filename = os.path.basename(path)
    name = filename.lower()

    if not name.startswith("token_usage_"):
        return None

    # Remove duplicate suffix like "(1).csv"
    name = re.sub(r"\(\d+\)\.csv$", ".csv", name)

    # Remove prefix and extension
    core = name.replace("token_usage_", "").replace(".csv", "")

    # -------------------------
    # Dataset
    # -------------------------
    if "hitom" in core:
        dataset = "HiToM"
    elif "fantom" in core or "fanton" in core:
        # "fanton" is treated as typo of fantom
        dataset = "FANTOM"
    elif "bigtom" in core:
        # Merge false_belief and true_belief into BigToM
        dataset = "BigToM"
    else:
        dataset = "Unknown"

    # -------------------------
    # Model
    # -------------------------
    if "gemini" in core:
        model = "Gemini"
    elif "5.4" in core or "gpt5.4" in core or "gpt_5.4" in core:
        model = "GPT-5.4"
    else:
        model = "Unknown"

    # -------------------------
    # Method
    # -------------------------
    if "cot" in core:
        method = "CoT"
    elif "simtom" in core:
        method = "SimToM"
    elif "timtom" in core:
        method = "TimeToM"
    else:
        method = "RecToM"

    # -------------------------
    # Number of questions
    # -------------------------
    m = re.search(r"_(50|100)(?:_|$)", core)
    if m is None:
        n_questions = None
    else:
        n_questions = int(m.group(1))

    # Optional: keep BigToM subtype for checking
    if "false_belief" in core:
        belief_type = "false_belief"
    elif "true_belief" in core:
        belief_type = "true_belief"
    else:
        belief_type = None

    return {
        "file": filename,
        "dataset": dataset,
        "model": model,
        "method": method,
        "n_questions": n_questions,
        "belief_type": belief_type,
    }


def summarize_token_files(input_dir: str, output_csv: str = "token_summary_per_question.csv"):
    paths = sorted(glob.glob(os.path.join(input_dir, "token_usage*.csv")))

    rows = []

    for path in paths:
        meta = parse_token_filename(path)

        if meta is None:
            continue

        if meta["n_questions"] is None:
            print(f"[skip] Cannot parse number of questions from filename: {path}")
            continue

        df = pd.read_csv(path)

        required_cols = [
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cached_tokens",
            "reasoning_tokens",
        ]

        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        rows.append({
            **meta,
            "api_calls": len(df),
            "sum_prompt_tokens": int(df["prompt_tokens"].sum()),
            "sum_completion_tokens": int(df["completion_tokens"].sum()),
            "sum_total_tokens": int(df["total_tokens"].sum()),
            "sum_cached_tokens": int(df["cached_tokens"].sum()),
            "sum_reasoning_tokens": int(df["reasoning_tokens"].sum()),
        })

    per_file = pd.DataFrame(rows)

    if per_file.empty:
        raise ValueError("No valid token_usage*.csv files found.")

    # --------------------------------------------------
    # Merge BigToM false_belief and true_belief here
    # Group by dataset/model/method
    # --------------------------------------------------
    summary = (
        per_file
        .groupby(["dataset", "model", "method"], as_index=False)
        .agg(
            n_questions=("n_questions", "sum"),
            api_calls=("api_calls", "sum"),
            sum_prompt_tokens=("sum_prompt_tokens", "sum"),
            sum_completion_tokens=("sum_completion_tokens", "sum"),
            sum_total_tokens=("sum_total_tokens", "sum"),
            sum_cached_tokens=("sum_cached_tokens", "sum"),
            sum_reasoning_tokens=("sum_reasoning_tokens", "sum"),
        )
    )

    # Average token cost per question
    summary["avg_prompt_tokens_per_question"] = (
        summary["sum_prompt_tokens"] / summary["n_questions"]
    )
    summary["avg_completion_tokens_per_question"] = (
        summary["sum_completion_tokens"] / summary["n_questions"]
    )
    summary["avg_total_tokens_per_question"] = (
        summary["sum_total_tokens"] / summary["n_questions"]
    )
    summary["avg_cached_tokens_per_question"] = (
        summary["sum_cached_tokens"] / summary["n_questions"]
    )
    summary["avg_reasoning_tokens_per_question"] = (
        summary["sum_reasoning_tokens"] / summary["n_questions"]
    )
    summary["avg_api_calls_per_question"] = (
        summary["api_calls"] / summary["n_questions"]
    )

    summary = summary.sort_values(
        ["dataset", "model", "method"]
    ).reset_index(drop=True)

    summary.to_csv(output_csv, index=False)

    return per_file, summary


if __name__ == "__main__":
    input_dir = "/Users/clei1/Downloads/hitom_llm_project/token_logs"  # 改成你的 token_usage csv 所在文件夹
    output_csv = "token_summary_per_question.csv"

    per_file, summary = summarize_token_files(
        input_dir=input_dir,
        output_csv=output_csv,
    )

    print("\nPer-file parsed metadata:")
    print(per_file[[
        "file",
        "dataset",
        "model",
        "method",
        "n_questions",
        "belief_type",
        "api_calls",
        "sum_total_tokens",
    ]].to_string(index=False))

    print("\nSummary: average token usage per question")
    print(summary[[
        "dataset",
        "model",
        "method",
        "n_questions",
        "api_calls",
        "avg_api_calls_per_question",
        "avg_prompt_tokens_per_question",
        "avg_completion_tokens_per_question",
        "avg_reasoning_tokens_per_question",
        "avg_total_tokens_per_question",
    ]].to_string(index=False))

    print(f"\nSaved summary to: {output_csv}")