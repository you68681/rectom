from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

GLOBAL_SEED=42

try:
    from hitom_llm.utils import load_json  # type: ignore
except Exception:
    def load_json(path: str | Path):
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)


ANSWER_LINE_RE = re.compile(r"answer\s*[:\-]?\s*([A-Z])\b", flags=re.IGNORECASE)
TIME_ID_RE = re.compile(r"\bt(\d+)\b", flags=re.IGNORECASE)


@dataclass
class StorySample:
    sample_id: int | str
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
    dataset_name: Optional[str] = None
    tom_type: Optional[str] = None


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_for_compare(text: str) -> str:
    return normalize_whitespace(text).lower()


def parse_choice_string(choice_string: str) -> List[Tuple[str, str]]:
    matches = list(
        re.finditer(
            r"([A-Z])\.\s*(.*?)(?=(?:,\s*[A-Z]\.\s)|$)",
            (choice_string or "").strip(),
            flags=re.DOTALL,
        )
    )
    return [(m.group(1).strip(), m.group(2).strip().rstrip(",")) for m in matches]


def answer_text_to_letter(answer_text: str, choice_string: str) -> str:
    target = normalize_for_compare(answer_text)
    for letter, text in parse_choice_string(choice_string):
        if normalize_for_compare(text) == target:
            return letter
    raise ValueError(f"Could not map answer text {answer_text!r} to choices {choice_string!r}")


def answer_letter_to_text(answer_letter: str, choice_string: str) -> str:
    letter = answer_letter.strip().upper()
    for cur_letter, text in parse_choice_string(choice_string):
        if cur_letter == letter:
            return text
    raise ValueError(f"Could not map answer letter {answer_letter!r} to choices {choice_string!r}")


def split_story_into_sentences_from_period(story: str) -> List[str]:
    story = re.sub(r"\s+", " ", str(story or "")).strip()
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
    return "\n".join(f"{i}. {step.rstrip('.')}." for i, step in enumerate(steps, start=1))


def _read_semicolon_csv_rows(path: Path) -> Iterable[List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        for row in reader:
            if row:
                yield row


def make_binary_choices(correct_answer: str, wrong_answer: str, seed_key: str) -> Tuple[str, str]:
    rng = random.Random(seed_key)
    items = [("correct", correct_answer.strip()), ("wrong", wrong_answer.strip())]
    rng.shuffle(items)
    rendered: List[str] = []
    answer_letter = ""
    for label, (tag, text) in zip(["A", "B"], items):
        rendered.append(f"{label}. {text}")
        if tag == "correct":
            answer_letter = label
    return ", ".join(rendered), answer_letter


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
                answer=answer_text_to_letter(row["answer"], row["choices"]),
                prompting_type=row.get("prompting_type"),
                question_order=row.get("question_order"),
                deception=row.get("deception"),
                dataset_name="hitom",
                tom_type=None,
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

    print(len(samples))
    return samples


def load_samples_csv(data_path: Path, sample_limit: int | None = None) -> List[StorySample]:
    samples: List[StorySample] = []
    for idx, row in enumerate(_read_semicolon_csv_rows(Path(data_path))):
        if len(row) < 4:
            raise ValueError(
                f"CSV row {idx} has fewer than 4 columns. Expected story/question/correct_answer/wrong_answer."
            )
        story_raw, question, correct_answer, wrong_answer = [str(x).strip() for x in row[:4]]
        story = rebuild_story_from_steps(split_story_into_sentences_from_period(story_raw))
        choices, answer_letter = make_binary_choices(
            correct_answer=correct_answer,
            wrong_answer=wrong_answer,
            seed_key=f"csv::{idx}::{question}",
        )
        samples.append(
            StorySample(
                sample_id=idx,
                story=story,
                question=question,
                choices=choices,
                answer=answer_letter,
                prompting_type="CoTP",
                question_order=1,
                deception=None,
                dataset_name="bigtom",
                tom_type=None,
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


def load_samples_fantom(data_path: Path, sample_limit: int | None = None, use_short_context: bool = True) -> List[StorySample]:
    raw = load_json(data_path)
    rows = raw if isinstance(raw, list) else raw.get("data", [])
    samples: List[StorySample] = []
    for idx, row in enumerate(rows):
        story = (row.get("short_context") if use_short_context else row.get("full_context")) or ""
        story = story.strip()
        if not story:
            continue

        parsed_steps = split_story_into_steps_from_newline(story)
        if len(parsed_steps)>=18:
            continue

        for bidx, belief in enumerate(row.get("beliefQAs", [])):
            choices, answer_letter = make_binary_choices(
                correct_answer=str(belief["correct_answer"]).strip(),
                wrong_answer=str(belief["wrong_answer"]).strip(),
                seed_key=f"fantom::{row.get('set_id', idx)}::{bidx}",
            )
            samples.append(
                StorySample(
                    sample_id=f"{row.get('set_id', idx)}-{bidx}",
                    story=story,
                    question=str(belief["question"]).strip(),
                    choices=choices,
                    answer=answer_letter,
                    prompting_type="CoTP",
                    question_order=None,
                    deception=None,
                    dataset_name="fantom",
                    tom_type=belief.get("tom_type"),
                )
            )
            # if sample_limit is not None and len(samples) >= sample_limit:
            #     return samples
    if sample_limit is not None:
        rng = random.Random(GLOBAL_SEED)
        samples = rng.sample(
            samples,
            k=min(sample_limit, len(samples)),
        )

    print(len(samples))
    return samples


def load_story_samples(
    dataset_name: str,
    data_path: str | Path,
    sample_limit: Optional[int] = None,
    fantom_use_short_context: bool = True,
) -> List[StorySample]:
    dataset_name = dataset_name.lower().strip()
    data_path = Path(data_path)

    if dataset_name in {"hitom", "hi-tom"}:
        return load_samples_hitom(data_path, sample_limit=sample_limit)
    if dataset_name in {"csv", "stories_csv", "story_csv", "bigtom", "big-tom"}:
        return load_samples_csv(data_path, sample_limit=sample_limit)
    if dataset_name in {"fantom", "fan-tom"}:
        return load_samples_fantom(data_path, sample_limit=sample_limit, use_short_context=fantom_use_short_context)
    raise ValueError(f"Unsupported dataset_name={dataset_name!r}")


FINAL_CHOICE_RE = re.compile(
    r"(?im)^\s*(?:answer|choice|conclusion|conlusion|result|prediction|final\s*answer|final\s*choice|final\s*conclusion)\s*[:：]\s*([A-Z])\b"
)

INLINE_CHOICE_RE = re.compile(
    r"(?i)\b(?:answer|choice|conclusion|conlusion|result|prediction|final\s*answer|final\s*choice|final\s*conclusion)\s*(?:is|=|:|：)?\s*([A-Z])\b"
)


def extract_choice_letter(text: str, choice_string: str) -> Optional[str]:
    if not text:
        return None

    text = str(text).strip()

    # 1. Strongest signal:
    # Answer: A
    # Choice: H
    # Final answer: C
    # Final choice: D
    m = FINAL_CHOICE_RE.search(text)
    if m:
        return m.group(1).upper()

    # 2. Also catch inline forms:
    # The answer is A
    # My choice is H
    # Therefore, final answer is C
    m = INLINE_CHOICE_RE.search(text)
    if m:
        return m.group(1).upper()

    # 3. Existing answer-line rule, if you still want to keep it
    m = ANSWER_LINE_RE.search(text)
    if m:
        return m.group(1).upper()

    # 4. Catch lines like:
    # A. something
    # B) something
    for line in text.splitlines():
        mm = re.match(r"^\s*([A-Z])[\.\)]\s+", line.strip())
        if mm:
            return mm.group(1).upper()

    # 5. Match generated text against choice text
    normalized = normalize_for_compare(text)
    for letter, choice_text in parse_choice_string(choice_string):
        choice_norm = normalize_for_compare(choice_text)
        if choice_norm in normalized or normalized in choice_norm:
            return letter

    # 6. Last fallback: return the last standalone capital letter
    tokens = re.findall(r"\b[A-Z]\b", text)
    if tokens:
        return tokens[-1].upper()

    return None


def strip_reasoning_keep_final_answer(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    m = re.search(r"(Answer\s*:\s*[^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.match(r"^\s*([A-Z])[\.\)]\s+.*$", text.strip())
    if m2:
        return text.strip()
    return text.strip()


def classify_scenario(dataset_name: str) -> str:
    name = str(dataset_name or "").strip().lower()
    if name in {"fantom", "fan-tom"}:
        return "dialogue"
    if name in {"hitom", "hi-tom", "bigtom", "big-tom", "csv", "stories_csv", "story_csv"}:
        return "reading"
    raise ValueError(f"Unsupported dataset_name for scenario classification: {dataset_name!r}")


def extract_characters(text: str) -> List[str]:
    chars: List[str] = []
    seen = set()

    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]

    for line in lines:
        # Support:
        # "1. Noor ..."
        # "1) Noor ..."
        # "1 Noor ..."
        line_clean = re.sub(r"^\d+(?:[\.\)]\s*|\s+)", "", line).strip()

        # 1) Dialogue speaker
        dm = re.match(r"^([A-Z][a-zA-Z'_-]+)\s*:", line_clean)
        if dm:
            name = dm.group(1)
            if name not in seen:
                seen.add(name)
                chars.append(name)
            continue

        # 2) Enter-list pattern
        em = re.search(r"^(.+?)\s+entered\s+the\s+[A-Za-z_]+", line_clean)
        if em:
            prefix = em.group(1).replace(" and ", ", ")
            for piece in [p.strip() for p in prefix.split(",") if p.strip()]:
                if re.fullmatch(r"[A-Z][a-zA-Z'_-]+", piece) and piece not in seen:
                    seen.add(piece)
                    chars.append(piece)
            continue

        # 3) Narrative sentence starting with a character name
        sm = re.match(r"^([A-Z][a-zA-Z'_-]+)\b", line_clean)
        if sm:
            name = sm.group(1)
            if name not in seen:
                seen.add(name)
                chars.append(name)

    return chars


def infer_question_order(sample: StorySample) -> int:
    dataset_name = str(sample.dataset_name or "").strip().lower()
    if dataset_name in {"hitom", "hi-tom"}:
        if sample.question_order is None:
            raise ValueError(f"Hi-ToM sample {sample.sample_id!r} is missing question_order.")
        return int(sample.question_order)
    if dataset_name in {"bigtom", "big-tom", "csv", "stories_csv", "story_csv"}:
        return 1
    if dataset_name in {"fantom", "fan-tom"}:
        tom_type = str(sample.tom_type or "").strip().lower()
        if tom_type == "first-order":
            return 1
        if tom_type.startswith("second-order"):
            return 2
        if tom_type.startswith("third-order"):
            return 3
        raise ValueError(f"Unsupported FanToM tom_type={sample.tom_type!r} for sample {sample.sample_id!r}")
    raise ValueError(f"Unsupported dataset_name={sample.dataset_name!r} for question-order inference.")


def infer_question_characters(question: str, characters: Sequence[str]) -> List[str]:
    ordered = []
    for ch in characters:
        idx = question.find(ch)
        if idx >= 0:
            ordered.append((idx, ch))
    ordered.sort()
    return [ch for _, ch in ordered]


TIME_LINE_START_RE = re.compile(r"^\s*t(\d+)\b", flags=re.IGNORECASE)


def extract_time_ids(text: str) -> List[int]:
    ids = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        m = TIME_LINE_START_RE.match(stripped)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def lines_by_time(text: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for line in str(text or "").splitlines():
        stripped = line.strip()
        m = TIME_LINE_START_RE.match(stripped)
        if m:
            mapping[int(m.group(1))] = stripped
    return mapping


def build_common_belief_text(tbsc_texts: Sequence[str]) -> str:
    valid_texts = [str(t) for t in tbsc_texts if str(t).strip()]
    if not valid_texts:
        return ""

    time_sets = [set(extract_time_ids(text)) for text in valid_texts]
    if not time_sets:
        return ""

    common_ids = sorted(set.intersection(*time_sets)) if len(time_sets) >= 2 else sorted(time_sets[0])
    if not common_ids:
        return ""

    # Prefer the longest text as source, since it is more likely to contain all lines.
    source_text = max(valid_texts, key=len)
    source_map = lines_by_time(source_text)

    lines: List[str] = []
    for tid in common_ids:
        if tid in source_map:
            lines.append(source_map[tid])

    if lines:
        return "\n".join(lines)

    # Fallback: reconstruct by scanning all texts if the chosen source missed some lines.
    merged_map: Dict[int, str] = {}
    for text in valid_texts:
        merged_map.update(lines_by_time(text))

    lines = [merged_map[tid] for tid in common_ids if tid in merged_map]
    if lines:
        return "\n".join(lines)

    # Final fallback: at least return the shared time ids
    return "\n".join(f"t{tid}" for tid in common_ids)

def reduce_higher_order_question_one_step(original_question: str, participants: List[str]) -> str:
    question = str(original_question or "").strip()
    if len(participants) <= 1:
        return question

    outer = re.escape(participants[0])

    # Pattern 1:
    # Where does A think B thinks C thinks X?
    # -> Where does B think C thinks X?
    pattern_where_think = re.compile(
        rf"^(Where\s+does\s+){outer}\s+think\s+(.+)$",
        flags=re.IGNORECASE,
    )
    m = pattern_where_think.match(question)
    if m:
        prefix = m.group(1)
        rest = m.group(2).strip()
        rest = re.sub(
            r"^([A-Z][a-zA-Z'_-]+)\s+thinks\b",
            r"\1 think",
            rest,
            count=1,
        )
        return f"{prefix}{rest}"

    # Pattern 2:
    # What does A think B thinks C thinks about X?
    # -> What does B think C thinks about X?
    pattern_what_think = re.compile(
        rf"^(What\s+does\s+){outer}\s+think\s+(.+)$",
        flags=re.IGNORECASE,
    )
    m = pattern_what_think.match(question)
    if m:
        prefix = m.group(1)
        rest = m.group(2).strip()
        rest = re.sub(
            r"^([A-Z][a-zA-Z'_-]+)\s+thinks\b",
            r"\1 think",
            rest,
            count=1,
        )
        return f"{prefix}{rest}"

    # Pattern 3:
    # What does A believe B believes C believes about X?
    # -> What does B believe C believes about X?
    pattern_what_believe = re.compile(
        rf"^(What\s+does\s+){outer}\s+believe\s+(.+)$",
        flags=re.IGNORECASE,
    )
    m = pattern_what_believe.match(question)
    if m:
        prefix = m.group(1)
        rest = m.group(2).strip()
        rest = re.sub(
            r"^([A-Z][a-zA-Z'_-]+)\s+believes\b",
            r"\1 believe",
            rest,
            count=1,
        )
        return f"{prefix}{rest}"

    # Fallback: remove only the outermost belief operator once
    fallback_patterns = [
        re.compile(rf"\b{outer}\s+think\s+", flags=re.IGNORECASE),
        re.compile(rf"\b{outer}\s+thinks\s+", flags=re.IGNORECASE),
        re.compile(rf"\b{outer}\s+believe\s+", flags=re.IGNORECASE),
        re.compile(rf"\b{outer}\s+believes\s+", flags=re.IGNORECASE),
    ]

    for pattern in fallback_patterns:
        new_question = pattern.sub("", question, count=1)
        if new_question != question:
            return new_question.strip()

    return question


def reduce_higher_order_question_to_first_order(original_question: str, participants: List[str]) -> str:
    question = str(original_question or "").strip()

    if len(participants) <= 1:
        return question

    current_question = question
    current_participants = list(participants)

    while len(current_participants) > 1:
        next_question = reduce_higher_order_question_one_step(
            current_question,
            current_participants,
        )

        # safety: stop if reducer failed to make progress
        if next_question == current_question:
            break

        current_question = next_question
        current_participants = current_participants[1:]

    return current_question