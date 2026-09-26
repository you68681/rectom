from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple,Optional

from rectom.models import ChoiceSet

logger = logging.getLogger(__name__)

FACT_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)$")
NAME_TOKEN_RE = re.compile(r"\b[A-Z][a-z]+\b")

NON_CHARACTER_WORDS = {
    "The",
    "You",
    "Story",
    "Question",
    "Choices",
    "Note",
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# def extract_json_block(text: str) -> Any:
#     text = text.strip()
#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         pass
#
#     start = text.find("{")
#     end = text.rfind("}")
#     if start >= 0 and end > start:
#         return json.loads(text[start:end + 1])
#
#     raise ValueError("Could not parse JSON from model response")
#

def extract_json_block(text: str) -> Any:
    text = text.strip()

    # 1. Fast path: the whole response is already JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. If there is a closing </think>, keep only the content after the LAST one
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()

    # 3. Remove paired <think> ... </think> blocks if present
    text_wo_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 4. Try parsing the whole remaining text again
    try:
        return json.loads(text_wo_think)
    except json.JSONDecodeError:
        pass

    # 5. If JSON is wrapped in fenced code blocks, prefer those.
    #    Check from the end so we prefer the last fenced JSON block.
    code_block_matches = re.findall(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        text_wo_think,
        flags=re.IGNORECASE,
    )
    for candidate in reversed(code_block_matches):
        candidate = candidate.strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    # 6. Robust fallback:
    #    scan every '{' position from RIGHT to LEFT and try to parse a JSON object.
    #    This prefers the last valid JSON object in the response.
    decoder = json.JSONDecoder()
    brace_positions = [i for i, ch in enumerate(text_wo_think) if ch == "{"]

    for i in reversed(brace_positions):
        try:
            obj, end = decoder.raw_decode(text_wo_think[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse JSON from model response")


def parse_fact(fact: str) -> Tuple[str, Tuple[str, ...]]:
    fact = fact.strip()
    m = FACT_RE.match(fact)
    if not m:
        return fact, tuple()
    pred = m.group(1)
    raw_args = m.group(2).strip()
    if not raw_args:
        return pred, tuple()
    # Split only at top-level commas.  A plain ``str.split(',')`` corrupts
    # nested propositions such as
    # ``private_tell(Alice,Bob,in(ball,green_box))``.
    args_list: List[str] = []
    current: List[str] = []
    depth = 0
    for char in raw_args:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth = max(0, depth - 1)
            current.append(char)
        elif char == "," and depth == 0:
            args_list.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    args_list.append("".join(current).strip())
    args = tuple(arg for arg in args_list if arg)
    return pred, args


def split_story_into_steps(story: str) -> List[str]:
    steps: List[str] = []
    for raw in story.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^\d+\s*", "", line).strip()
        if line:
            steps.append(line)
    return steps


def extract_story_characters(parsed_steps: List[str]) -> List[str]:
    found: List[str] = []
    seen = set()
    for step in parsed_steps:
        for token in NAME_TOKEN_RE.findall(step):
            if token in NON_CHARACTER_WORDS:
                continue
            if token not in seen:
                seen.add(token)
                found.append(token)
    return found


def extract_question_characters(question: str, story_characters: List[str]) -> List[str]:
    ordered = []
    indices = []
    for name in story_characters:
        idx = question.find(name)
        if idx >= 0:
            indices.append((idx, name))
    indices.sort()
    for _, name in indices:
        ordered.append(name)
    return ordered



MENTAL_VERBS = r"(?:believe|thinks?|knows?|remembers?|suspects?|assumes?)"




def extract_question_characters_fantom(
    question: str,
    story_characters: List[str],
    expected_num_characters: Optional[int] = None,
) -> List[str]:
    """
    Extract the ordered chain of mental-state holders from a question,
    preserving their order of appearance in the original question text.
    """
    question = question.strip()
    matches: List[Tuple[int, int, str]] = []
    # (start_index, priority, name)
    # lower priority value = preferred when same start_index

    def add_match(name: str, start: int, priority: int) -> None:
        if name in story_characters:
            matches.append((start, priority, name))

    # 1. outermost holder
    outer_pattern = re.compile(
        rf"^(?:What|Who|Where|Which)\s+does\s+([A-Z][a-zA-Z_-]*)\s+{MENTAL_VERBS}\b"
    )
    m = outer_pattern.search(question)
    if m:
        add_match(m.group(1), m.start(1), 0)

    # 2. possessive mental holder
    possessive_pattern = re.compile(
        rf"\b([A-Z][a-zA-Z_-]*)'s\s+(?:belief|thought|view|opinion|understanding|knowledge|memory)\b"
    )
    for m in possessive_pattern.finditer(question):
        add_match(m.group(1), m.start(1), 1)

    # 3. embedded clause holder
    inner_clause_pattern = re.compile(
        rf"\b(?:who|that|about)\s+([A-Z][a-zA-Z_-]*)\s+{MENTAL_VERBS}\b"
    )
    for m in inner_clause_pattern.finditer(question):
        add_match(m.group(1), m.start(1), 2)

    # 4. fallback: any bare "X believes/thinks/knows..."
    bare_inner_pattern = re.compile(
        rf"\b([A-Z][a-zA-Z_-]*)\s+{MENTAL_VERBS}\b"
    )
    for m in bare_inner_pattern.finditer(question):
        add_match(m.group(1), m.start(1), 3)

    # Sort by textual appearance first, then by rule priority
    matches.sort(key=lambda x: (x[0], x[1]))

    # Deduplicate while preserving text order
    results: List[str] = []
    seen = set()
    for _, _, name in matches:
        if name not in seen:
            seen.add(name)
            results.append(name)

    if expected_num_characters is not None:
        expected_num_characters = max(0, min(expected_num_characters, 2))

        if len(results) > expected_num_characters:
            results = results[:expected_num_characters]

        if len(results) < expected_num_characters:
            mentioned = []
            for name in story_characters:
                idx = question.find(name)
                if idx >= 0:
                    mentioned.append((idx, name))
            mentioned.sort()

            for _, name in mentioned:
                if name not in results:
                    results.append(name)
                if len(results) >= expected_num_characters:
                    break

    return results


def normalize_bool_sequence(values: Sequence[Any]) -> List[bool]:
    out = []
    for v in values:
        if isinstance(v, bool):
            out.append(v)
        elif isinstance(v, str):
            out.append(v.strip().lower() in {"true", "1", "yes"})
        else:
            out.append(bool(v))
    return out


def fill_perspective_from_observation(
    observation_states: List[List[str]],
    observable_persistent_deltas: Optional[Sequence[Tuple[Sequence[str], Sequence[str]]]] = None,
) -> List[List[str]]:
    """
    Complete a partial state sequence strictly from left to right.

    This is Equation (2) from RECTOM for the persistent-state track:

    * an observed state replaces the current completed state;
    * an unobserved state inherits the preceding completed state;
    * an observable persistent event may then remove/add facts to that
      inherited state.

    ``observable_persistent_deltas[t]`` is a pair ``(added, removed)`` for a
    persistent event that remains observable even when the paired state is
    hidden.  In Hi-ToM this is used only for universal room-entry/room-exit
    events.  Other persistent events are already represented by an observed
    state, while transient events remain on the independent action track.

    Crucially, this function never reads a future state to complete an earlier
    state.
    """
    n = len(observation_states)
    if observable_persistent_deltas is None:
        observable_persistent_deltas = [([], []) for _ in range(n)]
    if len(observable_persistent_deltas) != n:
        raise ValueError(
            "State observations and persistent deltas must have identical lengths: "
            f"{n} != {len(observable_persistent_deltas)}"
        )

    result: List[List[str]] = []
    previous_completed: Set[str] = set()
    for observed_state, (added_facts, removed_facts) in zip(
        observation_states, observable_persistent_deltas
    ):
        if observed_state:
            current = set(observed_state)
        else:
            current = set(previous_completed)
            current.difference_update(removed_facts)
            current.update(added_facts)

        completed = sorted(current)
        result.append(completed)
        previous_completed = current
    return result


def parse_choices(choices: str) -> ChoiceSet:
    values: List[str] = []
    for part in choices.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^[A-Z]\.\s*(.*)$", part)
        if m:
            values.append(m.group(1).strip())
        else:
            values.append(part)
    return ChoiceSet(values=values)



# def parse_choices_bigtom(choices: str) -> ChoiceSet:
#     matches = re.findall(r"[A-Z]\.\s*(.*?\.)(?=\s*[A-Z]\.|$)", choices)
#
#     values: List[str] = [m.strip() for m in matches]
#
#     return ChoiceSet(values=values)

def parse_choices_bigtom(choices: str) -> ChoiceSet:
    labels = re.findall(r"([A-Z])\.", choices)

    return ChoiceSet(values=labels)

def state_contains_character(state: Iterable[str], character: str) -> bool:
    for fact in state:
        pred, args = parse_fact(fact)

        if pred == "at" and len(args) == 2 and args[0] == character:
            location = args[1]
            if location != "waiting_room":
                return True

        if pred == "in_room" and len(args) >= 1 and args[0] == character:
            if len(args) >= 2:
                location = args[1]
                if location != "waiting_room":
                    return True

    return False


def states_to_pretty_json(states: Sequence[Sequence[str]]) -> str:
    return json.dumps([sorted(set(s)) for s in states], ensure_ascii=False, indent=2)
