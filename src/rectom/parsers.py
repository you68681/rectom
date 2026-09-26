from __future__ import annotations

from typing import Any, Dict, List, Set

from rectom.models import DeltaGenerationResult, ObservationMaskResult, StepDelta, ActionEffectResult,FinalStateResult,ExitStepsResult,FinalStateResult_Bigtom,InitialParticipantsResult,ApplyActionsToStateResultFantom,StateTraceStepFantom,ApplyActionToStateResultFantom,QuestionReductionResult
from rectom.utils import extract_json_block, normalize_bool_sequence


def parse_delta_result(raw_text: str) -> DeltaGenerationResult:
    obj = extract_json_block(raw_text)
    chars = list(obj.get("characters", []))
    steps_data = obj.get("steps", [])
    steps: List[StepDelta] = []
    for item in steps_data:
        steps.append(
            StepDelta(
                step_index=int(item["step_index"]),
                step_text=str(item["step_text"]),
                added_facts=list(item.get("added_facts", [])),
                removed_facts=list(item.get("removed_facts", [])),
            )
        )
    return DeltaGenerationResult(characters=chars, steps=steps)


def parse_initial_participants(raw_text: str) -> InitialParticipantsResult:
    obj = extract_json_block(raw_text)
    chars = list(obj.get("characters", []))
    initial_present = list(obj.get("initial_present_characters", []))
    return InitialParticipantsResult(
        characters=chars,
        initial_present_characters=initial_present,
    )


def parse_single_step_delta(raw_text: str) -> StepDelta:
    obj = extract_json_block(raw_text)
    return StepDelta(
        step_index=int(obj["step_index"]),
        step_text=str(obj["step_text"]),
        added_facts=list(obj.get("added_facts", [])),
        removed_facts=list(obj.get("removed_facts", [])),
    )


# def parse_observation_mask(raw_text: str) -> ObservationMaskResult:
#     obj = extract_json_block(raw_text)
#     return ObservationMaskResult(
#         character=str(obj.get("character", "")).strip(),
#         visible=normalize_bool_sequence(obj.get("visible", [])),
#     )


# def normalize_observation_basis_sequence(values) -> list[str]:
#     result = []
#     if not isinstance(values, list):
#         return result
#
#     for v in values:
#         if v is None:
#             result.append("not_observable")
#         else:
#             s = str(v).strip()
#             result.append(s if s else "not_observable")
#     return result
#
#
# def parse_observation_mask(raw_text: str) -> ObservationMaskResult:
#     obj = extract_json_block(raw_text)
#     return ObservationMaskResult(
#         character=str(obj.get("character", "")).strip(),
#         observation_basis=normalize_observation_basis_sequence(
#             obj.get("observation_basis", [])
#         ),
#     )


def normalize_observation_basis_sequence(
    values,
    default_value: str = "not_observable",
) -> list[str]:
    result = []
    if not isinstance(values, list):
        return result

    for v in values:
        if v is None:
            result.append(default_value)
        else:
            s = str(v).strip()
            result.append(s if s else default_value)
    return result


def parse_observation_mask(
    raw_text: str,
    default_value: str = "not_observable",
) -> ObservationMaskResult:
    obj = extract_json_block(raw_text)
    return ObservationMaskResult(
        character=str(obj.get("character", "")).strip(),
        observation_basis=normalize_observation_basis_sequence(
            obj.get("observation_basis", []),
            default_value=default_value,
        ),
    )


def parse_question_reduction_result(raw_text: str) -> QuestionReductionResult:
    obj = extract_json_block(raw_text)
    return QuestionReductionResult(
        original_question=str(obj.get("original_question", "")).strip(),
        character_chain=[str(x).strip() for x in obj.get("character_chain", [])],
        zero_order_question=str(obj.get("zero_order_question", "")).strip(),
    )

def normalize_fact_list(facts: List[str] | Set[str]) -> List[str]:
    return sorted(set(f.strip() for f in facts if f and f.strip()))


def normalize_action_list(facts: List[str] | Set[str]) -> List[str]:
    result: List[str] = []
    seen = set()

    for f in facts:
        if not f:
            continue
        s = f.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            result.append(s)

    return result


def normalize_effective_action_sequence(
    values,
    default_value: str = "not_effective",
) -> list[str]:
    result: list[str] = []
    if not isinstance(values, list):
        return result

    for v in values:
        if v is None:
            result.append(default_value)
        else:
            s = str(v).strip()
            result.append(s if s else default_value)
    return result



def normalize_exit_steps(value) -> dict[str, int]:
    result: dict[str, int] = {}
    if not isinstance(value, dict):
        return result

    for k, v in value.items():
        key = str(k).strip()
        if not key:
            continue
        try:
            step_idx = int(v)
        except (TypeError, ValueError):
            continue
        if step_idx > 0:
            result[key] = step_idx

    return result


def parse_action_effect(raw_text: str) -> ActionEffectResult:
    obj = extract_json_block(raw_text)
    return ActionEffectResult(
        character=str(obj.get("character", "")).strip(),
        exit_steps=normalize_exit_steps(obj.get("exit_steps", {})),
        effective_actions=normalize_effective_action_sequence(
            obj.get("effective_actions", []),
            default_value="not_effective",
        ),
    )


def normalize_symbolic_fact_sequence(values) -> list[str]:
    result: list[str] = []
    if not isinstance(values, list):
        return result

    seen = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def parse_final_state_result(raw_text: str) -> FinalStateResult:
    obj = extract_json_block(raw_text)
    return FinalStateResult(
        final_state=normalize_symbolic_fact_sequence(obj.get("final_state", []))
    )

def parse_final_state_result_bigtom(raw_text: str) -> FinalStateResult_Bigtom:
    obj = extract_json_block(raw_text)
    return FinalStateResult_Bigtom(
        final_state=normalize_symbolic_fact_sequence(obj.get("final_state", [])),
        added_facts=normalize_symbolic_fact_sequence(obj.get("action_added_facts", [])),
        removed_facts=normalize_symbolic_fact_sequence(obj.get("action_removed_facts", []))
    )


def parse_final_state_result_fantom(raw_text: str) -> ApplyActionsToStateResultFantom:
    obj = extract_json_block(raw_text)
    return ApplyActionsToStateResultFantom(
        state_trace=[
            StateTraceStepFantom(
                step=int(item["step"]),
                state_before=normalize_symbolic_fact_sequence(item.get("state_before", [])),
                action=normalize_symbolic_fact_sequence(item.get("action", [])),
                action_added_facts=normalize_symbolic_fact_sequence(item.get("action_added_facts", [])),
                action_removed_facts=normalize_symbolic_fact_sequence(item.get("action_removed_facts", [])),
                state_after=normalize_symbolic_fact_sequence(item.get("state_after", [])),
            )
            for item in obj.get("state_trace", [])
        ],
    )


def parse_final_state_result_fantom_v2(raw_text: str) -> ApplyActionToStateResultFantom:
    obj = extract_json_block(raw_text)
    return ApplyActionToStateResultFantom(
        state_before=normalize_symbolic_fact_sequence(obj.get("state_before", [])),
        action=normalize_symbolic_fact_sequence(obj.get("action", [])),
        action_added_facts=normalize_symbolic_fact_sequence(obj.get("action_added_facts", [])),
        action_removed_facts=normalize_symbolic_fact_sequence(obj.get("action_removed_facts", [])),
        state_after=normalize_symbolic_fact_sequence(obj.get("state_after", [])),
    )


def parse_answer_result(raw_text: str) -> Dict[str, Any]:
    obj = extract_json_block(raw_text)
    return {
        "predicted_answer": str(obj.get("predicted_answer", "")).strip(),
        "reasoning_summary": str(obj.get("reasoning_summary", "")).strip(),
    }


def parse_exit_steps(raw_text: str) -> ExitStepsResult:
    obj = extract_json_block(raw_text)
    return ExitStepsResult(
        exit_steps=normalize_exit_steps(obj.get("exit_steps", {})),
    )
