from __future__ import annotations
import re
import logging
from typing import Dict, List, Sequence, Set
from typing import Sequence
from rectom.models import DeltaGenerationResult, StorySample,ActionEffectResult,ExitStepsResult,FinalStateResult_Bigtom,StepDelta,ValidationIssue,ValidationResult,QuestionReductionResult
from rectom.parsers import parse_answer_result, parse_delta_result, parse_observation_mask, parse_single_step_delta,parse_action_effect,normalize_fact_list, normalize_action_list,parse_final_state_result,parse_exit_steps,parse_final_state_result_bigtom,parse_initial_participants,parse_final_state_result_fantom,parse_final_state_result_fantom_v2,parse_question_reduction_result
from rectom.prompts.templates import (
    build_answer_prompt,
    build_answer_repair_prompt,
build_answer_prompt_bigtom,
build_answer_prompt_bigtom_fix,
build_answer_repair_prompt_bigtom,
build_answer_repair_prompt_bigtom_fix,
build_answer_prompt_fantom,
build_answer_repair_prompt_fantom,
build_answer_prompt_fantom_from_conversations,
build_answer_repair_prompt_fantom_from_conversations,
build_answer_prompt_fantom_with_invisible_facts,
build_answer_repair_prompt_fantom_with_invisible_facts,
build_answer_prompt_fantom_from_conversations_with_invisible_conversations,
build_answer_repair_prompt_fantom_from_conversations_with_invisible_conversations,
    build_delta_prompt,
    build_delta_prompt_fantom,
    build_delta_prompt_bigtom,
    build_observation_mask_prompt,
    build_observation_mask_repair_prompt,
    build_state_observation_mask_prompt,
    build_state_observation_mask_repair_prompt,
    build_state_observation_mask_prompt_bigtom,
    build_state_observation_mask_repair_prompt_bigtom,
    build_state_observation_mask_prompt_bigtom_fix,
    build_state_observation_mask_repair_prompt_bigtom_fix,
    build_state_observation_mask_prompt_fantom,
    build_state_observation_mask_repair_prompt_fantom,
    build_action_observation_mask_prompt,
    build_action_observation_mask_repair_prompt,
    build_action_observation_mask_prompt_bigtom,
    build_action_observation_mask_repair_prompt_bigtom,
    build_step_repair_prompt,
    build_step_repair_prompt_fantom,
    build_step_repair_prompt_bigtom,
    build_action_effect_prompt,
    build_action_effect_repair_prompt,
    build_apply_actions_to_state_prompt,
    build_apply_actions_to_state_repair_prompt,
    build_apply_actions_to_state_prompt_bigtom,
    build_apply_actions_to_state_prompt_bigtom_fix,
    build_apply_actions_to_state_prompt_fantom,
    build_apply_actions_to_state_prompt_fantom_v2,
    build_apply_actions_to_state_repair_prompt_bigtom,
    build_apply_actions_to_state_repair_prompt_bigtom_fix,
    build_apply_actions_to_state_repair_prompt_fantom,
    build_apply_actions_to_state_repair_prompt_fantom_v2,
    build_exit_steps_repair_prompt,
    build_exit_steps_prompt,
build_action_effect_prompt_v2,
build_action_effect_repair_prompt_v2,
build_initial_participants_prompt_fantom,
build_initial_participants_repair_prompt_fantom,
build_step_repair_prompt_general,
build_participation_delta_prompt_fantom,
build_participation_step_repair_prompt_fantom,
build_communication_delta_prompt_fantom,
build_communication_step_repair_prompt_fantom,
build_question_reduction_prompt,
build_question_reduction_repair_prompt,

)
from rectom.runtime.validation import (
    apply_delta,
    apply_state_delta,
    apply_action_delta,
    apply_state_delta_fantom,
    apply_action_delta_fantom,
    group_issues_by_step,
    validate_answer_choice,
    validate_answer_choice_bigtom,
    validate_generation_result,
    validate_generation_result_fantom,
    validate_observation_mask,
    validate_state_observation_mask,
    validate_state_observation_mask_bigtom,
    validate_state_observation_mask_fantom,
    validate_action_observation_mask,
    validate_action_observation_mask_bigtom,
    validate_step_delta,
    validate_action_effect,
    validate_final_state_result,
    validate_final_state_result_bigtom,
    validate_exit_steps,
validate_action_effect_v2,
validate_initial_participants_fantom,
validate_final_state_result_fantom,
validate_final_state_result_fantom_v2,
validate_generation_result_bigtom,
validate_question_reduction,
)
from rectom.utils import (
    extract_question_characters,
    extract_question_characters_fantom,
    fill_perspective_from_observation,
    parse_fact,
)

logger = logging.getLogger(__name__)


def _parse_delta_for_validation(raw: str) -> DeltaGenerationResult:
    """Return an empty result when delta JSON is malformed.

    Every dataset validator already treats an empty step list as invalid JSON,
    allowing the existing general repair prompt to correct the response.
    """
    try:
        return parse_delta_result(raw)
    except Exception:
        return DeltaGenerationResult(characters=[], steps=[])


def _parse_question_reduction_for_validation(raw: str) -> QuestionReductionResult:
    """Convert malformed reduction JSON into a repairable empty result."""
    try:
        return parse_question_reduction_result(raw)
    except Exception:
        return QuestionReductionResult(
            original_question="",
            character_chain=[],
            zero_order_question="",
        )


def _parse_and_validate_answer(raw: str, choices: str, validator):
    """Turn malformed answer JSON into a repairable validation failure."""
    try:
        parsed = parse_answer_result(raw)
    except Exception as exc:
        parsed = {
            "predicted_answer": raw.strip(),
            "reasoning_summary": "",
        }
        return parsed, ValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(
                    issue_type="invalid_answer_json",
                    message="The model answer could not be parsed as the required JSON.",
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            ],
        )
    return parsed, validator(parsed["predicted_answer"], choices)


def _restore_delta_source_alignment(
    sample: StorySample,
    result: DeltaGenerationResult,
) -> DeltaGenerationResult:
    """Restore immutable indices/text after an LLM repair response.

    The LLM generates semantic fact deltas. Step indices and source sentences
    already come from the dataset and must remain byte-for-byte aligned; they
    are not semantic output fields. Count mismatches remain untouched so the
    validator can reject them.
    """
    if len(result.steps) != len(sample.parsed_steps):
        return result
    for index, (step_delta, source_text) in enumerate(
        zip(result.steps, sample.parsed_steps), start=1
    ):
        step_delta.step_index = index
        step_delta.step_text = source_text
    return result


def _complete_presence_alias_removals(
    result: DeltaGenerationResult,
) -> DeltaGenerationResult:
    """Remove every stored alias of a room-presence fact together.

    Hi-ToM prompts deliberately allow both ``at(X,room)`` and
    ``in_room(X,room)`` and commonly add both on entry.  They denote the same
    presence relation.  If a generated exit delta removes only one spelling,
    leaving the other behind would make the character incorrectly observe all
    future states.  Complete only aliases that are actually present in the
    preceding state; all other delta content remains unchanged and is still
    checked by the normal validator.
    """
    previous_state: Set[str] = set()
    for step_delta in result.steps:
        completed = list(step_delta.removed_facts)
        seen = set(completed)
        for fact in list(step_delta.removed_facts):
            predicate, args = parse_fact(fact)
            if predicate not in {"at", "in_room"} or len(args) != 2:
                continue
            character, location = args
            for alias in (
                f"at({character},{location})",
                f"in_room({character},{location})",
            ):
                if alias in previous_state and alias not in seen:
                    completed.append(alias)
                    seen.add(alias)
        step_delta.removed_facts = completed
        previous_state = apply_delta(previous_state, step_delta)
    return result


def _deterministic_state_visibility(
    character: str,
    source_states: Sequence[Sequence[str]],
) -> List[bool]:
    """Exact symbolic fallback for state-presence observability.

    This mirrors state-mask validation: a character observes the state iff an
    `at` or `in_room` fact places them somewhere other than `waiting_room`.
    It is used only after the model and its repair attempts fail validation.
    """
    visibility: List[bool] = []
    for state in source_states:
        observable = False
        for fact in state:
            predicate, args = parse_fact(fact)
            if (
                predicate in {"at", "in_room"}
                and len(args) == 2
                and args[0] == character
                and args[1] != "waiting_room"
            ):
                observable = True
                break
        visibility.append(observable)
    return visibility


def _deterministic_hitom_action_visibility(
    character: str,
    source_actions: Sequence[Sequence[str]],
) -> List[bool]:
    """Exact Hi-ToM fallback matching action-mask validation rules."""
    visibility: List[bool] = []
    for actions in source_actions:
        observable = False
        for action in actions:
            predicate, args = parse_fact(action)
            if predicate == "private_tell" and len(args) >= 2:
                observable = character in {args[0], args[1]}
            elif predicate == "public_claim":
                observable = True
            if observable:
                break
        visibility.append(observable)
    return visibility


def _is_room_transition_step(step_text: str) -> bool:
    """Return whether a Hi-ToM step is an explicit room entry or exit."""
    lowered = step_text.lower()
    return bool(re.search(r"\b(?:entered|exited)\b", lowered))


def _build_universal_room_transition_deltas(
    sample: StorySample,
) -> List[tuple[List[str], List[str]]]:
    """Build aligned public room-transition deltas for Hi-ToM completion.

    Hi-ToM makes room entry/exit events observable to every character even
    when the paired state is hidden.  The accumulated state track already
    represents every other persistent event, so only these public exceptions
    must accompany an unobserved state during Equation (2) completion.
    """
    if len(sample.deltas) != len(sample.parsed_steps):
        raise ValueError(
            "Delta/state alignment failure while building room transitions: "
            f"{len(sample.deltas)} deltas for {len(sample.parsed_steps)} steps"
        )

    aligned: List[tuple[List[str], List[str]]] = []
    for step_text, delta in zip(sample.parsed_steps, sample.deltas):
        if not _is_room_transition_step(step_text):
            aligned.append(([], []))
            continue

        added = []
        removed = []
        for fact in delta.get("added_facts", []):
            predicate, _args = parse_fact(fact)
            if predicate in {"at", "in_room", "present", "not_present"}:
                added.append(fact)
        for fact in delta.get("removed_facts", []):
            predicate, _args = parse_fact(fact)
            if predicate in {"at", "in_room", "present", "not_present"}:
                removed.append(fact)
        aligned.append((added, removed))
    return aligned


def _extract_exit_steps_from_persistent_deltas(sample: StorySample) -> Dict[str, int]:
    """Read exit order deterministically from the already validated deltas."""
    exit_steps: Dict[str, int] = {}
    known_characters = set(sample.characters)
    for step_index, (step_text, delta) in enumerate(
        zip(sample.parsed_steps, sample.deltas), start=1
    ):
        if not re.search(r"\bexited\b", step_text.lower()):
            continue
        for fact in delta.get("removed_facts", []):
            predicate, args = parse_fact(fact)
            if predicate not in {"at", "in_room", "present"} or not args:
                continue
            character = args[0]
            if character in known_characters:
                exit_steps[character] = step_index

        # A validated room-exit delta should normally identify the character.
        # Retain an exact-name textual fallback so a synonymous state schema
        # (for example only adding not_present(X)) cannot silently erase the
        # benchmark's deterministic listener/trust ordering.
        for character in known_characters:
            if re.search(rf"\b{re.escape(character)}\b", step_text, flags=re.IGNORECASE):
                exit_steps.setdefault(character, step_index)
    return exit_steps


def _action_is_effective_for_character(
    action: str,
    character: str,
    exit_steps: Dict[str, int],
) -> bool:
    """Apply the deterministic Hi-ToM listener/trust rule."""
    predicate, args = parse_fact(action)
    if predicate == "private_tell" and len(args) >= 2:
        speaker, listener = args[0], args[1]
        is_listener = character == listener
    elif predicate == "public_claim" and len(args) >= 1:
        speaker = args[0]
        is_listener = character != speaker
    else:
        return False

    if not is_listener or character not in exit_steps or speaker not in exit_steps:
        return False
    return exit_steps[character] < exit_steps[speaker]


def _reduce_question_with_llm(
    llm,
    sample: StorySample,
    max_repair_rounds: int,
) -> str:
    """Implement Algorithm 1 line 12 with validated LLM reduction."""
    if not sample.question_characters:
        return sample.question.strip()

    prompt = build_question_reduction_prompt(
        sample.question,
        sample.question_characters,
    )
    raw = llm.generate(prompt, max_tokens=2048, temperature=0.0)
    result = _parse_question_reduction_for_validation(raw)
    validation = validate_question_reduction(
        sample.question,
        sample.question_characters,
        result,
    )

    repair_round = 0
    while not validation.is_valid and repair_round < max_repair_rounds:
        repair_round += 1
        repair_prompt = build_question_reduction_repair_prompt(
            question=sample.question,
            question_characters=sample.question_characters,
            broken_result=raw,
            issues=validation.issues,
        )
        raw = llm.generate(repair_prompt, max_tokens=2048, temperature=0.0)
        result = _parse_question_reduction_for_validation(raw)
        validation = validate_question_reduction(
            sample.question,
            sample.question_characters,
            result,
        )

    if not validation.is_valid:
        issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
        raise RuntimeError(f"Question reduction failed: {issue_messages}")
    return result.zero_order_question


class SequentialHiToMEngine:
    """
    A debugging-friendly engine that keeps the control flow sequential and explicit.

    Pipeline:
    1. Generate step-wise DELTAS.
    2. Validate and repair DELTAS if needed.
    3. Compute full world states in code.
    4. Extract question characters in textual order.
    5. For each question character, ask the LLM for an observation mask.
    6. Build observation states and completed perspectives in code.
    7. Use the final full state / final nested perspective to answer the question.
    """

    def __init__(self, llm_client, debug: bool = False, max_repair_rounds: int = 2):
        self.llm = llm_client
        self.debug = debug
        self.max_repair_rounds = max_repair_rounds

    def process_sample(self, sample: StorySample) -> StorySample:
        try:
            delta_result = self._build_validated_deltas(sample)
            sample.characters = delta_result.characters
            sample.deltas = [
                {
                    "step_index": x.step_index,
                    "step_text": x.step_text,
                    "added_facts": x.added_facts,
                    "removed_facts": x.removed_facts,
                }
                for x in delta_result.steps
            ]

            sample.full_states = self._compute_full_states(delta_result)

            sample.only_states = self._compute_only_states(delta_result)


            sample.only_actions = self._compute_only_actions(delta_result)



            sample.question_characters = extract_question_characters(sample.question, sample.characters)
            # final_states = self._build_nested_perspectives(sample)

            constructed_states=self._build_nested_state_perspectives(sample)

            constructed_actions=self._build_nested_action_observations(sample)


            if self._has_any_actions(constructed_actions):
                characters_sequence = _extract_exit_steps_from_persistent_deltas(sample)
                # applicable_actions=self._build_applicable_actions(constructed_actions,sample)
                applicable_actions=self._build_applicable_actions(constructed_actions,sample,characters_sequence)
                if  self._has_any_actions(applicable_actions):
                    final_states=self._build_validated_final_state_from_actions(constructed_states[-1],applicable_actions[-1])
                else:
                    final_states=constructed_states[-1]
            else:
                final_states=constructed_states[-1]

            sample.prediction, sample.reasoning_summary = self._answer_question(sample, final_states if final_states else [])
            sample.is_correct = sample.prediction == sample.answer
        except Exception as exc:
            logger.exception("Failed to process sample_id=%s", sample.sample_id)
            sample.errors.append(str(exc))
            sample.is_correct = False
        return sample

    def _has_any_actions(self,actions: List[List[str]]) -> bool:
        for step_actions in actions:
            if step_actions:
                return True
        return False

    def _build_validated_deltas(self, sample: StorySample) -> DeltaGenerationResult:
        raw = self.llm.generate(build_delta_prompt(sample), max_tokens=8192, temperature=0.0)
        result = _parse_delta_for_validation(raw)
        result = _restore_delta_source_alignment(sample, result)
        result = _complete_presence_alias_removals(result)
        validation = validate_generation_result(sample, result)
        generation_history: List[str] = [raw]

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Delta repair round %s for sample_id=%s", repair_round, sample.sample_id)
            result = self._repair_invalid_steps(
                result,
                validation,
                sample,
                generation_history,
            )
            result = _complete_presence_alias_removals(result)
            validation = validate_generation_result(sample, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Delta validation failed after repair rounds: {issue_messages}")

        return result

    def _repair_invalid_steps(
        self,
        result: DeltaGenerationResult,
        validation,
        sample: StorySample,
        generation_history: List[str],
    ) -> DeltaGenerationResult:
        grouped = group_issues_by_step(validation.issues)
        if not grouped:

            repair_prompt = build_step_repair_prompt_general(
                story_steps=sample.parsed_steps,
                broken_step=generation_history[-1],
                issues=validation.issues,
            )

            repaired_raw = self.llm.generate(
                repair_prompt,
                max_tokens=8192,
                temperature=0.0,
            )
            generation_history.append(repaired_raw)
            result = _parse_delta_for_validation(repaired_raw)
            result = _restore_delta_source_alignment(sample, result)

            return result

        previous_state: Set[str] = set()
        repaired_steps = []
        for idx, step_delta in enumerate(result.steps, start=1):
            issues = grouped.get(idx, [])
            if issues:
                expected_step = StepDelta(
                    step_index=idx,
                    step_text=sample.parsed_steps[idx - 1],
                    added_facts=list(step_delta.added_facts),
                    removed_facts=list(step_delta.removed_facts),
                )
                current_state = apply_delta(previous_state, step_delta)
                current_delta_sequence = (
                    list(repaired_steps)
                    + [expected_step]
                    + list(result.steps[idx:])
                )
                prompt = build_step_repair_prompt(
                    story_steps=sample.parsed_steps,
                    all_step_deltas=current_delta_sequence,
                    previous_model_outputs=generation_history,
                    previous_state=previous_state,
                    broken_step=expected_step,
                    current_computed_state=current_state,
                    issues=issues,
                )
                raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
                generation_history.append(raw)
                repaired = parse_single_step_delta(raw)
                repaired.step_index = idx
                repaired.step_text = sample.parsed_steps[idx - 1]
                step_delta = repaired
                if self.debug:
                    logger.info("Repaired delta for step %s", idx)
            repaired_steps.append(step_delta)
            previous_state = apply_delta(previous_state, step_delta)

        result.steps = repaired_steps
        return result

    def _compute_full_states(self, result: DeltaGenerationResult) -> List[List[str]]:
        states: List[List[str]] = []
        previous_state: Set[str] = set()
        for step_delta in result.steps:
            previous_state = apply_delta(previous_state, step_delta)
            states.append(normalize_fact_list(previous_state))
        return states



    def _compute_only_states(self, result: DeltaGenerationResult) -> List[List[str]]:
        only_states: List[List[str]] = []
        previous_only_state: Set[str] = set()
        for step_delta in result.steps:
            previous_only_state = apply_state_delta(previous_only_state, step_delta)
            only_states.append(normalize_fact_list(previous_only_state))
        return only_states


    def _compute_only_actions(self, result: DeltaGenerationResult) -> List[List[str]]:
        actions: List[List[str]] = []
        for step_delta in result.steps:
            previous_actions: Set[str] = set()
            previous_actions = apply_action_delta(previous_actions, step_delta)
            actions.append(normalize_action_list(previous_actions))
        return actions


    def _build_nested_perspectives(self, sample: StorySample) -> List[List[str]]:
        source_states = [list(x) for x in sample.full_states]
        source_label = "world_state"
        room_transition_deltas = _build_universal_room_transition_deltas(sample)

        if not sample.question_characters:
            return source_states

        for character in sample.question_characters:
            visible = self._build_validated_observation_mask(character, source_label, source_states, sample.parsed_steps)
            observation_states = [list(state) if flag else [] for state, flag in zip(source_states, visible)]
            perspective_states = fill_perspective_from_observation(
                observation_states,
                room_transition_deltas,
            )

            sample.observation_masks[character] = list(visible)
            sample.observations[character] = [list(x) for x in observation_states]
            sample.perspectives[character] = [list(x) for x in perspective_states]

            source_states = perspective_states
            source_label = f"perspective_of({character})"

        return source_states


    def _build_nested_state_perspectives(self, sample: StorySample) -> List[List[str]]:
        source_states = [list(x) for x in sample.only_states]
        source_label = "world_state"
        room_transition_deltas = _build_universal_room_transition_deltas(sample)

        if not sample.question_characters:
            return source_states

        for character in sample.question_characters:
            visible = self._build_validated_state_observation_mask(character, source_label, source_states, sample.parsed_steps)
            observation_states = [list(state) if flag else [] for state, flag in zip(source_states, visible)]
            perspective_states = fill_perspective_from_observation(
                observation_states,
                room_transition_deltas,
            )

            sample.observation_masks[character] = list(visible)
            sample.observations[character] = [list(x) for x in observation_states]
            sample.perspectives[character] = [list(x) for x in perspective_states]

            source_states = perspective_states
            source_label = f"perspective_of({character})"

        return source_states


    def _build_nested_action_observations(self, sample: StorySample) -> List[List[str]]:
        source_actions = [list(x) for x in sample.only_actions]
        source_label = "world_state"

        if not sample.question_characters:

            return [[] for _ in source_actions]

        for character in sample.question_characters:
            observation_actions = self._build_validated_action_observation_mask(character, source_label, source_actions, sample.parsed_steps)
            observation_actions = [list(action) if flag else [] for action, flag in zip(source_actions, observation_actions)]
            source_actions = observation_actions
            if all(not step_actions for step_actions in source_actions):
                break

        return source_actions


    # def _build_applicable_actions(self, source_actions,sample) -> List[List[str]]:
    #     source_label = "world_state"
    #
    #     applicable_results = self._build_validated_applicable_actions(sample.question_characters[-1], source_label, sample.parsed_steps, source_actions)
    #     source_actions = [list(action) if flag else [] for action, flag in zip(source_actions, applicable_results)]
    #
    #     result = []
    #     accumulated = []
    #     seen_any_action = False
    #
    #     for step_actions in source_actions:
    #         current = [x for x in step_actions if x]
    #
    #         if current:
    #             seen_any_action = True
    #             accumulated.extend(current)
    #             result.append(list(accumulated))
    #         else:
    #             if seen_any_action:
    #                 result.append(list(accumulated))
    #             else:
    #                 result.append([])
    #
    #
    #     return result


    def _build_applicable_actions(self, source_actions,sample,known_exit_steps,) -> List[List[str]]:
        current_character = sample.question_characters[-1]
        source_actions = [
            [
                action
                for action in step_actions
                if _action_is_effective_for_character(
                    action,
                    current_character,
                    known_exit_steps,
                )
            ]
            for step_actions in source_actions
        ]

        result = []
        accumulated = []
        seen_any_action = False

        for step_actions in source_actions:
            current = [x for x in step_actions if x]

            if current:
                seen_any_action = True
                accumulated.extend(current)
                result.append(list(accumulated))
            else:
                if seen_any_action:
                    result.append(list(accumulated))
                else:
                    result.append([])


        return result


    def _build_validated_observation_mask(
        self,
        character: str,
        source_label: str,
        source_states: Sequence[Sequence[str]],
        story_steps: Sequence[str],
    ) -> List[bool]:
        prompt = build_observation_mask_prompt(character, source_label, source_states, story_steps)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_observation_mask(raw)
        validation = validate_observation_mask(character, source_states, story_steps, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Observation repair round %s for character=%s", repair_round, character)
            # repair_prompt = build_observation_mask_repair_prompt(
            #     viewer_character=character,
            #     source_label=source_label,
            #     story_steps=story_steps,
            #     source_states=source_states,
            #     broken_visible=result.visible,
            #     issues=validation.issues,
            # )
            repair_prompt = build_observation_mask_repair_prompt(
                viewer_character=character,
                source_label=source_label,
                story_steps=story_steps,
                source_states=source_states,
                broken_visible=result.observation_basis,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_observation_mask(raw)
            validation = validate_observation_mask(character, source_states, story_steps, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            logger.warning(
                "State observation mask validation failed for %s; using exact "
                "symbolic presence fallback: %s",
                character,
                issue_messages,
            )
            return _deterministic_state_visibility(character, source_states)

        visible = [basis != "not_observable" for basis in result.observation_basis]

        return visible

        # return result.visible


    def _build_validated_state_observation_mask(
        self,
        character: str,
        source_label: str,
        source_states: Sequence[Sequence[str]],
        story_steps: Sequence[str],
    ) -> List[bool]:
        prompt = build_state_observation_mask_prompt(character, source_label, source_states, story_steps)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_observation_mask(raw)
        print(f"observation_mask: {raw}")
        validation = validate_state_observation_mask(character, source_states, story_steps, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Observation repair round %s for character=%s", repair_round, character)

            repair_prompt = build_state_observation_mask_repair_prompt(
                viewer_character=character,
                source_label=source_label,
                story_steps=story_steps,
                source_states=source_states,
                broken_visible=result.observation_basis,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_observation_mask(raw)
            validation = validate_state_observation_mask(character, source_states, story_steps, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            logger.warning(
                "State observation mask validation failed for %s; using exact "
                "symbolic presence fallback: %s",
                character,
                issue_messages,
            )
            return _deterministic_state_visibility(character, source_states)

        visible = [basis != "not_observable" for basis in result.observation_basis]

        return visible

    def _build_validated_action_observation_mask(
        self,
        character: str,
        source_label: str,
        source_actions: Sequence[Sequence[str]],
        story_steps: Sequence[str],
    ) -> List[bool]:
        prompt = build_action_observation_mask_prompt(character, source_label, source_actions, story_steps)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_observation_mask(raw)
        validation = validate_action_observation_mask(character, source_actions, story_steps, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Observation repair round %s for character=%s", repair_round, character)
            repair_prompt = build_action_observation_mask_repair_prompt(
                viewer_character=character,
                source_label=source_label,
                story_steps=story_steps,
                source_actions=source_actions,
                broken_observation_basis=result.observation_basis,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_observation_mask(raw)
            validation = validate_action_observation_mask(character, source_actions, story_steps, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            logger.warning(
                "Hi-ToM action observation mask validation failed for %s; using "
                "exact private/public visibility fallback: %s",
                character,
                issue_messages,
            )
            return _deterministic_hitom_action_visibility(character, source_actions)

        visible = [basis != "not_observable" for basis in result.observation_basis]

        return visible

    # def _build_validated_applicable_actions(
    #         self,
    #         current_character: str,
    #         source_label: str,
    #         story_steps: Sequence[str],
    #         source_actions: Sequence[Sequence[str]],
    # ) -> List[bool]:
    #     prompt = build_action_effect_prompt(
    #         current_character=current_character,
    #         source_label=source_label,
    #         story_steps=story_steps,
    #         source_actions=source_actions,
    #     )
    #     raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
    #     result = parse_action_effect(raw)
    #
    #     validation = validate_action_effect(
    #         current_character=current_character,
    #         story_steps=story_steps,
    #         source_actions=source_actions,
    #         result=result,
    #     )
    #
    #     repair_round = 0
    #     while (not validation.is_valid) and repair_round < self.max_repair_rounds:
    #         repair_round += 1
    #         if self.debug:
    #             logger.info(
    #                 "Action effect repair round %s for character=%s",
    #                 repair_round,
    #                 current_character,
    #             )
    #
    #         repair_prompt = build_action_effect_repair_prompt(
    #             current_character=current_character,
    #             source_label=source_label,
    #             story_steps=story_steps,
    #             source_actions=source_actions,
    #             broken_exit_steps=result.exit_steps,
    #             broken_effective_actions=result.effective_actions,
    #             issues=validation.issues,
    #         )
    #
    #         raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
    #         result = parse_action_effect(raw)
    #
    #         validation = validate_action_effect(
    #             current_character=current_character,
    #             story_steps=story_steps,
    #             source_actions=source_actions,
    #             result=result,
    #         )
    #
    #     if not validation.is_valid:
    #         issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
    #         raise RuntimeError(
    #             f"Effective action validation failed for {current_character}: {issue_messages}"
    #         )
    #
    #     applicable = [basis != "not_effective" for basis in result.effective_actions]
    #
    #
    #     return applicable

    def _build_validated_exit_steps(
            self,sample
    ) -> dict[str, int]:
        prompt = build_exit_steps_prompt(
            all_characters=sample.characters,
            story_steps=sample.parsed_steps,
        )
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_exit_steps(raw)

        validation = validate_exit_steps(
            all_characters=sample.characters,
            story_steps=sample.parsed_steps,
            result=result,
        )

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Exit-steps repair round %s", repair_round)

            repair_prompt = build_exit_steps_repair_prompt(
                all_characters=sample.characters,
                story_steps=sample.parsed_steps,
                broken_exit_steps=result.exit_steps,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_exit_steps(raw)

            validation = validate_exit_steps(
                all_characters=sample.characters,
                story_steps=sample.parsed_steps,
                result=result,
            )

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Exit-step validation failed: {issue_messages}")

        return result.exit_steps

    def _build_validated_applicable_actions(
            self,
            current_character: str,
            source_label: str,  # kept for compatibility, not used
            story_steps: Sequence[str],
            source_actions: Sequence[Sequence[str]],
            known_exit_steps: dict[str, int],
            all_characters: Sequence[str],
    ) -> List[bool]:
        prompt = build_action_effect_prompt_v2(
            current_character=current_character,
            all_characters=all_characters,
            known_exit_steps=known_exit_steps,
            source_actions=source_actions,
        )
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_action_effect(raw)

        validation = validate_action_effect_v2(
            current_character=current_character,
            story_steps=story_steps,
            source_actions=source_actions,
            known_exit_steps=known_exit_steps,
            result=result,
        )

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info(
                    "Action effect repair round %s for character=%s",
                    repair_round,
                    current_character,
                )

            repair_prompt = build_action_effect_repair_prompt_v2(
                current_character=current_character,
                all_characters=all_characters,
                known_exit_steps=known_exit_steps,
                source_actions=source_actions,
                broken_effective_actions=result.effective_actions,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_action_effect(raw)

            validation = validate_action_effect_v2(
                current_character=current_character,
                story_steps=story_steps,
                source_actions=source_actions,
                known_exit_steps=known_exit_steps,
                result=result,
            )

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(
                f"Effective action validation failed for {current_character}: {issue_messages}"
            )

        applicable = [basis != "not_effective" for basis in result.effective_actions]
        return applicable


    def _build_validated_final_state_from_actions(
            self,initial_state: Sequence[str],applicable_actions: Sequence[str]
    ) -> List[str]:
        prompt = build_apply_actions_to_state_prompt(initial_state=initial_state,applicable_actions=applicable_actions)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_final_state_result(raw)

        validation = validate_final_state_result(initial_state,applicable_actions, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Final-state repair round %s", repair_round)

            repair_prompt = build_apply_actions_to_state_repair_prompt(
                initial_state=initial_state,
                applicable_actions=applicable_actions,
                broken_final_state=result.final_state,
                issues=validation.issues,
            )
            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_final_state_result(raw)
            validation = validate_final_state_result(initial_state,applicable_actions,result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Final state validation failed: {issue_messages}")

        return result.final_state


    def normalize_question_to_object_query(
            self,
            question: str,
            question_characters: Sequence[str] | None = None,
    ) -> str:
        q = question.strip()

        # No known nested characters -> keep original question
        if not question_characters:
            return q

        # First try a stricter pattern using the provided character chain.
        escaped_chars = [re.escape(ch) for ch in question_characters]
        belief_chain = r"\s+".join(fr"{ch}\s+think[s]?" for ch in escaped_chars)
        strict_pattern = rf"^Where\s+do(?:es)?\s+{belief_chain}\s+the\s+(.+?)\s+is\?$"
        m = re.match(strict_pattern, q, flags=re.IGNORECASE)
        if m:
            obj = m.group(1).strip()
            return f"Where is the {obj}?"

        # Fallback 1:
        # allow any number of "... think(s) ..." segments before "the X is?"
        loose_pattern = r"^Where\s+do(?:es)?\s+.+?\s+the\s+(.+?)\s+is\?$"
        m = re.match(loose_pattern, q, flags=re.IGNORECASE)
        if m:
            obj = m.group(1).strip()
            return f"Where is the {obj}?"

        # Fallback 2:
        # If we can at least find the trailing "the X is?" part, normalize from that.
        m = re.search(r"\bthe\s+(.+?)\s+is\?$", q, flags=re.IGNORECASE)
        if m:
            obj = m.group(1).strip()
            return f"Where is the {obj}?"

        return q

    def _answer_question(self, sample: StorySample, final_facts: List[str]) -> tuple[str, str]:

        normalize_question = _reduce_question_with_llm(
            self.llm,
            sample,
            self.max_repair_rounds,
        )

        prompt = build_answer_prompt(
            question=normalize_question,
            choices=sample.choices,
            final_facts=final_facts,
            question_characters=sample.question_characters,
        )
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        parsed, validation = _parse_and_validate_answer(
            raw, sample.choices, validate_answer_choice
        )

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Final answer repair round %s", repair_round)

            repair_prompt = build_answer_repair_prompt(
                question=normalize_question,
                choices=sample.choices,
                final_facts=final_facts,
                question_characters=sample.question_characters,
                broken_answer=parsed["predicted_answer"],
            )
            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            parsed, validation = _parse_and_validate_answer(
                raw, sample.choices, validate_answer_choice
            )

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Final answer validation failed: {issue_messages}")

        return parsed["predicted_answer"], parsed.get("reasoning_summary", "")





class SequentialFanToMEngine:
    """
    A debugging-friendly engine that keeps the control flow sequential and explicit.

    Pipeline:
    1. Generate step-wise DELTAS.
    2. Validate and repair DELTAS if needed.
    3. Compute full world states in code.
    4. Extract question characters in textual order.
    5. For each question character, ask the LLM for an observation mask.
    6. Build observation states and completed perspectives in code.
    7. Use the final full state / final nested perspective to answer the question.
    """

    def __init__(self, llm_client, debug: bool = False, max_repair_rounds: int = 2):
        self.llm = llm_client
        self.debug = debug
        self.max_repair_rounds = max_repair_rounds

    def _merge_delta_results(self, participation_delta, communication_delta):
        # Merge characters
        merged_characters = []
        seen = set()
        for name in participation_delta.characters + communication_delta.characters:
            if name not in seen:
                seen.add(name)
                merged_characters.append(name)

        # Sanity check
        if len(participation_delta.steps) != len(communication_delta.steps):
            raise ValueError(
                f"Step count mismatch: participation={len(participation_delta.steps)} "
                f"communication={len(communication_delta.steps)}"
            )

        merged_steps = []
        for p_step, c_step in zip(participation_delta.steps, communication_delta.steps):
            if p_step.step_index != c_step.step_index:
                print(f"p_step.step_index: {p_step.step_index}; c_step.step_index: {c_step.step_index}")
                raise ValueError(
                    f"Step index mismatch: participation={p_step.step_index}, "
                    f"communication={c_step.step_index}"
                )

            # if p_step.step_text != c_step.step_text:
            #     raise ValueError(
            #         f"Step text mismatch at step {p_step.step_index}"
            #     )

            added_facts = []
            removed_facts = []

            # Keep both aligned tracks.  The previous implementation returned
            # early whenever a participation delta existed and silently
            # discarded a communication from the same step.
            seen_added = set()
            for fact in p_step.added_facts + c_step.added_facts:
                if fact not in seen_added:
                    seen_added.add(fact)
                    added_facts.append(fact)

            seen_removed = set()
            # Communication events are transient and therefore cannot remove
            # facts from the global participation state.  Only participation
            # deltas contribute removals here.
            for fact in p_step.removed_facts:
                if fact not in seen_removed:
                    seen_removed.add(fact)
                    removed_facts.append(fact)

            merged_steps.append(
                StepDelta(
                    step_index=int(p_step.step_index),
                    step_text=str(p_step.step_text),
                    added_facts=list(added_facts),
                    removed_facts=list(removed_facts),
                )
            )
        return  DeltaGenerationResult(characters=merged_characters, steps=merged_steps)


    def process_sample(self, sample: StorySample) -> StorySample:
        sample.initial_participants=self._build_validated_initial_participants(sample)
        sample.parsed_steps=[self._build_initial_participants_step(initial_participants=sample.initial_participants)]+sample.parsed_steps
        try:
            # delta_result = self._build_validated_deltas(sample)
            participation_delta = self._build_validated_participation_deltas(sample)
            communication_delta = self._build_validated_communication_deltas(sample)
            # sample.characters = delta_result.characters
            delta_result = self._merge_delta_results(participation_delta, communication_delta)

            sample.characters = delta_result.characters
            sample.deltas = [
                {
                    "step_index": x.step_index,
                    "step_text": x.step_text,
                    "added_facts": x.added_facts,
                    "removed_facts": x.removed_facts,
                }
                for x in delta_result.steps
            ]

            sample.full_states = self._compute_full_states(delta_result)

            sample.only_states = self._compute_only_states(delta_result)


            sample.only_actions = self._compute_only_actions(delta_result)


            if "first-order" in sample.question_order:
                expected_num_characters=1
            else:
                expected_num_characters=2


            sample.question_characters = extract_question_characters_fantom(sample.question, sample.characters,expected_num_characters)

            constructed_states=self._build_nested_state_perspectives(sample)

            constructed_actions=self._build_nested_action_observations(sample)
            # constructed_actions,invisible_actions=self._build_nested_all_actions(sample)


            if self._has_any_actions(constructed_actions):
                final_states = self._build_validated_final_state_from_actions(
                    constructed_states[-1],
                    constructed_actions,
                )
                # final_conversations=[sample.parsed_steps[i] for i in range (len(sample.parsed_steps)) if constructed_actions[i]]
                # final_invisible_conversations=[sample.parsed_steps[i] for i in range (len(sample.parsed_steps)) if not constructed_actions[i]]
            else:
                final_states=constructed_states[-1]


            sample.prediction, sample.reasoning_summary = self._answer_question(sample, final_states if final_states else [])
            # sample.prediction, sample.reasoning_summary = self._answer_question_from_conversations(sample, final_conversations if final_conversations else [])
            # sample.prediction, sample.reasoning_summary = self._answer_question_from_conversations_with_invisible_conversations(sample,final_conversations if final_conversations else [],final_invisible_conversations if final_invisible_conversations else [])
            # sample.prediction, sample.reasoning_summary = self._answer_question_with_invisible_facts(sample, constructed_actions if constructed_actions else [],invisible_actions)
            sample.is_correct = sample.prediction == sample.answer
        except Exception as exc:
            logger.exception("Failed to process sample_id=%s", sample.sample_id)
            sample.errors.append(str(exc))
            sample.is_correct = False
        return sample

    def _build_initial_participants_step(self, initial_participants: Sequence[str]) -> str:
        names = list(initial_participants)
        if not names:
            return "Initially, no one was present in the conversation."

        if len(names) == 1:
            return f"Initially, {names[0]} was present in the conversation."
        if len(names) == 2:
            return f"Initially, {names[0]} and {names[1]} were present in the conversation."

        joined = ", ".join(names[:-1]) + f", and {names[-1]}"
        return f"Initially, {joined} were present in the conversation."


    def _has_any_actions(self,actions: List[List[str]]) -> bool:
        for step_actions in actions:
            if step_actions:
                return True
        return False

    def _build_validated_deltas(self, sample: StorySample) -> DeltaGenerationResult:
        prompt = build_delta_prompt_fantom(sample)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = _parse_delta_for_validation(raw)
        result = _restore_delta_source_alignment(sample, result)
        validation = validate_generation_result_fantom(sample, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Delta repair round %s for sample_id=%s", repair_round, sample.sample_id)

            repair_prompt = build_step_repair_prompt_fantom(
                story_steps=sample.parsed_steps,
                broken_step=raw,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = _parse_delta_for_validation(raw)
            result = _restore_delta_source_alignment(sample, result)

            validation = validate_generation_result_fantom(sample, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Delta validation failed after repair rounds: {issue_messages}")

        return result


    def _build_validated_participation_deltas(self, sample: StorySample) -> DeltaGenerationResult:
        prompt = build_participation_delta_prompt_fantom(sample)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = _parse_delta_for_validation(raw)
        result = _restore_delta_source_alignment(sample, result)
        validation = validate_generation_result_fantom(sample, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Delta repair round %s for sample_id=%s", repair_round, sample.sample_id)

            repair_prompt = build_participation_step_repair_prompt_fantom(
                story_steps=sample.parsed_steps,
                broken_step=raw,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = _parse_delta_for_validation(raw)
            result = _restore_delta_source_alignment(sample, result)

            validation = validate_generation_result_fantom(sample, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Delta validation failed after repair rounds: {issue_messages}")

        return result


    def _build_validated_communication_deltas(self, sample: StorySample) -> DeltaGenerationResult:
        prompt = build_communication_delta_prompt_fantom(sample)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = _parse_delta_for_validation(raw)
        result = _restore_delta_source_alignment(sample, result)
        validation = validate_generation_result_fantom(sample, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Delta repair round %s for sample_id=%s", repair_round, sample.sample_id)

            repair_prompt = build_communication_step_repair_prompt_fantom(
                story_steps=sample.parsed_steps,
                broken_step=raw,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = _parse_delta_for_validation(raw)
            result = _restore_delta_source_alignment(sample, result)

            validation = validate_generation_result_fantom(sample, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Delta validation failed after repair rounds: {issue_messages}")

        return result



    def _compute_full_states(self, result: DeltaGenerationResult) -> List[List[str]]:
        states: List[List[str]] = []
        previous_state: Set[str] = set()
        for step_delta in result.steps:
            previous_state = apply_delta(previous_state, step_delta)
            states.append(normalize_fact_list(previous_state))
        return states


    def _compute_only_states(self, result: DeltaGenerationResult) -> List[List[str]]:
        only_states: List[List[str]] = []
        previous_only_state: Set[str] = set()
        for step_delta in result.steps:
            previous_only_state = apply_state_delta_fantom(previous_only_state, step_delta)
            only_states.append(normalize_fact_list(previous_only_state))
        return only_states


    def _compute_only_actions(self, result: DeltaGenerationResult) -> List[List[str]]:
        actions: List[List[str]] = []
        for step_delta in result.steps:
            previous_actions: Set[str] = set()
            previous_actions = apply_action_delta_fantom(previous_actions, step_delta)
            actions.append(normalize_action_list(previous_actions))
        return actions


    def _build_nested_perspectives(self, sample: StorySample) -> List[List[str]]:
        source_states = [list(x) for x in sample.full_states]
        source_label = "world_state"

        if not sample.question_characters:
            return source_states

        for character in sample.question_characters:
            visible = self._build_validated_observation_mask(character, source_label, source_states, sample.parsed_steps)
            observation_states = [list(state) if flag else [] for state, flag in zip(source_states, visible)]
            perspective_states = fill_perspective_from_observation(observation_states)

            sample.observation_masks[character] = list(visible)
            sample.observations[character] = [list(x) for x in observation_states]
            sample.perspectives[character] = [list(x) for x in perspective_states]

            source_states = perspective_states
            source_label = f"perspective_of({character})"

        return source_states


    def _build_nested_state_perspectives(self, sample: StorySample) -> List[List[str]]:
        source_states = [list(x) for x in sample.only_states]
        source_label = "world_state"

        if not sample.question_characters:
            return source_states

        for character in sample.question_characters:
            visible = self._build_validated_state_observation_mask(character, source_label, source_states, sample.parsed_steps)
            observation_states = [list(state) if flag else [] for state, flag in zip(source_states, visible)]
            perspective_states = fill_perspective_from_observation(observation_states)

            sample.observation_masks[character] = list(visible)
            sample.observations[character] = [list(x) for x in observation_states]
            sample.perspectives[character] = [list(x) for x in perspective_states]

            source_states = perspective_states
            source_label = f"perspective_of({character})"

        return source_states


    def _build_nested_action_observations(self, sample: StorySample) -> List[List[str]]:
        source_actions = [list(x) for x in sample.only_actions]
        source_label = "world_state"

        if not sample.question_characters:

            return [[] for _ in source_actions]

        for character in sample.question_characters:
            observation_actions = sample.observation_masks[character]
            observation_actions = [list(action) if flag else [] for action, flag in zip(source_actions, observation_actions)]
            source_actions = observation_actions
            if all(not step_actions for step_actions in source_actions):
                break

        return source_actions

    def _build_nested_all_actions(self, sample: StorySample) -> List[List[str]]:
        source_actions = [list(x) for x in sample.only_actions]
        source_invisible_actions = [list(x) for x in sample.only_actions]
        source_label = "world_state"

        if not sample.question_characters:

            return [[] for _ in source_actions]

        for character in sample.question_characters:
            observation_actions = sample.observation_masks[character]
            invisible_actions = sample.observation_masks[character]
            observation_actions = [list(action) if flag else [] for action, flag in zip(source_actions, observation_actions)]
            invisible_actions   = [list(action) if not flag else [] for action, flag in zip(source_invisible_actions, invisible_actions)]
            source_actions = observation_actions
            source_invisible_actions=invisible_actions
            if all(not step_actions for step_actions in source_actions):
                break

        return source_actions, source_invisible_actions


    def _build_validated_observation_mask(
        self,
        character: str,
        source_label: str,
        source_states: Sequence[Sequence[str]],
        story_steps: Sequence[str],
    ) -> List[bool]:
        prompt = build_observation_mask_prompt(character, source_label, source_states, story_steps)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_observation_mask(raw)
        validation = validate_observation_mask(character, source_states, story_steps, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Observation repair round %s for character=%s", repair_round, character)
            # repair_prompt = build_observation_mask_repair_prompt(
            #     viewer_character=character,
            #     source_label=source_label,
            #     story_steps=story_steps,
            #     source_states=source_states,
            #     broken_visible=result.visible,
            #     issues=validation.issues,
            # )
            repair_prompt = build_observation_mask_repair_prompt(
                viewer_character=character,
                source_label=source_label,
                story_steps=story_steps,
                source_states=source_states,
                broken_visible=result.observation_basis,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_observation_mask(raw)
            validation = validate_observation_mask(character, source_states, story_steps, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Observation mask validation failed for {character}: {issue_messages}")

        visible = [basis != "not_observable" for basis in result.observation_basis]

        return visible

        # return result.visible

    def _build_validated_initial_participants(
            self,
            story_steps: Sequence[str],
    ) -> List[str]:
        prompt = build_initial_participants_prompt_fantom(story_steps)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_initial_participants(raw)
        validation = validate_initial_participants_fantom(story_steps, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Initial participants repair round %s", repair_round)

            repair_prompt = build_initial_participants_repair_prompt_fantom(
                story_steps=story_steps,
                broken_characters=result.characters,
                broken_initial_present_characters=result.initial_present_characters,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_initial_participants(raw)
            validation = validate_initial_participants_fantom(story_steps, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Initial participants validation failed: {issue_messages}")

        return result.initial_present_characters




    def _build_validated_state_observation_mask(
        self,
        character: str,
        source_label: str,
        source_states: Sequence[Sequence[str]],
        story_steps: Sequence[str],
    ) -> List[bool]:
        prompt = build_state_observation_mask_prompt_fantom(character, source_label, source_states, story_steps)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_observation_mask(raw)
        validation = validate_state_observation_mask_fantom(character, source_states, story_steps, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Observation repair round %s for character=%s", repair_round, character)

            repair_prompt = build_state_observation_mask_repair_prompt_fantom(
                viewer_character=character,
                source_label=source_label,
                story_steps=story_steps,
                source_states=source_states,
                broken_visible=result.observation_basis,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_observation_mask(raw)
            validation = validate_state_observation_mask_fantom(character, source_states, story_steps, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            logger.warning(
                "FanToM state observation mask validation failed for %s; using "
                "exact symbolic presence fallback: %s",
                character,
                issue_messages,
            )
            return _deterministic_state_visibility(character, source_states)

        visible = [basis != "not_observable" for basis in result.observation_basis]

        return visible

    def _build_validated_action_observation_mask(
        self,
        character: str,
        source_label: str,
        source_actions: Sequence[Sequence[str]],
        story_steps: Sequence[str],
    ) -> List[bool]:
        prompt = build_action_observation_mask_prompt_bigtom(character, source_label, source_actions, story_steps)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_observation_mask(raw)
        validation = validate_action_observation_mask_bigtom(character, source_actions, story_steps, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Observation repair round %s for character=%s", repair_round, character)
            repair_prompt = build_action_observation_mask_repair_prompt_bigtom(
                viewer_character=character,
                source_label=source_label,
                story_steps=story_steps,
                source_actions=source_actions,
                broken_observation_basis=result.observation_basis,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_observation_mask(raw)
            validation = validate_action_observation_mask_bigtom(character, source_actions, story_steps, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Observation mask validation failed for {character}: {issue_messages}")

        visible = [basis != "not_observable" for basis in result.observation_basis]

        return visible


    def _build_actions(self,source_actions:Sequence[str]):
        result_actions = []
        accumulated = []
        seen_any_action = False

        for step_actions in source_actions:
            current = [x for x in step_actions if x]

            if current:
                seen_any_action = True
                accumulated.extend(current)
                result_actions.append(list(accumulated))
            else:
                if seen_any_action:
                    result_actions.append(list(accumulated))
                else:
                    result_actions.append([])
        return result_actions

    def _build_validated_final_state_from_actions(
            self,initial_state: Sequence[str],source_actions:Sequence[str]
    ) -> List[str]:

        source_actions=self._build_actions(source_actions)[-1]

        prompt = build_apply_actions_to_state_prompt_fantom_v2(state=initial_state,actions=source_actions)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_final_state_result_fantom_v2(raw)

        validation = validate_final_state_result_fantom_v2(initial_state,source_actions, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Final-state repair round %s", repair_round)

            repair_prompt = build_apply_actions_to_state_repair_prompt_fantom_v2(
                state=initial_state,
                actions=source_actions,
                broken_result_text=raw,
                issues=validation.issues,
            )
            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_final_state_result_fantom_v2(raw)
            validation = validate_final_state_result_fantom_v2(initial_state,source_actions,result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Final state validation failed: {issue_messages}")

        return result.state_after



    def normalize_question_to_object_query(self,question: str, character: str | None = None) -> str:
        question = question.strip()

        if character:
            if len(character)==1:
                character=character[0]
            prefix = f"Does {character} believe "
            if question.startswith(prefix):
                out = question[len(prefix):].strip()
                return out[0].lower() + out[1:] if out else out

        match = re.match(r"^Does\s+.+?\s+believe\s+(.*)$", question)
        if match:
            out = match.group(1).strip()
            return out[0].lower() + out[1:] if out else out

        return question

    def _answer_question(self, sample: StorySample, final_facts: List[str]) -> tuple[str, str]:

        normalize_question = _reduce_question_with_llm(
            self.llm,
            sample,
            self.max_repair_rounds,
        )

        prompt = build_answer_prompt_fantom(
            question=normalize_question,
            choices=sample.choices,
            final_facts=final_facts,
            question_characters=sample.question_characters,
        )
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        parsed, validation = _parse_and_validate_answer(
            raw, sample.choices, validate_answer_choice_bigtom
        )


        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Final answer repair round %s", repair_round)

            repair_prompt = build_answer_repair_prompt_fantom(
                question=normalize_question,
                choices=sample.choices,
                final_facts=final_facts,
                question_characters=sample.question_characters,
                broken_answer=parsed["predicted_answer"],
            )
            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            parsed, validation = _parse_and_validate_answer(
                raw, sample.choices, validate_answer_choice_bigtom
            )

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            print(parsed["predicted_answer"], sample.choices)
            raise RuntimeError(f"Final answer validation failed: {issue_messages}")

        return parsed["predicted_answer"], parsed.get("reasoning_summary", "")

    def _answer_question_with_invisible_facts(self, sample: StorySample, final_facts: List[str], invisible_facts: List[str],) -> tuple[str, str]:

        normalize_question = _reduce_question_with_llm(
            self.llm,
            sample,
            self.max_repair_rounds,
        )
        final_facts = [item for sublist in final_facts for item in sublist]
        invisible_facts = [item for sublist in invisible_facts for item in sublist]

        prompt = build_answer_prompt_fantom_with_invisible_facts(
            question=normalize_question,
            choices=sample.choices,
            final_facts=final_facts,
            final_invisible_facts=invisible_facts,
            question_characters=sample.question_characters,
        )
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        parsed, validation = _parse_and_validate_answer(
            raw, sample.choices, validate_answer_choice_bigtom
        )


        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Final answer repair round %s", repair_round)

            repair_prompt = build_answer_repair_prompt_fantom_with_invisible_facts(
                question=normalize_question,
                choices=sample.choices,
                final_facts=final_facts,
                final_invisible_facts=invisible_facts,
                question_characters=sample.question_characters,
                broken_answer=parsed["predicted_answer"],
            )
            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            parsed, validation = _parse_and_validate_answer(
                raw, sample.choices, validate_answer_choice_bigtom
            )

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            print(parsed["predicted_answer"], sample.choices)
            raise RuntimeError(f"Final answer validation failed: {issue_messages}")

        return parsed["predicted_answer"], parsed.get("reasoning_summary", "")

    def _answer_question_from_conversations(self, sample: StorySample, final_conversations: List[str]) -> tuple[str, str]:

        normalize_question=self.normalize_question_to_object_query(sample.question,sample.characters)

        prompt = build_answer_prompt_fantom_from_conversations(
            question=normalize_question,
            choices=sample.choices,
            conversations=final_conversations,
            question_characters=sample.question_characters,
        )
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        parsed, validation = _parse_and_validate_answer(
            raw, sample.choices, validate_answer_choice_bigtom
        )


        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Final answer repair round %s", repair_round)

            repair_prompt = build_answer_repair_prompt_fantom_from_conversations(
                question=normalize_question,
                choices=sample.choices,
                conversations=final_conversations,
                question_characters=sample.question_characters,
                broken_answer=parsed["predicted_answer"],
            )
            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            parsed, validation = _parse_and_validate_answer(
                raw, sample.choices, validate_answer_choice_bigtom
            )

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            print(parsed["predicted_answer"], sample.choices)
            raise RuntimeError(f"Final answer validation failed: {issue_messages}")

        return parsed["predicted_answer"], parsed.get("reasoning_summary", "")

    def _answer_question_from_conversations_with_invisible_conversations(self, sample: StorySample, final_conversations: List[str], final_invisible_conversations: List[str]) -> tuple[str, str]:

        normalize_question=self.normalize_question_to_object_query(sample.question,sample.characters)

        prompt = build_answer_prompt_fantom_from_conversations_with_invisible_conversations(
            question=normalize_question,
            choices=sample.choices,
            conversations=final_conversations,
            invisible_conversations=final_invisible_conversations,
            question_characters=sample.question_characters,
        )
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        parsed = parse_answer_result(raw)
        validation = validate_answer_choice_bigtom(parsed["predicted_answer"], sample.choices)


        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Final answer repair round %s", repair_round)

            repair_prompt = build_answer_repair_prompt_fantom_from_conversations_with_invisible_conversations(
                question=normalize_question,
                choices=sample.choices,
                conversations=final_conversations,
                invisible_conversations=final_invisible_conversations,
                question_characters=sample.question_characters,
                broken_answer=parsed["predicted_answer"],
            )
            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            parsed = parse_answer_result(raw)
            validation = validate_answer_choice_bigtom(parsed["predicted_answer"], sample.choices)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            print(parsed["predicted_answer"], sample.choices)
            raise RuntimeError(f"Final answer validation failed: {issue_messages}")

        return parsed["predicted_answer"], parsed.get("reasoning_summary", "")


class SequentialBigToMEngine:
    """
    A debugging-friendly engine that keeps the control flow sequential and explicit.

    Pipeline:
    1. Generate step-wise DELTAS.
    2. Validate and repair DELTAS if needed.
    3. Compute full world states in code.
    4. Extract question characters in textual order.
    5. For each question character, ask the LLM for an observation mask.
    6. Build observation states and completed perspectives in code.
    7. Use the final full state / final nested perspective to answer the question.
    """

    def __init__(self, llm_client, debug: bool = False, max_repair_rounds: int = 2):
        self.llm = llm_client
        self.debug = debug
        self.max_repair_rounds = max_repair_rounds

    def process_sample(self, sample: StorySample) -> StorySample:
        try:
            delta_result = self._build_validated_deltas(sample)
            sample.characters = delta_result.characters
            sample.deltas = [
                {
                    "step_index": x.step_index,
                    "step_text": x.step_text,
                    "added_facts": x.added_facts,
                    "removed_facts": x.removed_facts,
                }
                for x in delta_result.steps
            ]

            sample.full_states = self._compute_full_states(delta_result)

            sample.only_states = self._compute_only_states(delta_result)

            # sample.only_actions = self._compute_only_actions(delta_result,sample)

            sample.question_characters = extract_question_characters(sample.question, sample.characters)

            constructed_states=self._build_nested_state_perspectives(sample)

            # constructed_actions=self._build_nested_action_observations(sample)

            final_states=constructed_states[-1]

            # if self._has_any_actions(constructed_actions):
            #     final_states = self._build_validated_final_state_from_actions(constructed_states[-1],
            #                                                                   constructed_actions[-1][0],sample.question_characters[-1])
            # else:
            #     final_states=constructed_states[-1]


            sample.prediction, sample.reasoning_summary = self._answer_question(sample, final_states if final_states else [])
            sample.is_correct = sample.prediction == sample.answer
        except Exception as exc:
            logger.exception("Failed to process sample_id=%s", sample.sample_id)
            sample.errors.append(str(exc))
            sample.is_correct = False
        return sample

    def _has_any_actions(self,actions: List[List[str]]) -> bool:
        for step_actions in actions:
            if step_actions:
                return True
        return False

    def _build_validated_deltas(self, sample: StorySample) -> DeltaGenerationResult:
        prompt=build_delta_prompt_bigtom(sample)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = _parse_delta_for_validation(raw)
        result = _restore_delta_source_alignment(sample, result)
        validation = validate_generation_result_bigtom(sample, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Delta repair round %s for sample_id=%s", repair_round, sample.sample_id)

            repair_prompt = build_step_repair_prompt_bigtom(
                story_steps=sample.parsed_steps,
                broken_step=raw,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = _parse_delta_for_validation(raw)
            result = _restore_delta_source_alignment(sample, result)

            validation = validate_generation_result_bigtom(sample, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Delta validation failed after repair rounds: {issue_messages}")

        return result

    def _repair_invalid_steps(self, result: DeltaGenerationResult, validation) -> DeltaGenerationResult:
        grouped = group_issues_by_step(validation.issues)
        if not grouped:
            return result

        previous_state: Set[str] = set()
        repaired_steps = []
        generation_history: List[str] = []
        for idx, step_delta in enumerate(result.steps, start=1):
            issues = grouped.get(idx, [])
            if issues:
                current_state = apply_delta(previous_state, step_delta)
                current_delta_sequence = (
                    list(repaired_steps)
                    + [step_delta]
                    + list(result.steps[idx:])
                )
                prompt = build_step_repair_prompt(
                    story_steps=[step.step_text for step in result.steps],
                    all_step_deltas=current_delta_sequence,
                    previous_model_outputs=generation_history,
                    previous_state=previous_state,
                    broken_step=step_delta,
                    current_computed_state=current_state,
                    issues=issues,
                )
                raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
                generation_history.append(raw)
                repaired = parse_single_step_delta(raw)
                step_delta = repaired
                if self.debug:
                    logger.info("Repaired delta for step %s", idx)
            repaired_steps.append(step_delta)
            previous_state = apply_delta(previous_state, step_delta)

        result.steps = repaired_steps
        return result

    def _compute_full_states(self, result: DeltaGenerationResult) -> List[List[str]]:
        states: List[List[str]] = []
        previous_state: Set[str] = set()
        for step_delta in result.steps:
            previous_state = apply_delta(previous_state, step_delta)
            states.append(normalize_fact_list(previous_state))
        return states



    # def _compute_only_states(self, result: DeltaGenerationResult) -> List[List[str]]:
    #     only_states: List[List[str]] = []
    #     previous_only_state: Set[str] = set()
    #     for step_delta in result.steps[:-1]:
    #         previous_only_state = apply_state_delta(previous_only_state, step_delta)
    #         only_states.append(normalize_fact_list(previous_only_state))
    #     only_states.append([])
    #     return only_states


    def _compute_only_states(self, result: DeltaGenerationResult) -> List[List[str]]:
        only_states: List[List[str]] = []
        previous_only_state: Set[str] = set()
        for step_delta in result.steps:
            previous_only_state = apply_state_delta(previous_only_state, step_delta)
            only_states.append(normalize_fact_list(previous_only_state))
        return only_states


    def _compute_only_actions(self, result: DeltaGenerationResult,sample) -> List[List[str]]:
        actions: List[List[str]] = []
        for step_delta in result.steps[:-1]:
            previous_actions =[]
            actions.append(normalize_action_list(previous_actions))
        actions.append([sample.parsed_steps[-1]])
        return actions


    def _build_nested_perspectives(self, sample: StorySample) -> List[List[str]]:
        source_states = [list(x) for x in sample.full_states]
        source_label = "world_state"

        if not sample.question_characters:
            return source_states

        for character in sample.question_characters:
            visible = self._build_validated_observation_mask(character, source_label, source_states, sample.parsed_steps)
            observation_states = [list(state) if flag else [] for state, flag in zip(source_states, visible)]
            perspective_states = fill_perspective_from_observation(observation_states)

            sample.observation_masks[character] = list(visible)
            sample.observations[character] = [list(x) for x in observation_states]
            sample.perspectives[character] = [list(x) for x in perspective_states]

            source_states = perspective_states
            source_label = f"perspective_of({character})"

        return source_states


    def _build_nested_state_perspectives(self, sample: StorySample) -> List[List[str]]:
        source_states = [list(x) for x in sample.only_states]
        source_label = "world_state"

        if not sample.question_characters:
            return source_states

        for character in sample.question_characters:
            visible = self._build_validated_state_observation_mask(character, source_label, source_states, sample.parsed_steps)
            if visible[-1]==True:
                print(f"source_states: {source_states}")
                print(f"visible: {visible}")
            observation_states = [list(state) if flag else [] for state, flag in zip(source_states, visible)]
            perspective_states = fill_perspective_from_observation(observation_states)

            sample.observation_masks[character] = list(visible)
            sample.observations[character] = [list(x) for x in observation_states]
            sample.perspectives[character] = [list(x) for x in perspective_states]

            source_states = perspective_states
            source_label = f"perspective_of({character})"

        return source_states


    def _build_nested_action_observations(self, sample: StorySample) -> List[List[str]]:
        source_actions = [list(x) for x in sample.only_actions]
        source_label = "world_state"

        if not sample.question_characters:

            return [[] for _ in source_actions]

        for character in sample.question_characters:
            observation_actions = self._build_validated_action_observation_mask(character, source_label, source_actions, sample.parsed_steps)
            observation_actions = [list(action) if flag else [] for action, flag in zip(source_actions, observation_actions)]
            source_actions = observation_actions
            if all(not step_actions for step_actions in source_actions):
                break

        return source_actions



    def _build_validated_observation_mask(
        self,
        character: str,
        source_label: str,
        source_states: Sequence[Sequence[str]],
        story_steps: Sequence[str],
    ) -> List[bool]:
        prompt = build_observation_mask_prompt(character, source_label, source_states, story_steps)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_observation_mask(raw)
        validation = validate_observation_mask(character, source_states, story_steps, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Observation repair round %s for character=%s", repair_round, character)
            # repair_prompt = build_observation_mask_repair_prompt(
            #     viewer_character=character,
            #     source_label=source_label,
            #     story_steps=story_steps,
            #     source_states=source_states,
            #     broken_visible=result.visible,
            #     issues=validation.issues,
            # )
            repair_prompt = build_observation_mask_repair_prompt(
                viewer_character=character,
                source_label=source_label,
                story_steps=story_steps,
                source_states=source_states,
                broken_visible=result.observation_basis,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_observation_mask(raw)
            validation = validate_observation_mask(character, source_states, story_steps, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Observation mask validation failed for {character}: {issue_messages}")

        visible = [basis != "not_observable" for basis in result.observation_basis]

        return visible

        # return result.visible


    def _build_validated_state_observation_mask(
        self,
        character: str,
        source_label: str,
        source_states: Sequence[Sequence[str]],
        story_steps: Sequence[str],
    ) -> List[bool]:
        prompt = build_state_observation_mask_prompt_bigtom_fix(character, source_label, source_states, story_steps)
        # prompt = build_state_observation_mask_prompt_bigtom(character, source_label, source_states, story_steps)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_observation_mask(raw)
        validation = validate_state_observation_mask_bigtom(character, source_states, story_steps, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Observation repair round %s for character=%s", repair_round, character)

            # repair_prompt = build_state_observation_mask_repair_prompt_bigtom(
            #     viewer_character=character,
            #     source_label=source_label,
            #     story_steps=story_steps,
            #     source_states=source_states,
            #     broken_visible=result.observation_basis,
            #     issues=validation.issues,
            # )

            repair_prompt = build_state_observation_mask_repair_prompt_bigtom_fix(
                viewer_character=character,
                source_label=source_label,
                story_steps=story_steps,
                source_states=source_states,
                broken_visible=result.observation_basis,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_observation_mask(raw)
            validation = validate_state_observation_mask_bigtom(character, source_states, story_steps, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            logger.warning(
                "Big-ToM state observation mask validation failed for %s; using "
                "exact symbolic presence fallback: %s",
                character,
                issue_messages,
            )
            return _deterministic_state_visibility(character, source_states)

        visible = [basis != "not_observable" for basis in result.observation_basis]

        return visible

    def _build_validated_action_observation_mask(
        self,
        character: str,
        source_label: str,
        source_actions: Sequence[Sequence[str]],
        story_steps: Sequence[str],
    ) -> List[bool]:
        prompt = build_action_observation_mask_prompt_bigtom(character, source_label, source_actions, story_steps)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_observation_mask(raw)
        validation = validate_action_observation_mask_bigtom(character, source_actions, story_steps, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Observation repair round %s for character=%s", repair_round, character)
            repair_prompt = build_action_observation_mask_repair_prompt_bigtom(
                viewer_character=character,
                source_label=source_label,
                story_steps=story_steps,
                source_actions=source_actions,
                broken_observation_basis=result.observation_basis,
                issues=validation.issues,
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_observation_mask(raw)
            validation = validate_action_observation_mask_bigtom(character, source_actions, story_steps, result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Observation mask validation failed for {character}: {issue_messages}")

        visible = [basis != "not_observable" for basis in result.observation_basis]

        return visible


    def _build_validated_final_state_from_actions(
            self,initial_state: Sequence[str],action:str,character:str
    ) -> List[str]:
        prompt = build_apply_actions_to_state_prompt_bigtom_fix(initial_state=initial_state,action=action,character=character)
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        result = parse_final_state_result_bigtom(raw)

        validation = validate_final_state_result_bigtom(initial_state,result, result)

        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Final-state repair round %s", repair_round)

            repair_prompt = build_apply_actions_to_state_repair_prompt_bigtom_fix(
                initial_state=initial_state,
                action=action,
                character=character,
                broken_final_state=result.final_state,
                issues=validation.issues,
            )
            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            result = parse_final_state_result_bigtom(raw)
            validation = validate_final_state_result_bigtom(initial_state,result,result)

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            raise RuntimeError(f"Final state validation failed: {issue_messages}")

        return result.final_state


    # def _build_validated_final_state_from_actions(
    #         self,initial_state: Sequence[str],action_effects,
    # ) -> List[str]:
    #     prompt = build_apply_actions_to_state_prompt_bigtom(initial_state=initial_state,action_effects=action_effects)
    #     raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
    #     result = parse_final_state_result(raw)
    #
    #     validation = validate_final_state_result_bigtom(initial_state,action_effects, result)
    #
    #     repair_round = 0
    #     while (not validation.is_valid) and repair_round < self.max_repair_rounds:
    #         repair_round += 1
    #         if self.debug:
    #             logger.info("Final-state repair round %s", repair_round)
    #
    #         repair_prompt = build_apply_actions_to_state_repair_prompt_bigtom(
    #             initial_state=initial_state,
    #             action_effects=action_effects,
    #             broken_final_state=result.final_state,
    #             issues=validation.issues,
    #         )
    #         raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
    #         result = parse_final_state_result(raw)
    #         validation = validate_final_state_result_bigtom(initial_state,action_effects,result)
    #
    #     if not validation.is_valid:
    #         issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
    #         raise RuntimeError(f"Final state validation failed: {issue_messages}")
    #
    #     return result.final_state


    def normalize_question_to_object_query(self,question: str, character: str | None = None) -> str:
        question = question.strip()

        if character:
            if len(character)==1:
                character=character[0]
            prefix = f"Does {character} believe "
            if question.startswith(prefix):
                out = question[len(prefix):].strip()
                return out[0].lower() + out[1:] if out else out

        match = re.match(r"^Does\s+.+?\s+believe\s+(.*)$", question)
        if match:
            out = match.group(1).strip()
            return out[0].lower() + out[1:] if out else out

        return question

    def _answer_question(self, sample: StorySample, final_facts: List[str]) -> tuple[str, str]:

        normalize_question = _reduce_question_with_llm(
            self.llm,
            sample,
            self.max_repair_rounds,
        )

        prompt = build_answer_prompt_bigtom_fix(
            question=normalize_question,
            choices=sample.choices,
            final_facts=final_facts,
            question_characters=sample.question_characters,
        )

        # prompt = build_answer_prompt_bigtom(
        #     question=normalize_question,
        #     choices=sample.choices,
        #     final_facts=final_facts,
        #     question_characters=sample.question_characters,
        # )
        raw = self.llm.generate(prompt, max_tokens=8192, temperature=0.0)
        parsed, validation = _parse_and_validate_answer(
            raw, sample.choices, validate_answer_choice_bigtom
        )


        repair_round = 0
        while (not validation.is_valid) and repair_round < self.max_repair_rounds:
            repair_round += 1
            if self.debug:
                logger.info("Final answer repair round %s", repair_round)

            # repair_prompt = build_answer_repair_prompt_bigtom(
            #     question=normalize_question,
            #     choices=sample.choices,
            #     final_facts=final_facts,
            #     question_characters=sample.question_characters,
            #     broken_answer=parsed["predicted_answer"],
            # )
            repair_prompt = build_answer_repair_prompt_bigtom_fix(
                question=normalize_question,
                choices=sample.choices,
                final_facts=final_facts,
                question_characters=sample.question_characters,
                broken_answer=parsed["predicted_answer"],
            )

            raw = self.llm.generate(repair_prompt, max_tokens=8192, temperature=0.0)
            parsed, validation = _parse_and_validate_answer(
                raw, sample.choices, validate_answer_choice_bigtom
            )

        if not validation.is_valid:
            issue_messages = [f"{x.issue_type}: {x.message}" for x in validation.issues]
            print(parsed["predicted_answer"], sample.choices)
            raise RuntimeError(f"Final answer validation failed: {issue_messages}")

        return parsed["predicted_answer"], parsed.get("reasoning_summary", "")
