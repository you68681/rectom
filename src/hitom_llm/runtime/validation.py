from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Iterable

from hitom_llm.models import (
    DeltaGenerationResult,
    ObservationMaskResult,
    StepDelta,
    StorySample,
    ValidationIssue,
    ValidationResult,
    ActionEffectResult,
    FinalStateResult,
ExitStepsResult,
InitialParticipantsResult,
ApplyActionsToStateResultFantom,
    ApplyActionToStateResultFantom,
    QuestionReductionResult,
)
from hitom_llm.utils import (
    extract_story_characters,
    parse_fact,
    parse_choices,
    state_contains_character,
parse_choices_bigtom,
)


def apply_delta(previous_state: Set[str], step_delta: StepDelta) -> Set[str]:
    """
    Apply only the persistent portion of one delta.

    The model may emit a transient communication/question fact alongside a
    persistent participation fact.  Partitioning by predicate preserves the
    latter without ever inserting the former into the ontic state.  If a step
    contains only transient additions, any model-produced removals are also
    ignored: by definition that event has an empty persistent delta.
    """
    transient_predicates = {
        "private_tell",
        "public_claim",
        "communicate",
        "question",
    }
    parsed_added = [(fact, parse_fact(fact)[0]) for fact in step_delta.added_facts]
    persistent_added = [
        fact for fact, predicate in parsed_added
        if predicate not in transient_predicates
    ]
    has_transient = any(
        predicate in transient_predicates for _fact, predicate in parsed_added
    )

    new_state = set(previous_state)
    if has_transient and not persistent_added:
        return new_state

    for fact in step_delta.removed_facts:
        new_state.discard(fact)
    for fact in persistent_added:
        new_state.add(fact)
    return new_state


def apply_state_delta(previous_state: Set[str], step_delta: StepDelta) -> Set[str]:
    """
    Apply the persistent Hi-ToM portion of one delta.
    """
    return apply_delta(previous_state, step_delta)

def apply_state_delta_fantom(previous_state: Set[str], step_delta: StepDelta) -> Set[str]:
    """
    Apply the persistent FanToM portion of one delta.
    """
    return apply_delta(previous_state, step_delta)


def apply_action_delta(previous_state: list[str], step_delta: StepDelta) -> list[str]:
    new_actions = list(previous_state)

    for fact in step_delta.added_facts:
        if "private_tell" in fact or "public_claim" in fact:
            if fact not in new_actions:
                new_actions.append(fact)
    return new_actions


def apply_action_delta_fantom(previous_state: list[str], step_delta: StepDelta) -> list[str]:
    new_actions = list(previous_state)

    for fact in step_delta.added_facts:
        if "communicate" in fact or "question" in fact:
            if fact not in new_actions:
                new_actions.append(fact)
    return new_actions


def extract_step_tokens(step_text: str) -> Set[str]:
    """
    Extract simple lexical tokens from a story step.

    Example:
        'Carter exited the bedroom.'
    ->
        {'carter', 'exited', 'the', 'bedroom'}
    """
    return set(re.findall(r"[A-Za-z_]+", step_text.lower()))


def extract_fact_content_tokens(fact: str) -> Set[str]:
    """
    Extract argument tokens from a symbolic fact.

    Examples:
        at(Carter,bedroom) -> {'carter', 'bedroom'}
        in(tomato,green_box) -> {'tomato', 'green_box'}
        in_room(Carter) -> {'carter'}
        likes(Elizabeth,tangerine) -> {'elizabeth', 'tangerine'}

    We intentionally ignore the predicate name and only inspect fact arguments.
    """
    _pred, args = parse_fact(fact)
    return {arg.strip().lower() for arg in args if arg.strip()}


def fact_is_supported_by_step_text(fact: str, step_text: str) -> bool:
    """
    A fact is considered supported if at least one argument token of the fact
    appears in the current step text.

    Example:
        fact = "at(Carter,bedroom)"
        step = "Carter exited the bedroom."
        -> supported

        fact = "in(tomato,green_box)"
        step = "Carter exited the bedroom."
        -> unsupported
    """
    step_tokens = extract_step_tokens(step_text)
    fact_tokens = extract_fact_content_tokens(fact)

    if not fact_tokens:
        return False

    return len(fact_tokens & step_tokens) > 0


def validate_mutual_exclusion(state: Set[str], step_index: int) -> List[ValidationIssue]:
    """
    Validate obvious mutual-exclusion constraints in a full state.

    Current checks:
    - one object should not be in multiple locations at the same time
    - one person should not be at multiple locations at the same time
    """
    issues: List[ValidationIssue] = []
    object_locations = defaultdict(set)
    person_locations = defaultdict(set)

    for fact in state:
        pred, args = parse_fact(fact)

        if pred in {"in", "in_container", "at_object"} and len(args) == 2:
            obj, loc = args
            object_locations[obj].add(loc)

        if pred == "at" and len(args) == 2:
            person, loc = args
            person_locations[person].add(loc)

    for obj, locs in object_locations.items():
        if len(locs) > 1:
            issues.append(
                ValidationIssue(
                    issue_type="object_location_conflict",
                    message=f"Object '{obj}' has multiple locations in the same state.",
                    details={
                        "object": obj,
                        "locations": sorted(locs),
                        "step_index": step_index,
                    },
                )
            )

    for person, locs in person_locations.items():
        if len(locs) > 1:
            issues.append(
                ValidationIssue(
                    issue_type="person_location_conflict",
                    message=f"Person '{person}' has multiple locations in the same state.",
                    details={
                        "person": person,
                        "locations": sorted(locs),
                        "step_index": step_index,
                    },
                )
            )

    return issues


def validate_step_delta(previous_state: Optional[Set[str]], step_delta: StepDelta) -> ValidationResult:
    """
    Validate one delta step.

    Validation philosophy:
    1. A fact cannot be both added and removed.
    2. Removed facts must already exist in the previous state.
    3. Added facts must be fully supported by the current step text.
    4. Removed facts must also be fully supported by the current step text.
    5. The resulting full state should not contain obvious conflicts.
    """
    issues: List[ValidationIssue] = []
    prev = previous_state or set()
    added = set(step_delta.added_facts)
    removed = set(step_delta.removed_facts)

    # 1. No overlap between added and removed.
    overlap = added & removed
    if overlap:
        issues.append(
            ValidationIssue(
                issue_type="delta_overlap",
                message="A fact cannot appear in both added_facts and removed_facts.",
                details={
                    "overlap": sorted(overlap),
                    "step_index": step_delta.step_index,
                },
            )
        )

    # 2. Removed facts must already be true in the previous state.
    invalid_removed = removed - prev
    if invalid_removed:
        issues.append(
            ValidationIssue(
                issue_type="invalid_removed_facts",
                message="Some removed facts were not true in the previous state.",
                details={
                    "invalid_removed": sorted(invalid_removed),
                    "step_index": step_delta.step_index,
                },
            )
        )

    # 3. Added facts must be fully supported by the current step text.
    unsupported_added = {
        fact for fact in added
        if not fact_is_supported_by_step_text(fact, step_delta.step_text)
    }
    if unsupported_added:
        issues.append(
            ValidationIssue(
                issue_type="unsupported_added_facts",
                message="Some added facts are not fully supported by the current step text.",
                details={
                    "unsupported_added": sorted(unsupported_added),
                    "step_index": step_delta.step_index,
                    "step_text": step_delta.step_text,
                },
            )
        )

    # 4. Removed facts must also be fully supported by the current step text.
    unsupported_removed = {
        fact for fact in removed
        if not fact_is_supported_by_step_text(fact, step_delta.step_text)
    }
    if unsupported_removed:
        issues.append(
            ValidationIssue(
                issue_type="unsupported_removed_facts",
                message="Some removed facts are not fully supported by the current step text.",
                details={
                    "unsupported_removed": sorted(unsupported_removed),
                    "step_index": step_delta.step_index,
                    "step_text": step_delta.step_text,
                },
            )
        )

    # 5. Validate the resulting full state.
    current_state = apply_delta(prev, step_delta)
    issues.extend(validate_mutual_exclusion(current_state, step_delta.step_index))

    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)


def group_issues_by_step(issues: List[ValidationIssue]) -> Dict[int, List[ValidationIssue]]:
    """
    Group validation issues by step_index when available.
    """
    grouped: Dict[int, List[ValidationIssue]] = {}
    for issue in issues:
        step_index = issue.details.get("step_index")
        if step_index is None:
            continue
        grouped.setdefault(int(step_index), []).append(issue)
    return grouped


def is_public_claim_step(step_text: str) -> bool:
    """
    Heuristic detection of a public-claim step.
    """
    t = step_text.lower()
    return "public" in t and ("claim" in t or "said" in t or "tell" in t)


def is_private_communication_step(step_text: str, character: str) -> bool:
    """
    Heuristic detection of a private communication step relevant to a character.
    """
    t = step_text.lower()
    if "private" not in t:
        return False
    return character.lower() in t


# def validate_observation_mask(
#     character: str,
#     source_states: Sequence[Sequence[str]],
#     story_steps: Sequence[str],
#     mask_result: ObservationMaskResult,
# ) -> ValidationResult:
#     """
#     Validate an LLM-generated observation mask.
#
#     Current conservative rule:
#     if the character is explicitly present in the state, or the step is a relevant
#     private/public communication, the mask should not say False.
#     """
#     issues: List[ValidationIssue] = []
#
#     if mask_result.character and mask_result.character != character:
#         issues.append(
#             ValidationIssue(
#                 issue_type="observation_character_mismatch",
#                 message="Returned character in observation mask does not match the requested character.",
#                 details={
#                     "expected": character,
#                     "returned": mask_result.character,
#                 },
#             )
#         )
#
#     if len(mask_result.visible) != len(story_steps):
#         issues.append(
#             ValidationIssue(
#                 issue_type="observation_length_mismatch",
#                 message="Visibility mask length does not match the number of steps.",
#                 details={
#                     "expected": len(story_steps),
#                     "got": len(mask_result.visible),
#                 },
#             )
#         )
#         return ValidationResult(is_valid=False, issues=issues)
#
#     for i, (state, step_text, visible) in enumerate(
#         zip(source_states, story_steps, mask_result.visible),
#         start=1,
#     ):
#         code_should_be_visible = (
#             state_contains_character(state, character)
#             or is_private_communication_step(step_text, character)
#             or is_public_claim_step(step_text)
#         )
#
#         if code_should_be_visible and not visible:
#             issues.append(
#                 ValidationIssue(
#                     issue_type="observation_false_negative",
#                     message="The character should be able to observe this step, but the mask returned false.",
#                     details={
#                         "step_index": i,
#                         "step_text": step_text,
#                         "state": state,
#                         "character": character,
#                     },
#                 )
#             )
#
#         if (not code_should_be_visible) and visible:
#             issues.append(
#                 ValidationIssue(
#                     issue_type="observation_false_positive",
#                     message="The character should not be able to observe this step, but the mask returned true.",
#                     details={
#                         "step_index": i,
#                         "step_text": step_text,
#                         "state": state,
#                         "character": character,
#                     },
#                 )
#             )
#
#     return ValidationResult(is_valid=(len(issues) == 0), issues=issues)

def extract_character_observation_basis_from_state(
    state: Iterable[str], character: str
) -> str | None:
    """
    Return the exact observation basis for the character from the state.

    Priority:
    1. at(character, location)
    2. in_room(character, location) -> normalized to at(character, location)

    waiting_room is treated as not observable for this task.
    """
    fallback_location = None

    for fact in state:
        pred, args = parse_fact(fact)

        if pred == "at" and len(args) == 2 and args[0] == character:
            location = args[1]
            if location != "waiting_room":
                return f"at({character},{location})"

        if pred == "in_room" and len(args) == 2 and args[0] == character:
            location = args[1]
            if location != "waiting_room":
                fallback_location = location

    if fallback_location is not None:
        return f"at({character},{fallback_location})"

    return None


def extract_character_observation_signature_from_state(
    state: Iterable[str],
    character: str,
) -> tuple[str, str] | None:
    """
    Return the observable signature (character, location) from the state.

    Treat at(character, location) and in_room(character, location) as equivalent.
    waiting_room is treated as not observable.
    """
    fallback_location = None

    for fact in state:
        pred, args = parse_fact(fact)

        if pred == "at" and len(args) == 2 and args[0] == character:
            location = args[1]
            if location != "waiting_room":
                return (character, location)

        if pred == "in_room" and len(args) == 2 and args[0] == character:
            location = args[1]
            if location != "waiting_room":
                fallback_location = location

    if fallback_location is not None:
        return (character, fallback_location)

    return None


def extract_observation_signature_from_basis(
    basis: str,
) -> tuple[str, str] | None:
    """
    Accept:
    - at(character, location)
    - in_room(character, location)

    Return (character, location) if valid, otherwise None.
    """
    basis = basis.strip()
    if basis == "not_observable":
        return None

    pred, args = parse_fact(basis)

    if pred in {"at", "in_room"} and len(args) == 2:
        character = args[0]
        location = args[1]
        if location != "waiting_room":
            return (character, location)

    return None

def validate_observation_mask(
    character: str,
    source_states: Sequence[Sequence[str]],
    story_steps: Sequence[str],
    mask_result: ObservationMaskResult,
) -> ValidationResult:
    """
    Validate an LLM-generated observation basis sequence.

    Allowed outputs:
    - at(character,location)
    - not_observable
    """
    issues: List[ValidationIssue] = []

    if mask_result.character and mask_result.character != character:
        issues.append(
            ValidationIssue(
                issue_type="observation_character_mismatch",
                message="Returned character in observation mask does not match the requested character.",
                details={
                    "expected": character,
                    "returned": mask_result.character,
                },
            )
        )

    if len(mask_result.observation_basis) != len(story_steps):
        issues.append(
            ValidationIssue(
                issue_type="observation_length_mismatch",
                message="Observation basis length does not match the number of steps.",
                details={
                    "expected": len(story_steps),
                    "got": len(mask_result.observation_basis),
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    for i, (state, step_text, returned_basis) in enumerate(
        zip(source_states, story_steps, mask_result.observation_basis),
        start=1,
    ):
        expected_basis = extract_character_observation_basis_from_state(state, character)
        if expected_basis is None:
            expected_basis = "not_observable"

        expected_observable = expected_basis != "not_observable"
        returned_observable = returned_basis != "not_observable"

        if expected_observable and not returned_observable:
            issues.append(
                ValidationIssue(
                    issue_type="observation_false_negative",
                    message="The character should be able to observe this step, but the result returned not_observable.",
                    details={
                        "step_index": i,
                        "state": state,
                        "character": character,
                        "expected_basis": expected_basis,
                        "returned_basis": returned_basis,
                    },
                )
            )
            continue

        if (not expected_observable) and returned_observable:
            issues.append(
                ValidationIssue(
                    issue_type="observation_false_positive",
                    message="The character should not be able to observe this step, but the result returned an observable basis.",
                    details={
                        "step_index": i,
                        "state": state,
                        "character": character,
                        "expected_basis": expected_basis,
                        "returned_basis": returned_basis,
                    },
                )
            )
            continue

        if expected_observable and returned_observable and returned_basis != expected_basis:
            issues.append(
                ValidationIssue(
                    issue_type="observation_wrong_basis",
                    message="The character is observable at this step, but the returned observation basis is incorrect.",
                    details={
                        "step_index": i,
                        "state": state,
                        "character": character,
                        "expected_basis": expected_basis,
                        "returned_basis": returned_basis,
                    },
                )
            )

    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)



def validate_state_observation_mask(
    character: str,
    source_states: Sequence[Sequence[str]],
    story_steps: Sequence[str],
    mask_result: ObservationMaskResult,
) -> ValidationResult:
    """
    Validate an LLM-generated observation basis sequence.

    Allowed outputs:
    - at(character,location)
    - in_room(character,location)
    - not_observable

    Validation compares (character, location), not the exact predicate string.
    """
    issues: List[ValidationIssue] = []

    if mask_result.character and mask_result.character != character:
        issues.append(
            ValidationIssue(
                issue_type="observation_character_mismatch",
                message="Returned character in observation mask does not match the requested character.",
                details={
                    "expected": character,
                    "returned": mask_result.character,
                },
            )
        )

    if len(mask_result.observation_basis) != len(story_steps):
        issues.append(
            ValidationIssue(
                issue_type="observation_length_mismatch",
                message="Observation basis length does not match the number of steps.",
                details={
                    "expected": len(story_steps),
                    "got": len(mask_result.observation_basis),
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    for i, (state, returned_basis) in enumerate(
        zip(source_states, mask_result.observation_basis),
        start=1,
    ):
        expected_sig = extract_character_observation_signature_from_state(state, character)
        returned_sig = extract_observation_signature_from_basis(returned_basis)

        expected_observable = expected_sig is not None
        returned_observable = returned_basis != "not_observable"

        if expected_observable and not returned_observable:
            issues.append(
                ValidationIssue(
                    issue_type="observation_false_negative",
                    message="The character should be observable at this step, but the result returned not_observable.",
                    details={
                        "step_index": i,
                        "state": state,
                        "character": character,
                        "expected_signature": expected_sig,
                        "returned_basis": returned_basis,
                    },
                )
            )
            continue

        if (not expected_observable) and returned_observable:
            issues.append(
                ValidationIssue(
                    issue_type="observation_false_positive",
                    message="The character should not be observable at this step, but the result returned an observable basis.",
                    details={
                        "step_index": i,
                        "state": state,
                        "character": character,
                        "expected_signature": expected_sig,
                        "returned_basis": returned_basis,
                    },
                )
            )
            continue

        if expected_observable and returned_observable:
            if returned_sig is None:
                issues.append(
                    ValidationIssue(
                        issue_type="observation_invalid_basis",
                        message="Returned basis is observable but not a valid basis of the form at(character,location) or in_room(character,location).",
                        details={
                            "step_index": i,
                            "state": state,
                            "character": character,
                            "expected_signature": expected_sig,
                            "returned_basis": returned_basis,
                        },
                    )
                )
                continue

            if returned_sig != expected_sig:
                issues.append(
                    ValidationIssue(
                        issue_type="observation_wrong_basis",
                        message="The returned observation basis has the wrong character and/or location.",
                        details={
                            "step_index": i,
                            "state": state,
                            "character": character,
                            "expected_signature": expected_sig,
                            "returned_signature": returned_sig,
                            "returned_basis": returned_basis,
                        },
                    )
                )

    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)



def validate_state_observation_mask_bigtom(
    character: str,
    source_states: Sequence[Sequence[str]],
    story_steps: Sequence[str],
    mask_result: ObservationMaskResult,
) -> ValidationResult:
    """
    Validate an LLM-generated observation basis sequence.

    Allowed outputs:
    - at(character,location)
    - in_room(character,location)
    - not_observable

    Validation compares (character, location), not the exact predicate string.
    """
    issues: List[ValidationIssue] = []

    if mask_result.character and mask_result.character != character:
        issues.append(
            ValidationIssue(
                issue_type="observation_character_mismatch",
                message="Returned character in observation mask does not match the requested character.",
                details={
                    "expected": character,
                    "returned": mask_result.character,
                },
            )
        )

    if len(mask_result.observation_basis) != len(story_steps):
        issues.append(
            ValidationIssue(
                issue_type="observation_length_mismatch",
                message="Observation basis length does not match the number of steps.",
                details={
                    "expected": len(story_steps),
                    "got": len(mask_result.observation_basis),
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)



def validate_state_observation_mask_fantom(
    character: str,
    source_states: Sequence[Sequence[str]],
    story_steps: Sequence[str],
    mask_result: ObservationMaskResult,
) -> ValidationResult:
    """
    Validate an LLM-generated observation basis sequence.

    Allowed outputs:
    - at(character,location)
    - in_room(character,location)
    - not_observable

    Validation compares (character, location), not the exact predicate string.
    """
    issues: List[ValidationIssue] = []

    if mask_result.character and mask_result.character != character:
        issues.append(
            ValidationIssue(
                issue_type="observation_character_mismatch",
                message="Returned character in observation mask does not match the requested character.",
                details={
                    "expected": character,
                    "returned": mask_result.character,
                },
            )
        )

    if len(mask_result.observation_basis) != len(story_steps):
        issues.append(
            ValidationIssue(
                issue_type="observation_length_mismatch",
                message="Observation basis length does not match the number of steps.",
                details={
                    "expected": len(story_steps),
                    "got": len(mask_result.observation_basis),
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)


def validate_initial_participants_fantom(
    story_steps: Sequence[str],
    result: InitialParticipantsResult,
) -> ValidationResult:
    issues: List[ValidationIssue] = []

    if len(result.characters) != len(set(result.characters)):
        issues.append(
            ValidationIssue(
                issue_type="duplicate_characters",
                message="`characters` contains duplicates.",
                details={"characters": result.characters},
            )
        )

    if len(result.initial_present_characters) != len(set(result.initial_present_characters)):
        issues.append(
            ValidationIssue(
                issue_type="duplicate_initial_present_characters",
                message="`initial_present_characters` contains duplicates.",
                details={"initial_present_characters": result.initial_present_characters},
            )
        )

    extra_initial = sorted(set(result.initial_present_characters) - set(result.characters))
    if extra_initial:
        issues.append(
            ValidationIssue(
                issue_type="initial_present_not_subset",
                message="`initial_present_characters` must be a subset of `characters`.",
                details={
                    "extra_initial_present_characters": extra_initial,
                    "characters": result.characters,
                    "initial_present_characters": result.initial_present_characters,
                },
            )
        )

    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)

def extract_character_action_observation_basis_from_state(
    state: Sequence[str],
    character: str,
) -> str | None:
    """
    Return the exact observable action predicate for `character` from an aligned source action state.

    Rules:
    - If the state is empty, return None.
    - For private_tell(speaker,listener,proposition), only speaker and listener can observe it.
    - For public_claim(speaker,proposition), all characters can observe it.
    - For unknown action predicates, return None by default unless explicit rules are added later.
    """
    for fact in state:
        pred, args = parse_fact(fact)

        if pred == "private_tell":
            if len(args) >= 2:
                speaker, listener = args[0], args[1]
                if character == speaker or character == listener:
                    return fact

        elif pred == "public_claim":
            return fact

    return None

def validate_action_observation_mask(
    character: str,
    source_actions: Sequence[Sequence[str]],
    story_steps: Sequence[str],
    mask_result: ObservationMaskResult,
) -> ValidationResult:
    issues: List[ValidationIssue] = []

    if mask_result.character and mask_result.character != character:
        issues.append(
            ValidationIssue(
                issue_type="action_observation_character_mismatch",
                message="Returned character in action observation mask does not match the requested character.",
                details={
                    "expected": character,
                    "returned": mask_result.character,
                },
            )
        )

    if len(mask_result.observation_basis) != len(source_actions):
        issues.append(
            ValidationIssue(
                issue_type="action_observation_length_mismatch",
                message="Action observation basis length does not match the number of aligned source actions.",
                details={
                    "expected": len(source_actions),
                    "got": len(mask_result.observation_basis),
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    for i, (action_state, returned_basis) in enumerate(
        zip(source_actions, mask_result.observation_basis),
        start=1,
    ):
        expected_basis = extract_character_action_observation_basis_from_state(
            action_state, character
        )
        if expected_basis is None:
            expected_basis = "not_observable"

        expected_observable = expected_basis != "not_observable"
        returned_observable = returned_basis != "not_observable"

        if expected_observable and not returned_observable:
            issues.append(
                ValidationIssue(
                    issue_type="action_observation_false_negative",
                    message="The character should be able to observe this action, but the result returned not_observable.",
                    details={
                        "step_index": i,
                        "action_state": action_state,
                        "character": character,
                        "expected_basis": expected_basis,
                        "returned_basis": returned_basis,
                    },
                )
            )
            continue

        if (not expected_observable) and returned_observable:
            issues.append(
                ValidationIssue(
                    issue_type="action_observation_false_positive",
                    message="The character should not be able to observe this action, but the result returned an observable action basis.",
                    details={
                        "step_index": i,
                        "action_state": action_state,
                        "character": character,
                        "expected_basis": expected_basis,
                        "returned_basis": returned_basis,
                    },
                )
            )
            continue

        if expected_observable and returned_observable and returned_basis != expected_basis:
            issues.append(
                ValidationIssue(
                    issue_type="action_observation_wrong_basis",
                    message="The character can observe this action, but the returned action basis is incorrect.",
                    details={
                        "step_index": i,
                        "action_state": action_state,
                        "character": character,
                        "expected_basis": expected_basis,
                        "returned_basis": returned_basis,
                    },
                )
            )

    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)


def validate_action_observation_mask_bigtom(
    character: str,
    source_actions: Sequence[Sequence[str]],
    story_steps: Sequence[str],
    mask_result: ObservationMaskResult,
) -> ValidationResult:
    issues: List[ValidationIssue] = []

    if mask_result.character and mask_result.character != character:
        issues.append(
            ValidationIssue(
                issue_type="action_observation_character_mismatch",
                message="Returned character in action observation mask does not match the requested character.",
                details={
                    "expected": character,
                    "returned": mask_result.character,
                },
            )
        )

    if len(mask_result.observation_basis) != len(source_actions):
        issues.append(
            ValidationIssue(
                issue_type="action_observation_length_mismatch",
                message="Action observation basis length does not match the number of aligned source actions.",
                details={
                    "expected": len(source_actions),
                    "got": len(mask_result.observation_basis),
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)
    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)

def validate_answer_choice(predicted_answer: str, choices_text: str) -> ValidationResult:
    """
    Validate that the final predicted answer is one of the allowed choice values.
    """
    choice_set = parse_choices(choices_text)
    if choice_set.contains(predicted_answer):
        return ValidationResult(is_valid=True)

    return ValidationResult(
        is_valid=False,
        issues=[
            ValidationIssue(
                issue_type="invalid_choice_value",
                message="Predicted answer is not one of the allowed choice values.",
                details={
                    "predicted_answer": predicted_answer,
                    "allowed_values": choice_set.values,
                },
            )
        ],
    )


def validate_question_reduction(
    question: str,
    question_characters: Sequence[str],
    result: QuestionReductionResult,
) -> ValidationResult:
    issues: List[ValidationIssue] = []
    if result.original_question != question.strip():
        issues.append(
            ValidationIssue(
                issue_type="question_reduction_original_mismatch",
                message="The returned original_question must exactly match the input question.",
                details={"expected": question.strip(), "got": result.original_question},
            )
        )
    if result.character_chain != list(question_characters):
        issues.append(
            ValidationIssue(
                issue_type="question_reduction_chain_mismatch",
                message="The returned character chain must match the validated chain.",
                details={
                    "expected": list(question_characters),
                    "got": result.character_chain,
                },
            )
        )
    if not result.zero_order_question:
        issues.append(
            ValidationIssue(
                issue_type="question_reduction_empty",
                message="zero_order_question must be non-empty.",
            )
        )
    mental_pattern = re.compile(
        r"\b(?:believes?|beliefs?|thinks?|thoughts?|knows?|knowledge|"
        r"remembers?|suspects?|assumes?|opinions?)\b",
        flags=re.IGNORECASE,
    )
    if mental_pattern.search(result.zero_order_question):
        issues.append(
            ValidationIssue(
                issue_type="question_reduction_still_nested",
                message="zero_order_question still contains a mental-state operator.",
                details={"zero_order_question": result.zero_order_question},
            )
        )
    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)

def validate_answer_choice_bigtom(predicted_answer: str, choices_text: str) -> ValidationResult:
    """
    Validate that the final predicted answer is one of the allowed choice values.
    """
    choice_set = parse_choices_bigtom(choices_text)
    if choice_set.contains(predicted_answer):
        return ValidationResult(is_valid=True)

    return ValidationResult(
        is_valid=False,
        issues=[
            ValidationIssue(
                issue_type="invalid_choice_value",
                message="Predicted answer is not one of the allowed choice values.",
                details={
                    "predicted_answer": predicted_answer,
                    "allowed_values": choice_set.values,
                },
            )
        ],
    )


def validate_action_effect(
    current_character: str,
    story_steps: Sequence[str],
    source_actions: Sequence[Sequence[str]],
    result: ActionEffectResult,
) -> ValidationResult:
    """
    Weak validation:
    - check returned character
    - check effective_actions length
    - if a step has no action, output must be not_effective
    - if a step has actions, output must be either:
      - not_effective
      - or one of the exact predicates from that step
    - check exit_steps format and range
    """
    issues: List[ValidationIssue] = []

    if result.character and result.character != current_character:
        issues.append(
            ValidationIssue(
                issue_type="action_effect_character_mismatch",
                message="Returned character in effective action result does not match the requested character.",
                details={
                    "expected": current_character,
                    "returned": result.character,
                },
            )
        )

    if len(result.effective_actions) != len(source_actions):
        issues.append(
            ValidationIssue(
                issue_type="action_effect_length_mismatch",
                message="Effective action sequence length does not match the number of aligned action steps.",
                details={
                    "expected": len(source_actions),
                    "got": len(result.effective_actions),
                },
            )
        )

    for ch, step_idx in result.exit_steps.items():
        if step_idx < 1 or step_idx > len(story_steps):
            issues.append(
                ValidationIssue(
                    issue_type="action_effect_exit_step_out_of_range",
                    message="A returned exit step is outside the valid step range.",
                    details={
                        "character": ch,
                        "step_index": step_idx,
                        "valid_range": [1, len(story_steps)],
                    },
                )
            )

    if len(result.effective_actions) != len(source_actions):
        return ValidationResult(is_valid=False, issues=issues)

    for i, (step_actions, returned_basis) in enumerate(
        zip(source_actions, result.effective_actions),
        start=1,
    ):
        if not step_actions:
            if returned_basis != "not_effective":
                issues.append(
                    ValidationIssue(
                        issue_type="action_effect_on_empty_step",
                        message="The step contains no action, so the result must be not_effective.",
                        details={
                            "step_index": i,
                            "step_actions": list(step_actions),
                            "returned_basis": returned_basis,
                        },
                    )
                )
            continue

        valid_values = set(step_actions) | {"not_effective"}
        if returned_basis not in valid_values:
            issues.append(
                ValidationIssue(
                    issue_type="action_effect_invalid_value",
                    message="Returned value must be either not_effective or an exact aligned action predicate from the current step.",
                    details={
                        "step_index": i,
                        "step_actions": list(step_actions),
                        "returned_basis": returned_basis,
                        "valid_values": sorted(valid_values),
                    },
                )
            )

    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)



def validate_action_effect_v2(
    current_character: str,
    story_steps: Sequence[str],
    source_actions: Sequence[Sequence[str]],
    known_exit_steps: dict[str, int],
    result: ActionEffectResult,
) -> ValidationResult:
    """
    Validation for phase 2:
    - check returned character
    - check returned exit_steps exactly match known_exit_steps
    - check effective_actions length
    - if a step has no action, output must be not_effective
    - if a step has actions, output must be either:
      - not_effective
      - or one of the exact predicates from that step
    """
    issues: List[ValidationIssue] = []

    if result.character and result.character != current_character:
        issues.append(
            ValidationIssue(
                issue_type="action_effect_character_mismatch",
                message="Returned character in effective action result does not match the requested character.",
                details={
                    "expected": current_character,
                    "returned": result.character,
                },
            )
        )

    if result.exit_steps != known_exit_steps:
        issues.append(
            ValidationIssue(
                issue_type="action_effect_exit_steps_mismatch",
                message="Returned exit_steps must exactly match the provided known_exit_steps.",
                details={
                    "expected": known_exit_steps,
                    "returned": result.exit_steps,
                },
            )
        )

    if len(result.effective_actions) != len(source_actions):
        issues.append(
            ValidationIssue(
                issue_type="action_effect_length_mismatch",
                message="Effective action sequence length does not match the number of aligned action steps.",
                details={
                    "expected": len(source_actions),
                    "got": len(result.effective_actions),
                },
            )
        )

    if len(result.effective_actions) != len(source_actions):
        return ValidationResult(is_valid=False, issues=issues)

    for i, (step_actions, returned_basis) in enumerate(
        zip(source_actions, result.effective_actions),
        start=1,
    ):
        if not step_actions:
            if returned_basis != "not_effective":
                issues.append(
                    ValidationIssue(
                        issue_type="action_effect_on_empty_step",
                        message="The step contains no action, so the result must be not_effective.",
                        details={
                            "step_index": i,
                            "step_actions": list(step_actions),
                            "returned_basis": returned_basis,
                        },
                    )
                )
            continue

        valid_values = set(step_actions) | {"not_effective"}
        if returned_basis not in valid_values:
            issues.append(
                ValidationIssue(
                    issue_type="action_effect_invalid_value",
                    message="Returned value must be either not_effective or an exact aligned action predicate from the current step.",
                    details={
                        "step_index": i,
                        "step_actions": list(step_actions),
                        "returned_basis": returned_basis,
                        "valid_values": sorted(valid_values),
                    },
                )
            )

    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)



def _step_explicitly_mentions_character(step_text: str, character: str) -> bool:
    return character in step_text



def validate_exit_steps(
    all_characters: Sequence[str],
    story_steps: Sequence[str],
    result: ExitStepsResult,
) -> ValidationResult:
    """
    Weak validation:
    - check returned characters are from all_characters
    - check exit step range
    - if a character is returned, the corresponding story step must explicitly mention that character
    """
    issues: List[ValidationIssue] = []

    all_character_set = set(all_characters)
    max_step = len(story_steps)

    for ch, step_idx in result.exit_steps.items():
        if ch not in all_character_set:
            issues.append(
                ValidationIssue(
                    issue_type="exit_steps_unknown_character",
                    message="Returned character is not in the provided character list.",
                    details={
                        "character": ch,
                        "all_characters": list(all_characters),
                    },
                )
            )
            continue

        if step_idx < 1 or step_idx > max_step:
            issues.append(
                ValidationIssue(
                    issue_type="exit_steps_step_out_of_range",
                    message="A returned exit step is outside the valid story step range.",
                    details={
                        "character": ch,
                        "step_index": step_idx,
                        "valid_range": [1, max_step],
                    },
                )
            )
            continue

        step_text = story_steps[step_idx - 1]
        if not _step_explicitly_mentions_character(step_text, ch):
            issues.append(
                ValidationIssue(
                    issue_type="exit_steps_character_not_in_step",
                    message="Returned exit step points to a story step that does not explicitly mention the character.",
                    details={
                        "character": ch,
                        "step_index": step_idx,
                        "step_text": step_text,
                    },
                )
            )

    return ValidationResult(
        is_valid=(len(issues) == 0),
        issues=issues,
    )


_IN_FACT_RE = re.compile(r"^in\(([^,]+),([^)]+)\)$")


def validate_final_state_result(
    initial_state: Sequence[str],
    applicable_actions: Sequence[str],
    result: FinalStateResult,
) -> ValidationResult:
    """
    Weak validation:
    - final_state must be a list
    - each item must be a non-empty string
    - no duplicate facts
    - final_state must not contain action predicates
    - final_state should not contain multiple in(object,location) facts for the same object
    """
    issues: List[ValidationIssue] = []

    if not isinstance(result.final_state, list):
        issues.append(
            ValidationIssue(
                issue_type="final_state_not_list",
                message="The returned final_state must be a list.",
                details={"returned": result.final_state},
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    normalized_facts: list[str] = []
    for i, fact in enumerate(result.final_state, start=1):
        if not isinstance(fact, str) or not fact.strip():
            issues.append(
                ValidationIssue(
                    issue_type="final_state_invalid_item",
                    message="Each final_state item must be a non-empty string.",
                    details={"index": i, "value": fact},
                )
            )
            continue
        normalized_facts.append(fact.strip())

    returned_set = set(normalized_facts)
    if len(returned_set) != len(normalized_facts):
        issues.append(
            ValidationIssue(
                issue_type="final_state_duplicate_facts",
                message="The returned final_state contains duplicate facts.",
                details={"final_state": normalized_facts},
            )
        )

    # final_state should not contain any raw applicable action predicate
    applicable_action_set = {
        x.strip() for x in applicable_actions
        if isinstance(x, str) and x.strip()
    }

    for i, fact in enumerate(normalized_facts, start=1):
        if fact in applicable_action_set:
            issues.append(
                ValidationIssue(
                    issue_type="final_state_contains_action_predicate",
                    message="The final_state must contain only state facts, not action predicates.",
                    details={"index": i, "value": fact},
                )
            )


    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)



def validate_final_state_result_bigtom(
    initial_state: Sequence[str],
    action_effects,
    result: FinalStateResult,
) -> ValidationResult:
    """
    Weak validation:
    - final_state must be a list
    - each item must be a non-empty string
    - no duplicate facts
    - final_state must not contain action predicates
    - final_state should not contain multiple in(object,location) facts for the same object
    """
    issues: List[ValidationIssue] = []

    if not isinstance(result.final_state, list):
        issues.append(
            ValidationIssue(
                issue_type="final_state_not_list",
                message="The returned final_state must be a list.",
                details={"returned": result.final_state},
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    normalized_facts: list[str] = []
    for i, fact in enumerate(result.final_state, start=1):
        if not isinstance(fact, str) or not fact.strip():
            issues.append(
                ValidationIssue(
                    issue_type="final_state_invalid_item",
                    message="Each final_state item must be a non-empty string.",
                    details={"index": i, "value": fact},
                )
            )
            continue
        normalized_facts.append(fact.strip())

    returned_set = set(normalized_facts)
    if len(returned_set) != len(normalized_facts):
        issues.append(
            ValidationIssue(
                issue_type="final_state_duplicate_facts",
                message="The returned final_state contains duplicate facts.",
                details={"final_state": normalized_facts},
            )
        )

    # final_state should not contain any raw applicable action predicate
    action_added_fact_set = {
        x.strip() for x in action_effects.added_facts
        if isinstance(x, str) and x.strip()
    }
    action_removed_fact_set = {
        x.strip() for x in action_effects.removed_facts
        if isinstance(x, str) and x.strip()
    }

    for fact in action_added_fact_set:
        if fact not in normalized_facts:
            issues.append(
                ValidationIssue(
                    issue_type="final_state_contains_removed_fact",
                    message="The final_state must contains facts in the action added facts.",
                    details={"value": fact},
                )
            )

    for i, fact in enumerate(normalized_facts, start=1):
        if fact in action_removed_fact_set:
            issues.append(
                ValidationIssue(
                    issue_type="final_state_contains_removed_fact",
                    message="The final_state should not contain facts in the action removed facts.",
                    details={"index": i, "value": fact},
                )
            )


    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)




def normalize_symbolic_fact_sequence(facts) -> List[str]:
    """
    Normalize a sequence of symbolic facts for stable comparison.

    Behavior:
    - if input is not a list/tuple/set, return []
    - convert each item to str
    - strip leading/trailing whitespace
    - drop empty strings
    - remove duplicates while preserving order
    """
    if not isinstance(facts, (list, tuple, set)):
        return []

    normalized: List[str] = []
    seen = set()

    for fact in facts:
        text = str(fact).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        normalized.append(text)

    return normalized



def validate_final_state_result_fantom(
    states: Sequence[Sequence[str]],
    actions: Sequence[Sequence[str]],
    result: ApplyActionsToStateResultFantom,
) -> ValidationResult:
    issues: List[ValidationIssue] = []


    expected_len = len(states)
    if len(result.state_trace) != expected_len:
        issues.append(
            ValidationIssue(
                issue_type="state_trace_length_mismatch",
                message="state_trace length does not match the number of input states.",
                details={
                    "expected": expected_len,
                    "got": len(result.state_trace),
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    if len(actions) != expected_len:
        issues.append(
            ValidationIssue(
                issue_type="input_length_mismatch",
                message="Number of input states does not match number of input actions.",
                details={
                    "states_len": len(states),
                    "actions_len": len(actions),
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    for idx, trace_step in enumerate(result.state_trace, start=1):
        expected_state_before = normalize_symbolic_fact_sequence(states[idx - 1])
        expected_action = normalize_symbolic_fact_sequence(actions[idx - 1])

        if trace_step.step != idx:
            issues.append(
                ValidationIssue(
                    issue_type="step_index_mismatch",
                    message="Step index in state_trace does not match its position.",
                    details={
                        "expected": idx,
                        "got": trace_step.step,
                    },
                )
            )

        if normalize_symbolic_fact_sequence(trace_step.state_before) != expected_state_before:
            issues.append(
                ValidationIssue(
                    issue_type="state_before_mismatch",
                    message="state_before does not match the corresponding input state.",
                    details={
                        "step": idx,
                        "expected": expected_state_before,
                        "got": trace_step.state_before,
                    },
                )
            )

        if normalize_symbolic_fact_sequence(trace_step.action) != expected_action:
            issues.append(
                ValidationIssue(
                    issue_type="action_mismatch",
                    message="action does not match the corresponding input action list.",
                    details={
                        "step": idx,
                        "expected": expected_action,
                        "got": trace_step.action,
                    },
                )
            )

        state_before_set = set(expected_state_before)
        added_set = set(normalize_symbolic_fact_sequence(trace_step.action_added_facts))
        removed_set = set(normalize_symbolic_fact_sequence(trace_step.action_removed_facts))
        state_after_set = set(normalize_symbolic_fact_sequence(trace_step.state_after))

        expected_state_after_set = (state_before_set - removed_set) | added_set

        if state_after_set != expected_state_after_set:
            issues.append(
                ValidationIssue(
                    issue_type="state_after_incorrect",
                    message="state_after is not consistent with state_before, action_added_facts, and action_removed_facts.",
                    details={
                        "step": idx,
                        "state_before": sorted(state_before_set),
                        "action_added_facts": sorted(added_set),
                        "action_removed_facts": sorted(removed_set),
                        "expected_state_after": sorted(expected_state_after_set),
                        "got_state_after": sorted(state_after_set),
                    },
                )
            )

        # Optional sanity checks
        overlap = added_set & removed_set
        if overlap:
            issues.append(
                ValidationIssue(
                    issue_type="added_removed_overlap",
                    message="Some facts appear in both action_added_facts and action_removed_facts.",
                    details={
                        "step": idx,
                        "overlap": sorted(overlap),
                    },
                )
            )

        missing_removed = removed_set - state_before_set
        if missing_removed:
            issues.append(
                ValidationIssue(
                    issue_type="remove_nonexistent_fact",
                    message="Some removed facts are not present in state_before.",
                    details={
                        "step": idx,
                        "missing_removed": sorted(missing_removed),
                    },
                )
            )

    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)


from typing import Sequence, List


def validate_final_state_result_fantom_v2(
    state: Sequence[str],
    action: Sequence[str],
    result: ApplyActionToStateResultFantom,
) -> ValidationResult:
    issues: List[ValidationIssue] = []

    expected_state_before = normalize_symbolic_fact_sequence(state)
    expected_action = normalize_symbolic_fact_sequence(action)

    got_state_before = normalize_symbolic_fact_sequence(result.state_before)
    got_action = normalize_symbolic_fact_sequence(result.action)
    got_added = normalize_symbolic_fact_sequence(result.action_added_facts)
    got_removed = normalize_symbolic_fact_sequence(result.action_removed_facts)
    got_state_after = normalize_symbolic_fact_sequence(result.state_after)

    if got_state_before != expected_state_before:
        issues.append(
            ValidationIssue(
                issue_type="state_before_mismatch",
                message="state_before does not match the input state.",
                details={
                    "expected": expected_state_before,
                    "got": got_state_before,
                },
            )
        )

    if got_action != expected_action:
        issues.append(
            ValidationIssue(
                issue_type="action_mismatch",
                message="action does not match the input action list.",
                details={
                    "expected": expected_action,
                    "got": got_action,
                },
            )
        )

    overlap = sorted(set(got_added) & set(got_removed))
    if overlap:
        issues.append(
            ValidationIssue(
                issue_type="added_removed_overlap",
                message="Some facts appear in both action_added_facts and action_removed_facts.",
                details={"overlap": overlap},
            )
        )

    missing_removed = sorted(set(got_removed) - set(expected_state_before))
    if missing_removed:
        issues.append(
            ValidationIssue(
                issue_type="remove_nonexistent_fact",
                message="Some removed facts are not present in state_before.",
                details={"missing_removed": missing_removed},
            )
        )

    expected_state_after = sorted((set(expected_state_before) - set(got_removed)) | set(got_added))
    if sorted(got_state_after) != expected_state_after:
        issues.append(
            ValidationIssue(
                issue_type="state_after_incorrect",
                message="state_after is inconsistent with state_before, action_added_facts, and action_removed_facts.",
                details={
                    "state_before": expected_state_before,
                    "action_added_facts": got_added,
                    "action_removed_facts": got_removed,
                    "expected_state_after": expected_state_after,
                    "got_state_after": sorted(got_state_after),
                },
            )
        )

    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)



def validate_generation_result(sample: StorySample, result: DeltaGenerationResult) -> ValidationResult:
    issues: List[ValidationIssue] = []

    if len(result.steps)==0:
        issues.append(
            ValidationIssue(
                issue_type="invalid_json",
                message="Returned  is not the json formate.",
                details={
                    "expected": "valid json response",
                    "got": "invalid response",
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    expected_characters = set(extract_story_characters(sample.parsed_steps))
    returned_characters = set(result.characters)
    if expected_characters != returned_characters:
        issues.append(
            ValidationIssue(
                issue_type="character_mismatch",
                message="Returned character set does not match the story characters.",
                details={
                    "expected_characters": sorted(expected_characters),
                    "returned_characters": sorted(returned_characters),
                    "missing": sorted(expected_characters - returned_characters),
                    "extra": sorted(returned_characters - expected_characters),
                },
            )
        )

    if len(result.steps) != len(sample.parsed_steps):
        issues.append(
            ValidationIssue(
                issue_type="step_count_mismatch",
                message="Returned step count does not match story step count.",
                details={
                    "expected": len(sample.parsed_steps),
                    "got": len(result.steps),
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    previous_state: Set[str] = set()

    for i, step_delta in enumerate(result.steps):
        expected_index = i + 1
        expected_text = sample.parsed_steps[i]

        if step_delta.step_index != expected_index:
            issues.append(
                ValidationIssue(
                    issue_type="step_index_mismatch",
                    message="Step index does not match story order.",
                    details={
                        "expected_step_index": expected_index,
                        "returned_step_index": step_delta.step_index,
                        "step_index": expected_index,
                    },
                )
            )

        if step_delta.step_text.strip() != expected_text.strip():
            issues.append(
                ValidationIssue(
                    issue_type="step_text_mismatch",
                    message="Step text does not match the original story step.",
                    details={
                        "expected_step_text": expected_text,
                        "returned_step_text": step_delta.step_text,
                        "step_index": expected_index,
                    },
                )
            )

        step_validation = validate_step_delta(previous_state, step_delta)
        issues.extend(step_validation.issues)
        previous_state = apply_delta(previous_state, step_delta)

    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)


def validate_generation_result_bigtom(sample: StorySample, result: DeltaGenerationResult) -> ValidationResult:
    issues: List[ValidationIssue] = []

    if len(result.steps)==0:
        issues.append(
            ValidationIssue(
                issue_type="invalid_json",
                message="Returned  is not the json formate.",
                details={
                    "expected": "valid json response",
                    "got": "invalid response",
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    if len(result.steps) != len(sample.parsed_steps):
        issues.append(
            ValidationIssue(
                issue_type="step_count_mismatch",
                message="Returned step count does not match story step count.",
                details={
                    "expected": len(sample.parsed_steps),
                    "got": len(result.steps),
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    # previous_state: Set[str] = set()
    #
    for i, step_delta in enumerate(result.steps):
        expected_index = i + 1
        expected_text = sample.parsed_steps[i]
        if step_delta.step_index != expected_index:
            issues.append(
                ValidationIssue(
                    issue_type="step_index_mismatch",
                    message="Step index does not match story order.",
                    details={
                        "expected_step_index": expected_index,
                        "returned_step_index": step_delta.step_index,
                        "step_index": expected_index,
                    },
                )
            )
        if step_delta.step_text.strip() != expected_text.strip():
            issues.append(
                ValidationIssue(
                    issue_type="step_text_mismatch",
                    message="Step text does not match the original story step.",
                    details={
                        "expected_step_text": expected_text,
                        "returned_step_text": step_delta.step_text,
                        "step_index": expected_index,
                    },
                )
            )


    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)



def validate_generation_result_fantom(sample: StorySample, result: DeltaGenerationResult) -> ValidationResult:
    issues: List[ValidationIssue] = []

    if len(result.steps)==0:
        issues.append(
            ValidationIssue(
                issue_type="invalid_json",
                message="Returned  is not the json formate.",
                details={
                    "expected": "valid json response",
                    "got": "invalid response",
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    if len(result.steps) != len(sample.parsed_steps):
        issues.append(
            ValidationIssue(
                issue_type="step_count_mismatch",
                message="Returned step count does not match story step count.",
                details={
                    "expected": len(sample.parsed_steps),
                    "got": len(result.steps),
                },
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    # previous_state: Set[str] = set()
    #
    for i, step_delta in enumerate(result.steps):
        expected_index = i + 1
        expected_text = sample.parsed_steps[i]
        if step_delta.step_index != expected_index:
            issues.append(
                ValidationIssue(
                    issue_type="step_index_mismatch",
                    message="Step index does not match story order.",
                    details={
                        "expected_step_index": expected_index,
                        "returned_step_index": step_delta.step_index,
                        "step_index": expected_index,
                    },
                )
            )
        if step_delta.step_text.strip() != expected_text.strip():
            issues.append(
                ValidationIssue(
                    issue_type="step_text_mismatch",
                    message="Step text does not match the original story step.",
                    details={
                        "expected_step_text": expected_text,
                        "returned_step_text": step_delta.step_text,
                        "step_index": expected_index,
                    },
                )
            )


    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)
