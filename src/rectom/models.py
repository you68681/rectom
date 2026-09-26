from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set


@dataclass
class StorySample:
    sample_id: int
    story: str
    question: str
    choices: str
    answer: str
    parsed_steps: List[str]
    prompting_type: Optional[str] = None
    question_order: Optional[int] = None
    deception: Optional[bool] = None

    characters: List[str] = field(default_factory=list)
    deltas: List[Dict[str, Any]] = field(default_factory=list)
    full_states: List[List[str]] = field(default_factory=list)

    action_deltas: List[Dict[str, Any]] = field(default_factory=list)
    only_actions: List[List[str]] = field(default_factory=list)

    state_deltas: List[Dict[str, Any]] = field(default_factory=list)
    only_states: List[List[str]] = field(default_factory=list)

    question_characters: List[str] = field(default_factory=list)
    observation_masks: Dict[str, List[bool]] = field(default_factory=dict)
    observations: Dict[str, List[List[str]]] = field(default_factory=dict)
    perspectives: Dict[str, List[List[str]]] = field(default_factory=dict)

    prediction: Optional[str] = None
    reasoning_summary: Optional[str] = None
    is_correct: Optional[bool] = None
    errors: List[str] = field(default_factory=list)

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepDelta:
    step_index: int
    step_text: str
    added_facts: List[str]
    removed_facts: List[str]


@dataclass
class DeltaGenerationResult:
    characters: List[str]
    steps: List[StepDelta]


@dataclass
# class ObservationMaskResult:
#     character: str
#     visible: List[bool]
class ObservationMaskResult:
    character: str
    observation_basis: list[str]

@dataclass
class ValidationIssue:
    issue_type: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)


@dataclass
class ChoiceSet:
    values: List[str]

    def contains(self, value: str) -> bool:
        return value.strip() in self.values



@dataclass
class ActionEffectResult:
    character: str
    exit_steps: Dict[str, int]
    effective_actions: List[str]

@dataclass
class FinalStateResult:
    final_state: List[str]


@dataclass
class QuestionReductionResult:
    original_question: str
    character_chain: List[str]
    zero_order_question: str

@dataclass
class FinalStateResult_Bigtom:
    final_state: List[str]
    added_facts: List[str]
    removed_facts: List[str]


@dataclass
class ExitStepsResult:
    exit_steps: dict[str, int]

@dataclass
class InitialParticipantsResult:
    characters: List[str]
    initial_present_characters: List[str]


@dataclass
class StateTraceStepFantom:
    step: int
    state_before: List[str]
    action: List[str]
    action_added_facts: List[str]
    action_removed_facts: List[str]
    state_after: List[str]


@dataclass
class ApplyActionsToStateResultFantom:
    state_trace: List[StateTraceStepFantom]


@dataclass
class ApplyActionToStateResultFantom:
    state_before: List[str]
    action: List[str]
    action_added_facts: List[str]
    action_removed_facts: List[str]
    state_after: List[str]
