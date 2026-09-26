from __future__ import annotations
import re
import logging
from typing import Dict, List, Sequence, Set
from typing import Sequence
from hitom_llm.models import DeltaGenerationResult, StorySample,ActionEffectResult,ExitStepsResult,FinalStateResult_Bigtom,StepDelta
from hitom_llm.parsers import parse_answer_result, parse_delta_result, parse_observation_mask, parse_single_step_delta,parse_action_effect,normalize_fact_list, normalize_action_list,parse_final_state_result,parse_exit_steps,parse_final_state_result_bigtom,parse_initial_participants,parse_final_state_result_fantom,parse_final_state_result_fantom_v2
from hitom_llm.prompts.templates import (
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

)
from hitom_llm.runtime.validation import (
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
)
from hitom_llm.utils import extract_question_characters, fill_perspective_from_observation,extract_question_characters_fantom

logger = logging.getLogger(__name__)


class SequentialHiToMCoTEngine:
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
        DEFAULT_TOM_NOTE = """You should assume the following.
        (1) An agent witnesses everything and every movement before exiting a location.
        (2) An agent A can infer another agent B's mental state only if A and B have been in the same location, or have private or public interactions.
        (3) Note that every agent tends to lie. What an agent A tells others doesn't affect A's actual belief. An agent tends to trust an agent that exited the room later than himself. The exit order is known to all agents.
        (4) Agents in private communications know that others won't hear them, but they know that anyone can hear any public claims."""


        try:
            prompt=self.build_answer_prompt(sample.question, sample.choices, sample.story, DEFAULT_TOM_NOTE)
            raw = self.llm.generate((prompt), max_tokens=8192, temperature=0.0)
            parsed=parse_answer_result(raw)
            predicted_answer,reasoning_summary = parsed["predicted_answer"], parsed.get("reasoning_summary", "")
            sample.prediction=predicted_answer
            print(f"sample.prediction: {sample.prediction}, sample.answer: {sample.answer}")
            sample.is_correct = sample.prediction == sample.answer

        except Exception as exc:
            logger.exception("Failed to process sample_id=%s", sample.sample_id)
            sample.errors.append(str(exc))
            sample.is_correct = False
        return sample

    def build_answer_prompt(self, question: str, choices: str, story: str, note: str) -> str:
        return f"""
    You are answering a multiple-choice theory-of-mind question. Think step-by-step.

    Return valid JSON only.
    The output schema:
    {{
      "reasoning_summary": "short summary",
      "predicted_answer": "green_drawer"
    }}

    Story:
    {story}

    Question:
    {question}

    Choices:
    {choices}

    Assumptions:
    {note}

    Predict exactly one choice value, not the letter.
    Do not output the option letter.
    Do not output any extra text outside the JSON.
    """.strip()


class SequentialBigToMCoTEngine:
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
            prompt=self.build_answer_prompt_bigtom(sample.question, sample.choices, sample.story)
            raw = self.llm.generate((prompt), max_tokens=8192, temperature=0.0)
            parsed=parse_answer_result(raw)
            predicted_answer,reasoning_summary = parsed["predicted_answer"], parsed.get("reasoning_summary", "")
            sample.prediction=predicted_answer
            sample.is_correct = sample.prediction == sample.answer

        except Exception as exc:
            logger.exception("Failed to process sample_id=%s", sample.sample_id)
            sample.errors.append(str(exc))
            sample.is_correct = False
        return sample

    def build_answer_prompt_bigtom(self, question: str, choices: str, story: str) -> str:
        return f"""
    You are answering a multiple-choice theory-of-mind question. Think step-by-step.

    Return valid JSON only.

    Output example schema:
    {{
      "reasoning_summary": "short summary"
      "predicted_answer": "C",
    }}

    Important constraints:
    - Do not output any extra text outside the JSON.
    - Output ONLY the option letter (e.g., "A", "B", "C").
    - Do NOT output the choice text.
    - Do NOT include any additional characters or punctuation.
    - The predicted_answer must be one of the option letters.

    Story:
    {story}


    Question:
    {question}

    Choices:
    {choices}
    """.strip()


class SequentialFanToMCoTEngine:
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
            prompt=self.build_answer_prompt_bigtom_fix(sample.question, sample.choices, sample.story)
            raw = self.llm.generate((prompt), max_tokens=8192, temperature=0.0)
            parsed=parse_answer_result(raw)
            predicted_answer,reasoning_summary = parsed["predicted_answer"], parsed.get("reasoning_summary", "")
            sample.prediction=predicted_answer
            sample.is_correct = sample.prediction == sample.answer

        except Exception as exc:
            logger.exception("Failed to process sample_id=%s", sample.sample_id)
            sample.errors.append(str(exc))
            sample.is_correct = False
        return sample

    def build_answer_prompt_bigtom_fix(self, question: str, choices: str, story: str) -> str:
        return f"""
    You are answering a multiple-choice theory-of-mind question. Think step-by-step.

    Return valid JSON only.

    Output example schema:
    {{
      "reasoning_summary": "short summary"
      "predicted_answer": "C",
    }}

    Important constraints:
    - Do not output any extra text outside the JSON.
    - Output ONLY the option letter (e.g., "A", "B", "C").
    - Do NOT output the choice text.
    - Do NOT include any additional characters or punctuation.
    - The predicted_answer must be one of the option letters.


    Story:
    {story}

    Question:
    {question}

    Choices:
    {choices}
    """.strip()


