from __future__ import annotations

import unittest

from rectom.clients.trace_logger import infer_call_type
from rectom.models import (
    DeltaGenerationResult,
    StepDelta,
    StorySample,
    ValidationIssue,
    ValidationResult,
)
from rectom.main import (
    split_story_into_sentences_from_period,
    split_story_into_steps_from_newline,
)
from rectom.prompts.templates import (
    build_delta_prompt,
    build_step_repair_prompt,
)
from rectom.runtime.engine import (
    SequentialFanToMEngine,
    SequentialHiToMEngine,
    _action_is_effective_for_character,
    _complete_presence_alias_removals,
    _deterministic_state_visibility,
    _deterministic_hitom_action_visibility,
    _extract_exit_steps_from_persistent_deltas,
    _parse_and_validate_answer,
    _parse_delta_for_validation,
    _reduce_question_with_llm,
    _restore_delta_source_alignment,
)
from rectom.runtime.validation import apply_delta, apply_state_delta
from rectom.runtime.validation import validate_answer_choice_bigtom
from rectom.utils import fill_perspective_from_observation, parse_fact


class PerspectiveCompletionTests(unittest.TestCase):
    def test_never_backfills_from_a_future_observation(self) -> None:
        completed = fill_perspective_from_observation(
            [[], [], ["in(ball,box)"], []]
        )
        self.assertEqual(completed, [[], [], ["in(ball,box)"], ["in(ball,box)"]])

    def test_figure_two_room_events_update_hidden_states(self) -> None:
        s5 = [
            "in(tomato,blue_container)",
            "in_room(Elizabeth,bedroom)",
            "in_room(Jacob,bedroom)",
        ]
        observations = [s5, [], [], [], []]
        room_deltas = [
            ([], []),
            ([], ["in_room(Elizabeth,bedroom)"]),
            ([], []),
            ([], ["in_room(Jacob,bedroom)"]),
            ([], []),
        ]
        completed = fill_perspective_from_observation(observations, room_deltas)
        self.assertEqual(completed[0], sorted(s5))
        self.assertEqual(
            completed[1],
            ["in(tomato,blue_container)", "in_room(Jacob,bedroom)"],
        )
        self.assertEqual(completed[2], completed[1])
        self.assertEqual(completed[3], ["in(tomato,blue_container)"])
        self.assertEqual(completed[4], completed[3])

    def test_length_mismatch_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            fill_perspective_from_observation([[]], [])

    def test_symbolic_state_visibility_fallback(self) -> None:
        states = [
            ["at(Alice,kitchen)"],
            ["in_room(Alice,garden)"],
            ["at(Alice,waiting_room)"],
            ["at(Bob,kitchen)"],
        ]
        self.assertEqual(
            _deterministic_state_visibility("Alice", states),
            [True, True, False, False],
        )


class PromptConstructionTests(unittest.TestCase):
    def test_dataset_numbers_are_removed_before_prompt_numbering(self) -> None:
        period_steps = split_story_into_sentences_from_period(
            "1 Sophia entered the living_room. 14 Benjamin told Sophia."
        )
        newline_steps = split_story_into_steps_from_newline(
            "1. Sophia entered the living_room\n2 Benjamin exited"
        )
        self.assertEqual(
            period_steps,
            ["Sophia entered the living_room", "Benjamin told Sophia"],
        )
        self.assertEqual(
            newline_steps,
            ["Sophia entered the living_room", "Benjamin exited"],
        )

        sample = StorySample(
            sample_id=1,
            story="",
            question="Where is the lettuce?",
            choices="A: box; B: bucket",
            answer="A",
            parsed_steps=period_steps,
        )
        prompt = build_delta_prompt(sample)
        self.assertIn("1. Sophia entered the living_room", prompt)
        self.assertIn("2. Benjamin told Sophia", prompt)
        self.assertNotIn("1. 1 Sophia", prompt)
        self.assertNotIn("2. 14 Benjamin", prompt)

    def test_trace_call_types_cover_observation_generation_and_repairs(self) -> None:
        self.assertEqual(
            infer_call_type("You are deciding whether Isla is present in each aligned source state."),
            "state_observation_mask",
        )
        self.assertEqual(
            infer_call_type("You are repairing an invalid observation basis sequence."),
            "state_observation_repair",
        )
        self.assertEqual(
            infer_call_type("You are deciding whether Isla can observe an aligned action predicate."),
            "action_observation_mask",
        )
        self.assertEqual(
            infer_call_type("You are repairing an invalid action observation basis sequence."),
            "action_observation_repair",
        )

    def test_single_step_repair_receives_full_story_deltas_and_history(self) -> None:
        story_steps = [
            "Sophia entered the living_room",
            "The lettuce is in the red_bucket",
            "Sophia moved the lettuce to the red_box",
        ]
        deltas = [
            StepDelta(1, story_steps[0], ["at(Sophia,living_room)"], []),
            StepDelta(2, story_steps[1], ["in(lettuce,red_bucket)"], []),
            StepDelta(
                3,
                story_steps[2],
                ["in(lettuce,red_box)"],
                [],
            ),
        ]
        prompt = build_step_repair_prompt(
            story_steps=story_steps,
            all_step_deltas=deltas,
            previous_model_outputs=["INITIAL FULL OUTPUT", "STEP 1 REPAIR OUTPUT"],
            previous_state={"in(lettuce,red_bucket)"},
            broken_step=deltas[2],
            current_computed_state={
                "in(lettuce,red_bucket)",
                "in(lettuce,red_box)",
            },
            issues=[],
        )
        self.assertIn("Complete ordered story events", prompt)
        self.assertIn("3. Sophia moved the lettuce to the red_box", prompt)
        self.assertIn("Complete current delta sequence", prompt)
        self.assertIn('"step_index": 1', prompt)
        self.assertIn('"step_index": 3', prompt)
        self.assertIn("Model output 1:\nINITIAL FULL OUTPUT", prompt)
        self.assertIn("Model output 2:\nSTEP 1 REPAIR OUTPUT", prompt)

    def test_engine_carries_generation_history_into_later_repairs(self) -> None:
        class RepairLLM:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def generate(self, prompt: str, **_kwargs) -> str:
                self.prompts.append(prompt)
                return (
                    '{"step_index":2,"step_text":"Sophia moved the lettuce '
                    'to the red_box","added_facts":["in(lettuce,red_box)"],'
                    '"removed_facts":["in(lettuce,red_bucket)"]}'
                )

        story_steps = [
            "The lettuce is in the red_bucket",
            "Sophia moved the lettuce to the red_box",
        ]
        sample = StorySample(
            sample_id=1,
            story="",
            question="Where is the lettuce?",
            choices="A: box; B: bucket",
            answer="A",
            parsed_steps=story_steps,
        )
        result = DeltaGenerationResult(
            characters=["Sophia"],
            steps=[
                StepDelta(1, story_steps[0], ["in(lettuce,red_bucket)"], []),
                StepDelta(2, story_steps[1], ["in(lettuce,red_box)"], []),
            ],
        )
        validation = ValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(
                    issue_type="object_location_conflict",
                    message="The lettuce has two locations.",
                    details={"step_index": 2},
                )
            ],
        )
        history = ["INITIAL OUTPUT", "EARLIER ROUND OUTPUT"]
        llm = RepairLLM()
        engine = SequentialHiToMEngine(llm_client=llm)
        repaired = engine._repair_invalid_steps(
            result,
            validation,
            sample,
            history,
        )
        self.assertIn("1. The lettuce is in the red_bucket", llm.prompts[0])
        self.assertIn("2. Sophia moved the lettuce to the red_box", llm.prompts[0])
        self.assertIn("Model output 1:\nINITIAL OUTPUT", llm.prompts[0])
        self.assertIn("Model output 2:\nEARLIER ROUND OUTPUT", llm.prompts[0])
        self.assertEqual(len(history), 3)
        self.assertEqual(
            repaired.steps[1].removed_facts,
            ["in(lettuce,red_bucket)"],
        )


class SymbolicRepresentationTests(unittest.TestCase):
    def test_exit_removal_completes_equivalent_presence_aliases(self) -> None:
        result = DeltaGenerationResult(
            characters=["Isla"],
            steps=[
                StepDelta(
                    1,
                    "Isla entered the living_room",
                    ["at(Isla,living_room)", "in_room(Isla,living_room)"],
                    [],
                ),
                StepDelta(
                    2,
                    "Isla exited the living_room",
                    [],
                    ["in_room(Isla,living_room)"],
                ),
            ],
        )
        normalized = _complete_presence_alias_removals(result)
        self.assertEqual(
            set(normalized.steps[1].removed_facts),
            {"at(Isla,living_room)", "in_room(Isla,living_room)"},
        )
        state = apply_delta(set(), normalized.steps[0])
        self.assertEqual(apply_delta(state, normalized.steps[1]), set())

    def test_nested_predicate_arguments_are_preserved(self) -> None:
        predicate, arguments = parse_fact(
            "private_tell(Jacob,Elizabeth,in(tomato,green_box))"
        )
        self.assertEqual(predicate, "private_tell")
        self.assertEqual(
            arguments,
            ("Jacob", "Elizabeth", "in(tomato,green_box)"),
        )

    def test_pure_transient_step_does_not_change_global_state(self) -> None:
        previous = {"in(tomato,blue_container)"}
        transient = StepDelta(
            step_index=2,
            step_text="Jacob privately told Elizabeth that the tomato is in the green box.",
            added_facts=[
                "private_tell(Jacob,Elizabeth,in(tomato,green_box))"
            ],
            removed_facts=["in(tomato,blue_container)"],
        )
        self.assertEqual(apply_state_delta(previous, transient), previous)

    def test_mixed_step_keeps_persistent_and_excludes_transient_facts(self) -> None:
        previous = {"not_present(Alice)"}
        mixed = StepDelta(
            step_index=1,
            step_text="Alice joined and greeted Bob.",
            added_facts=[
                "present(Alice)",
                "communicate(Alice,greeting)",
            ],
            removed_facts=["not_present(Alice)"],
        )
        self.assertEqual(apply_delta(previous, mixed), {"present(Alice)"})

    def test_repair_restores_immutable_source_alignment(self) -> None:
        sample = StorySample(
            sample_id=1,
            story="Alice entered the room.",
            question="Where is Alice?",
            choices="A: room; B: garden",
            answer="A",
            parsed_steps=["Alice entered the room."],
        )
        repaired = DeltaGenerationResult(
            characters=["Alice"],
            steps=[StepDelta(99, "Alice went in.", ["present(Alice)"], [])],
        )
        aligned = _restore_delta_source_alignment(sample, repaired)
        self.assertEqual(aligned.steps[0].step_index, 1)
        self.assertEqual(aligned.steps[0].step_text, sample.parsed_steps[0])
        self.assertEqual(aligned.steps[0].added_facts, ["present(Alice)"])


class HiToMActionTests(unittest.TestCase):
    def test_listener_trust_rule_is_deterministic(self) -> None:
        exits = {"Elizabeth": 6, "Jacob": 8}
        action = "private_tell(Jacob,Elizabeth,in(tomato,green_box))"
        self.assertTrue(
            _action_is_effective_for_character(action, "Elizabeth", exits)
        )
        self.assertFalse(_action_is_effective_for_character(action, "Jacob", exits))

    def test_symbolic_private_and_public_action_visibility_fallback(self) -> None:
        actions = [
            ["private_tell(Jacob,Elizabeth,in(tomato,green_box))"],
            ["public_claim(Jacob,in(tomato,red_box))"],
            [],
        ]
        self.assertEqual(
            _deterministic_hitom_action_visibility("Elizabeth", actions),
            [True, True, False],
        )
        self.assertEqual(
            _deterministic_hitom_action_visibility("Isla", actions),
            [False, True, False],
        )

    def test_exit_step_has_textual_fallback_for_synonymous_delta(self) -> None:
        sample = StorySample(
            sample_id=1,
            story="Elizabeth exited the bedroom.",
            question="Where is the tomato?",
            choices="A: red; B: blue",
            answer="A",
            parsed_steps=["Elizabeth exited the bedroom."],
            characters=["Elizabeth"],
            deltas=[{
                "added_facts": ["not_present(Elizabeth)"],
                "removed_facts": [],
            }],
        )
        self.assertEqual(
            _extract_exit_steps_from_persistent_deltas(sample),
            {"Elizabeth": 1},
        )


class QuestionReductionTests(unittest.TestCase):
    class FakeLLM:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str, **_kwargs) -> str:
            self.prompts.append(prompt)
            return (
                '{"original_question":"Where does Alice believe Bob thinks '
                'the ball is?","character_chain":["Alice","Bob"],'
                '"zero_order_question":"Where is the ball?"}'
            )

    def test_question_reduction_is_performed_by_llm_and_validated(self) -> None:
        llm = self.FakeLLM()
        sample = StorySample(
            sample_id=1,
            story="",
            question="Where does Alice believe Bob thinks the ball is?",
            choices="A: box; B: basket",
            answer="A",
            parsed_steps=[],
            question_characters=["Alice", "Bob"],
        )
        reduced = _reduce_question_with_llm(llm, sample, max_repair_rounds=1)
        self.assertEqual(reduced, "Where is the ball?")
        self.assertEqual(len(llm.prompts), 1)

    def test_malformed_answer_json_enters_repair_validation(self) -> None:
        parsed, validation = _parse_and_validate_answer(
            "The answer is A.",
            "A: room\nB: garden",
            validate_answer_choice_bigtom,
        )
        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.issues[0].issue_type, "invalid_answer_json")
        self.assertEqual(parsed["predicted_answer"], "The answer is A.")

    def test_malformed_delta_json_enters_repair_validation(self) -> None:
        parsed = _parse_delta_for_validation("I cannot provide JSON.")
        self.assertEqual(parsed.characters, [])
        self.assertEqual(parsed.steps, [])


class FanToMMergeTests(unittest.TestCase):
    def test_participation_and_communication_survive_same_step(self) -> None:
        participation = DeltaGenerationResult(
            characters=["Alice", "Bob"],
            steps=[
                StepDelta(
                    step_index=1,
                    step_text="Alice joined and greeted Bob.",
                    added_facts=["present(Alice)"],
                    removed_facts=["not_present(Alice)"],
                )
            ],
        )
        communication = DeltaGenerationResult(
            characters=["Alice", "Bob"],
            steps=[
                StepDelta(
                    step_index=1,
                    step_text="Alice joined and greeted Bob.",
                    added_facts=["communicate(Alice,greeting)"] ,
                    removed_facts=[],
                )
            ],
        )
        merged = SequentialFanToMEngine(llm_client=None)._merge_delta_results(
            participation,
            communication,
        )
        self.assertEqual(
            merged.steps[0].added_facts,
            ["present(Alice)", "communicate(Alice,greeting)"],
        )
        self.assertEqual(merged.steps[0].removed_facts, ["not_present(Alice)"])


if __name__ == "__main__":
    unittest.main()
