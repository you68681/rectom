# # src/hitom_llm/tomi/sim_utils_tomi.py
#
# from __future__ import annotations
#
# from dataclasses import dataclass, field
# from typing import Dict, List, Optional, Protocol, Tuple
#
# from .fantom_prompts import (
#     PERSPECTIVE_PROMPT,
#     LLAMA_SIM_PROMPT,
#     GPT_SIM_PROMPT,
# )
#
#
# class LLMClient(Protocol):
#     @property
#     def model_name(self) -> str:
#         ...
#
#     def generate(self, prompt: str) -> str:
#         ...
#
#
# @dataclass
# class Agent:
#     llm: LLMClient
#     name: str = ""
#     perspective: str = ""
#     was_asked: Optional[str] = None
#     replied: Optional[str] = None
#     debug: bool = False
#
#     def __post_init__(self) -> None:
#         if "llama" in self.llm.model_name.lower():
#             self.eval_prompt = LLAMA_SIM_PROMPT
#         else:
#             self.eval_prompt = GPT_SIM_PROMPT
#
#     def eval_question(self, question: str) -> Tuple[str, str]:
#         prompt = self.eval_prompt.format(
#             perspective=self.perspective,
#             name=self.name,
#             question=question,
#         )
#         choice = self.llm.generate(prompt)
#         self.was_asked = prompt
#         self.replied = choice
#         return choice, self.perspective
#
#
# @dataclass
# class World:
#     llm: LLMClient
#     context: dict
#     debug: bool = False
#     sim_model: Optional[LLMClient] = None
#     story: str = field(init=False)
#     agent_names: List[str] = field(default_factory=list)
#     agents: Dict[str, Agent] = field(default_factory=dict)
#     perspectives: Dict[str, str] = field(default_factory=dict)
#
#     def __post_init__(self) -> None:
#         self.story = self.context["story"]
#         if self.sim_model is None:
#             self.sim_model = self.llm
#
#     def parse_characters(self) -> None:
#         prompt = f"""\
# {self.story}
# What are the characters in this story?
# Output only the character names, separated by commas."""
#         names = self.llm.generate(prompt).replace(" ", "").split(",")
#         self.agent_names = [n.strip() for n in names if n.strip()]
#         if self.debug:
#             print("Agent names:", self.agent_names)
#
#     def take_perspective(self, character_name: Optional[str] = None) -> None:
#         if character_name is None:
#             for character in self.agent_names:
#                 self.perspectives[character] = self.llm.generate(
#                     PERSPECTIVE_PROMPT.format(story=self.story, character=character)
#                 )
#                 if self.debug:
#                     print(f"Perspective of {character}:", self.perspectives[character])
#         else:
#             self.perspectives[character_name] = self.llm.generate(
#                 PERSPECTIVE_PROMPT.format(story=self.story, character=character_name)
#             )
#             if self.debug:
#                 print(f"Perspective of {character_name}:", self.perspectives[character_name])
#
#     def setup_agents(self) -> None:
#         for agent_name, perspective in self.perspectives.items():
#             agent = Agent(self.sim_model, name=agent_name)
#             agent.perspective = perspective
#             self.agents[agent_name] = agent
#
#     def eval_question(self, question: str, agent_name: Optional[str] = None) -> Tuple[str, str]:
#         if agent_name is None:
#             answer = self.llm.generate(
#                 f"{self.story}\nBased on the above information, answer the following question:\n{question}"
#             )
#             return answer, "Truth Question"
#         return self.agents[agent_name].eval_question(question)
#
#
# def eval_question(
#     llm: LLMClient,
#     context: dict,
#     question: str,
#     debug: bool = False,
#     sim_model: Optional[LLMClient] = None,
# ) -> Tuple[str, str]:
#     world = World(llm=llm, context=context, debug=debug, sim_model=sim_model)
#
#     # Original Shawn logic: question subject is the 3rd token. :contentReference[oaicite:2]{index=2}
#     question_subject = question.split(" ")[2]
#
#     world.parse_characters()
#
#     if question_subject not in world.agent_names:
#         answer, perspective = world.eval_question(question)
#         return answer, perspective
#
#     world.take_perspective(character_name=question_subject)
#     world.setup_agents()
#     answer, perspective = world.eval_question(question, agent_name=question_subject)
#     return answer, perspective
#
#
# def one_big_prompt(llm: LLMClient, story: str, question: str) -> Tuple[str, str]:
#     character = question.split(" ")[2]
#
#     prompt = f"""\
# {story}
# What are the characters in this story?
# Output only the character names, separated by commas."""
#     agent_names = llm.generate(prompt).replace(" ", "").split(",")
#     agent_names = [n.strip() for n in agent_names if n.strip()]
#
#     if character not in agent_names:
#         answer = llm.generate(
#             f"{story}\nBased on the above information, answer the following question:\n{question}"
#         )
#         return answer, "Truth Question"
#
#     output = llm.generate(
#         ABLATION_ONEPROMPT_SIM.format(character=character, story=story, question=question)
#     )
#     marker = "Answer:"
#     if marker in output:
#         answer = output[output.rfind(marker) + len(marker):].strip()
#     else:
#         answer = output.strip()
#     return answer, output


from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from .fantom_prompts import (
    PERSPECTIVE_PROMPT,
    LLAMA_SIM_PROMPT,
    GPT_SIM_PROMPT,
    MULTI_PERSPECTIVE_QA_PROMPT,
)


class LLMClient(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class Agent:
    llm: LLMClient
    name: str = ""
    perspective: str = ""
    was_asked: Optional[str] = None
    replied: Optional[str] = None
    debug: bool = False

    def __post_init__(self) -> None:
        if "llama" in self.llm.model_name.lower():
            self.eval_prompt = LLAMA_SIM_PROMPT
        else:
            self.eval_prompt = GPT_SIM_PROMPT

    def eval_question(self, question: str) -> Tuple[str, str]:
        prompt = self.eval_prompt.format(
            perspective=self.perspective,
            name=self.name,
            question=question,
        )
        choice = self.llm.generate(prompt)
        self.was_asked = prompt
        self.replied = choice
        return choice, self.perspective


@dataclass
class World:
    llm: LLMClient
    context: dict
    debug: bool = False
    sim_model: Optional[LLMClient] = None
    story: str = field(init=False)
    agent_names: List[str] = field(default_factory=list)
    agents: Dict[str, Agent] = field(default_factory=dict)
    perspectives: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.story = self.context["story"]
        if self.sim_model is None:
            self.sim_model = self.llm

    def strip_reasoning(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        return text.strip()

    def parse_characters(self) -> None:
        self.agent_names = extract_characters_from_story(self.story)

        if self.debug:
            print("Agent names:", self.agent_names)

    def take_perspective(self, character_name: Optional[str] = None) -> None:
        if character_name is None:
            for character in self.agent_names:
                raw = self.llm.generate(
                    PERSPECTIVE_PROMPT.format(
                        story=self.story,
                        character=character,
                    )
                )
                self.perspectives[character] = self.strip_reasoning(raw)

                if self.debug:
                    print(f"Perspective of {character}:", self.perspectives[character])
        else:
            raw = self.llm.generate(
                PERSPECTIVE_PROMPT.format(
                    story=self.story,
                    character=character_name,
                )
            )
            self.perspectives[character_name] = self.strip_reasoning(raw)

            if self.debug:
                print(f"Perspective of {character_name}:", self.perspectives[character_name])

    def setup_agents(self) -> None:
        for agent_name, perspective in self.perspectives.items():
            agent = Agent(self.sim_model, name=agent_name)
            agent.perspective = perspective
            self.agents[agent_name] = agent

    def eval_truth_question(self, question: str) -> Tuple[str, str]:
        answer = self.llm.generate(
            f"""{self.story}

Based on the above information, answer the following question:
{question}

You must choose one of the above choices.
Do not output explanations.
Please output only the final choice starting with Answer:
For example: Answer: A
"""
        )
        return answer, "Truth Question"

    def eval_multi_perspective_question(
        self,
        question: str,
        question_characters: List[str],
    ) -> Tuple[str, str]:
        perspective_block = build_perspective_block(
            perspectives=self.perspectives,
            question_characters=question_characters,
        )

        prompt = MULTI_PERSPECTIVE_QA_PROMPT.format(
            story=self.story,
            character_chain=" -> ".join(question_characters),
            perspective_block=perspective_block,
            question=question,
        )

        answer = self.sim_model.generate(prompt)
        return answer, perspective_block


def extract_characters_from_story(story: str) -> List[str]:
    """
    FanTom/dialogue version.

    1. First try structured dialogue-style patterns.
    2. If no strong pattern is found, fallback to capitalized names.
    """

    names: List[str] = []
    seen = set()

    def add_name(name: str) -> None:
        name = name.strip()
        name = re.sub(r"[^A-Za-z\-']", "", name)

        if not name:
            return

        if not re.match(r"^[A-Z][A-Za-z\-']*$", name):
            return

        stop_words = {
            "The", "A", "An", "And", "But", "Or",
            "Question", "Answer", "Choices", "Story",
            "Based", "Choose", "For", "Do", "Please",
            "What", "Where", "Who", "When", "Why", "How",
        }

        if name in stop_words:
            return

        if name not in seen:
            seen.add(name)
            names.append(name)

    # Strong patterns
    for raw_line in story.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line = re.sub(r"^\s*\d+[\.\)]?\s*", "", line).strip()

        # Pattern: "Alice: ..."
        m = re.match(r"^([A-Z][A-Za-z\-']+)\s*:", line)
        if m:
            add_name(m.group(1))
            continue

        # Pattern: "Alice said ..." / "Alice asked ..." / "Alice replied ..."
        m = re.match(
            r"^([A-Z][A-Za-z\-']+)\s+(said|asked|replied|answered|told|mentioned|explained|shared|joined|left|entered)\b",
            line,
        )
        if m:
            add_name(m.group(1))
            continue

        # Pattern: "Alice, Bob and Charlie were ..."
        m = re.match(r"^(.+?)\s+(were|are|joined|entered)\b", line)
        if m:
            part = m.group(1)
            if "," in part or " and " in part:
                part = part.replace(" and ", ", ")
                for x in part.split(","):
                    add_name(x)

    if names:
        return names

    # Fallback only when no strong dialogue pattern found
    for raw_line in story.splitlines():
        line = re.sub(r"^\s*\d+[\.\)]?\s*", "", raw_line).strip()

        for name in re.findall(r"\b[A-Z][A-Za-z\-']*\b", line):
            add_name(name)

    return names


def extract_question_characters(
    question: str,
    character_names: List[str],
) -> List[str]:
    found: List[tuple[int, str]] = []

    for name in character_names:
        m = re.search(rf"\b{re.escape(name)}\b", question)
        if m:
            found.append((m.start(), name))

    found.sort(key=lambda x: x[0])
    return [name for _, name in found]


def normalize_tom_type(value: Optional[str]) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def limit_question_characters_by_tom_type(
    question_characters: List[str],
    tom_type: Optional[str],
) -> List[str]:
    """
    first-order  => exactly/at most 1 relevant character
    second-order => exactly/at most 2 relevant characters
    """

    tom_type_norm = normalize_tom_type(tom_type)

    if "first-order" in tom_type_norm or "first order" in tom_type_norm:
        return question_characters[:1]

    if "second-order" in tom_type_norm or "second order" in tom_type_norm:
        return question_characters[:2]

    return question_characters


def is_truth_question(question_characters: List[str]) -> bool:
    return len(question_characters) == 0


def build_perspective_block(
    perspectives: Dict[str, str],
    question_characters: List[str],
) -> str:
    blocks: List[str] = []

    for character in question_characters:
        perspective = perspectives.get(character, "")
        blocks.append(
            f"""Perspective of {character}:
{perspective}"""
        )

    return "\n\n".join(blocks)


def eval_question(
    llm: LLMClient,
    context: dict,
    question: str,
    debug: bool = False,
    sim_model: Optional[LLMClient] = None,
) -> Tuple[str, str]:
    world = World(
        llm=llm,
        context=context,
        debug=debug,
        sim_model=sim_model,
    )

    world.parse_characters()

    raw_question_characters = extract_question_characters(
        question=question,
        character_names=world.agent_names,
    )

    tom_type = context.get("question_order")
    question_characters = limit_question_characters_by_tom_type(
        question_characters=raw_question_characters,
        tom_type=tom_type,
    )

    if debug:
        print("Question:", question)
        print("tom_type:", tom_type)
        print("Agent names:", world.agent_names)
        print("Raw question characters:", raw_question_characters)
        print("Limited question characters:", question_characters)

    if is_truth_question(question_characters):
        return world.eval_truth_question(question)

    for character in question_characters:
        world.take_perspective(character_name=character)

    world.setup_agents()

    answer, perspective = world.eval_multi_perspective_question(
        question=question,
        question_characters=question_characters,
    )

    return answer, perspective


def one_big_prompt(llm: LLMClient, story: str, question: str) -> Tuple[str, str]:
    agent_names = extract_characters_from_story(story)

    question_characters = extract_question_characters(
        question=question,
        character_names=agent_names,
    )

    if is_truth_question(question_characters):
        answer = llm.generate(
            f"""{story}

Based on the above information, answer the following question:
{question}

You must choose one of the above choices.
Do not output explanations.
Please output only the final choice starting with Answer:
For example: Answer: A
"""
        )
        return answer, "Truth Question"

    perspectives: Dict[str, str] = {}

    for character in question_characters:
        raw = llm.generate(
            PERSPECTIVE_PROMPT.format(
                story=story,
                character=character,
            )
        )

        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        perspectives[character] = raw.strip()

    perspective_block = build_perspective_block(
        perspectives=perspectives,
        question_characters=question_characters,
    )

    prompt = MULTI_PERSPECTIVE_QA_PROMPT.format(
        story=story,
        character_chain=" -> ".join(question_characters),
        perspective_block=perspective_block,
        question=question,
    )

    answer = llm.generate(prompt)
    return answer, prompt