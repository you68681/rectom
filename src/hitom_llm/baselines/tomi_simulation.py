# # src/hitom_llm/tomi/sim_utils_tomi.py
#
# from __future__ import annotations
#
# from dataclasses import dataclass, field
# from typing import Dict, List, Optional, Protocol, Tuple
#
# from .tomi_prompts import (
#     PERSPECTIVE_PROMPT,
#     LLAMA_SIM_PROMPT,
#     GPT_SIM_PROMPT,
#     ABLATION_ONEPROMPT_SIM,
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

from .tomi_prompts import (
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

    def __post_init__(self) -> None:
        if "llama" in self.llm.model_name.lower():
            self.eval_prompt = LLAMA_SIM_PROMPT
        else:
            self.eval_prompt = GPT_SIM_PROMPT


@dataclass
class World:
    llm: LLMClient
    context: dict
    debug: bool = False
    sim_model: Optional[LLMClient] = None

    story: str = field(init=False)
    agent_names: List[str] = field(default_factory=list)
    perspectives: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.story = self.context["story"]
        if self.sim_model is None:
            self.sim_model = self.llm

    def parse_characters(self) -> None:
        self.agent_names = extract_characters_from_story(self.story)

        if self.debug:
            print("Agent names:", self.agent_names)

    def take_perspective(self, character_name: str) -> None:
        self.perspectives[character_name] = self.llm.generate(
            PERSPECTIVE_PROMPT.format(
                story=self.story,
                character=character_name,
            )
        ).strip()

        if self.debug:
            print(f"Perspective of {character_name}:", self.perspectives[character_name])

    def eval_truth_question(self, question: str) -> Tuple[str, str]:
        answer = self.llm.generate(
            f"""{self.story}

Based on the above information, answer the following question:
{question}

Answer using one of the given choices.
Please output the final choices starting with Answer:
For example Answer: A. red_crate
"""
        )
        return answer, "Truth"

    def eval_multi_perspective_question(
        self,
        question: str,
        question_characters: List[str],
    ) -> Tuple[str, str]:
        perspective_block = build_perspective_block(
            self.perspectives,
            question_characters,
        )

        prompt = MULTI_PERSPECTIVE_QA_PROMPT.format(
            story=self.story,
            character_chain=" -> ".join(question_characters),
            perspective_block=perspective_block,
            question=question,
        )

        answer = self.sim_model.generate(prompt)
        return answer, perspective_block


# ========================
# Character Extraction
# ========================

def extract_characters_from_story(story: str) -> List[str]:

    names = set()

    STOP_WORDS = {
            "The", "Where", "Question", "Choices", "Answer",
            "Story", "Choose", "Based", "And"
        }

    for line in story.splitlines():
            for name in re.findall(r"\b[A-Z][a-z]+\b", line):
                if name not in STOP_WORDS:
                    names.add(name)

    return list(names)

def extract_question_characters(question: str, character_names: List[str]) -> List[str]:
    found = []

    for name in character_names:
        m = re.search(rf"\b{name}\b", question)
        if m:
            found.append((m.start(), name))

    found.sort(key=lambda x: x[0])
    return [name for _, name in found]


def is_truth_question(question: str, question_characters: List[str]) -> bool:
    if len(question_characters) == 0:
        return True
    return False


def build_perspective_block(
    perspectives: Dict[str, str],
    question_characters: List[str],
) -> str:
    blocks = []

    for c in question_characters:
        p = perspectives.get(c, "")
        blocks.append(f"Perspective of {c}:\n{p}")

    return "\n\n".join(blocks)


# ========================
# MAIN ENTRY
# ========================

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

    question_characters = extract_question_characters(
        question,
        world.agent_names,
    )

    if debug:
        print("Question:", question)
        print("Characters:", world.agent_names)
        print("In question:", question_characters)

    # Truth question
    if is_truth_question(question, question_characters):
        return world.eval_truth_question(question)

    # Build perspective for ALL characters in question
    for c in question_characters:
        world.take_perspective(c)

    # Final multi-perspective reasoning
    return world.eval_multi_perspective_question(
        question,
        question_characters,
    )


# ========================
# One prompt version
# ========================

def one_big_prompt(llm: LLMClient, story: str, question: str):
    characters = extract_characters_from_story(story)
    q_chars = extract_question_characters(question, characters)

    if is_truth_question(question, q_chars):
        return llm.generate(f"{story}\n\n{question}"), "Truth"

    perspectives = {}

    for c in q_chars:
        perspectives[c] = llm.generate(
            PERSPECTIVE_PROMPT.format(
                story=story,
                character=c,
            )
        )

    perspective_block = build_perspective_block(perspectives, q_chars)

    prompt = MULTI_PERSPECTIVE_QA_PROMPT.format(
        story=story,
        character_chain=" -> ".join(q_chars),
        perspective_block=perspective_block,
        question=question,
    )

    answer = llm.generate(prompt)

    return answer, prompt