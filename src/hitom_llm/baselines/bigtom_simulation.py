# src/hitom_llm/baselines/bigtom_simulation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

from .bigtom_prompts import (
    GPT_PERSPECTIVE_PROMPT,
    GPT_SIM_PROMPT,
    LLAMA_PERSPECTIVE_PROMPT,
    LLAMA_SIM_PROMPT,
    ABLATION_ONEPROMPT_SIM,
)


class LLMClient(Protocol):
    """Minimal interface expected by the BigToM baseline."""

    @property
    def model_name(self) -> str:
        ...

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class Agent:
    """Agent class for simulation."""

    llm: LLMClient
    name: str = ""
    perspective: str = ""
    debug: bool = False
    was_asked: Optional[str] = None
    replied: Optional[str] = None

    def __post_init__(self) -> None:
        if "llama" in self.llm.model_name.lower():
            self.eval_prompt = LLAMA_SIM_PROMPT
        else:
            self.eval_prompt = GPT_SIM_PROMPT

    def eval_question(self, question: str) -> str:
        prompt = self.eval_prompt.format(
            perspective=self.perspective,
            name=self.name,
            question=question,
        )
        choice = self.llm.generate(prompt)
        print(f"choice:{choice}")
        self.was_asked = prompt
        self.replied = choice
        return choice


@dataclass
class World:
    """World class that keeps track of true world state."""

    llm: LLMClient
    context: str
    agent: Optional[Agent] = None
    perspective: Optional[str] = None
    knows_change: Optional[bool] = None
    sim_model: Optional[LLMClient] = None
    agent_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.sim_model is None:
            self.sim_model = self.llm

    def take_perspective(self) -> None:
        """Function that takes perspective."""
        self.agent_name = self.context.split(" ")[0]

        story = self.context
        name = self.agent_name

        prompt_template = GPT_PERSPECTIVE_PROMPT
        if "llama" in self.sim_model.model_name.lower():
            prompt_template = LLAMA_PERSPECTIVE_PROMPT

        prompt = prompt_template.format(story=story, name=name)
        perspective = self.sim_model.generate(prompt).split(":")[-1].strip()

        print(f"perspective: {perspective}")

        if self.perspective is None:
            self.perspective = perspective

    def setup_agent(self) -> None:
        if self.agent is None:
            self.agent = Agent(self.sim_model, "")

        self.agent.name = self.agent_name or ""
        self.agent.perspective = self.perspective or ""

    def get_world_state(self) -> str:
        return f"""\
Agent Name: {self.agent_name}

Context: {self.context}

Perspective: {self.perspective}
"""

    def get_agent_state(self) -> str:
        if self.agent is None:
            return "Agent not initialized."

        return f"""\
Agent was prompted:
{self.agent.was_asked}

and outputted:
{self.agent.replied}
"""


def eval_question(
    llm: LLMClient,
    context: str,
    question: str,
    knows_change: bool,
    perspective_gold: bool,
    sim_model: Optional[LLMClient] = None,
) -> Tuple[str, World]:
    """Evaluation function matching the original baseline logic."""
    if perspective_gold:
        perspective = ".".join(context.split(".")[:-3]) + "."
        knows_change = knows_change
    else:
        perspective = None
        knows_change = None

    world = World(
        llm=llm,
        context=context,
        perspective=perspective,
        knows_change=knows_change,
        sim_model=sim_model,
    )
    world.take_perspective()
    world.setup_agent()
    assert world.agent is not None
    answer = world.agent.eval_question(question)
    return answer, world


def one_big_prompt(llm: LLMClient, story: str, question: str) -> Tuple[str, str]:
    """One-prompt ablation baseline."""
    name = story.split()[0]
    prompt = ABLATION_ONEPROMPT_SIM.format(
        name=name,
        story=story,
        question=question,
    )
    output = llm.generate(prompt)

    marker = "Answer:"
    if marker in output:
        answer = output[output.rfind(marker) + len(marker):].strip()
    else:
        answer = output.strip()

    return answer, output