# src/hitom_llm/tomi/prompts_tomi.py

MULTI_PERSPECTIVE_QA_PROMPT = """\
You are answering a Theory-of-Mind multiple-choice question in a dialogue setting.

The original dialogue is provided only for reference:
{story}

The question mentions the following character chain:
{character_chain}

Below are the generated perspectives of the relevant character(s).
Use these perspectives to answer the question.

Important:
- For first-order questions, use only the single target character's perspective.
- For second-order questions, reason from the first character's perspective about the second character's belief.
- Do not ignore any provided perspective.

{perspective_block}

Question:
{question}

You must choose one of the above choices.
Do not output explanations.
Please output the final choices starting with Answer:
For example Answer: A. Alissa discusses gaming. 
"""

PERSPECTIVE_PROMPT = """\
The following is a sequence of events about some characters, that takes place in multiple locations.
Your job is to output only the events that the specified character, {character}, knows about.
You should assume the following.
(1) Conversations involve multiple characters and are mostly casual (small talk).
(2) Participants can join or leave the conversation at different points.
(3) While a character is absent, the conversation continues and new information may be introduced.
(4) Returning characters do not know what was discussed during their absence.

Story:
{story}

What events does {character} know about? Only output the events according to the above rules, do not provide an explanation."""

LLAMA_SIM_PROMPT = """\
{perspective}
You are {name}.
Based on the above information, answer the following question:
{question}
You must choose one of the above choices.
Do not output explanations.
Please output the final choices starting with Answer:
For example Answer: A. Alissa discusses gaming. 
"""

GPT_SIM_PROMPT = """\
{perspective}
You are {name}.
Based on the above information, answer the following question:
{question}
You must choose one of the above choices.
Do not output explanations.
Please output the final choices starting with Answer:
For example Answer: A. Alissa discusses gaming. 
"""


# baselinePrompt = """\
# {story}
# {question}
# Choose from the following:
# {containers_0}, {containers_1}
# """

baselinePrompt = """\
{story}
{question}
Choose from the following:
{choices}
"""

# questionPrompt = """\
# {question}
# Choose from the following:
# {containers_0}, {containers_1}
# """

questionPrompt = """\
{question}
Choose from the following:
{choices}
"""



llamaprompt = """\
{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
Keep your answer concise. Answer with a single word.

The answer is
"""

llamaCotPrompt = """
{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
Reason step by step before answering in 'Thought: Let's think step by step'. Write your final answer as 'Answer: <answer>'. Answer with a single word.

Thought: Let's think step by step.
"""