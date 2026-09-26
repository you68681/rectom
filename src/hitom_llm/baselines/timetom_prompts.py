from __future__ import annotations

READING_TEMPORAL_SPACE_PROMPT = """\
The following is a story. Your task is to add timeline to the story.
Here is one rule: Each sentence corresponds to a moment t. Use \n as a delimiter, and the timeline is t1, t2, ... , tN.

Story:
{story}

Only output the story with the added timeline, do not provide explanations.

Example:
Input story:
John entered the kitchen.
The apple is on the table.
John moved the apple to the drawer.

Output:
t1: John entered the kitchen.
t2: The apple is on the table.
t3: John moved the apple to the drawer.
"""

DIALOGUE_TEMPORAL_SPACE_PROMPT = """\
The following is a dialogue. Your task is to add timeline to the dialogue.
Here is one rule: Each utterance spoken by a character corresponds to a moment t. Use \n as a delimiter, and the timeline is t1, t2, ... , tN.

Dialogue:
{dialogue}

Only output the dialogue content with the added timeline, do not provide explanations.

Example:
Input dialogue:
Sara: Hi, how are you?
Javier: I'm good, thanks.
Sara: Glad to hear that.

Correct output:
t1: Sara: Hi, how are you?
t2: Javier: I'm good, thanks.
t3: Sara: Glad to hear that.
"""

READING_TBSC_PROMPT = """\
The following is a sequence of events with a timeline about some characters, that takes place in multiple locations.
Your job is to output only the events on the timeline that character {character} can aware of.

Here are a few commonsense rules:
1. If a character is in a certain room/location, they will be aware of all other events happening in that room.
This includes other characters entering or leaving the location, the locations of objects within it, and whether someone has moved an object to another location.
2. If a character leaves a location and is no longer there, they will no longer be aware of any events occurring at that location. However, they can re-enter the location.
3. A character is aware of all the events that they do.

Story:
{story}

What events on the timeline does {character} aware of?
Only output the events (include both time step and following event) according to the above rules, do not provide an explanation.

Output format example only:
t3: Avery moved the lettuce to the green_bathtub.
t4: Elizabeth dislikes the tangerine.
"""

DIALOGUE_TBSC_PROMPT = """\
The following is a dialogue with a timeline between multiple characters.
Your task is to only output the dialogue content on the timeline that the character {character} can aware of.

Here are two rules:
1. If a character leaves the conversation to do something else and then back after a few rounds of dialogue, they are unaware of the content of the conversation that took place during their absence, but they aware of the content of the conversation besides their absence.
2. If a character doesn't leave the conversation to do something else and then back after a few rounds of dialogue, they are aware of all the content of dialogue with all timeline.

Dialogue:
{dialogue}

What dialogue content on the timeline does {character} aware of?
Only output the dialogue content  (include both time step and following dialogue content)  according to the above rules, do not provide an explanation.

Output format example only:
t2 Sara: Sure thing, Gianna. Take care!
t4 Javier: Have you ever tried training Bruno?
"""

READING_BELIEF_COMPRESSION_PROMPT = """\
Belief Compression: The following is information from the perspective of the character, {character}.

Perspective:
{perspective}

Output the remaining perspective information (the events includes both time step and following statement) after removing the events of characters enter or leave / exit the room / location, do not provide an explanation.

Output format example only:
t2: The lettuce is in the green_drawer.
t3: Avery moved the lettuce to the green_bathtub.
t7: Owen likes the green_envelope.
"""

READING_FIRST_ORDER_QA_PROMPT = """\
Time-Aware Belief Question Answer:
{perspective}

You are {name}.
Based on the above information, answer the following question:
{question}

Keep your answer concise, one sentence is enough.
You must choose one of the above choices.

Choices: {choices}
"""

READING_HIGHER_ORDER_QA_PROMPT = """\
{perspective}

You are {name}.
Based on the above information, answer the following question:
{question}

Keep your answer concise, one sentence is enough.
You must choose one of the above choices.

Choices: {choices}
"""

DIALOGUE_FIRST_ORDER_QA_PROMPT = """\
The following is the belief states chain of character {name}. This is the content known to {name}:
[{perspective}]

You are {name}.
Based on the above information, answer the following question:
{question}

When answering questions, based on own belief, simply focus on the information of things asked in the question and ignore other distracting factors.
You must choose one of the above choices.

Choices: {choices}
"""

DIALOGUE_HIGHER_ORDER_QA_PROMPT = """\
The following is the belief states chain of character {name}. This is the content known to {name}:
[{perspective}]

You are {name}.
Based on the above information, answer the following question:
{question}

You must choose one of the above choices.

Choices: {choices}
"""

READING_SOLVER_REDUCED_QA_PROMPT = """\
The following is the event corresponding to the period of belief communication:
{common_belief}

Based only on the above information, answer the following question:
{reduced_question}

Choices: {choices}

Keep your answer concise, one sentence is enough.
You must choose one of the above choices.
"""

READING_SOLVER_FEEDBACK_PROMPT = """\
Perspective1:
{perspective}

You are {name}.
Based on the above information, answer the following question:
{question}

Choices: {choices}
Answer1: {answer1}

Feedback Perspective2: The event corresponding to the period of belief communication for the belief chain ({belief_chain_text}), focusing on the transition from {question_subject} to {remaining_chain_text}:
{common_belief}

Based on this information, the answer we get to the reduced question:
{reduced_question}
is
Answer2: {answer2}

Consider Perspective1, Feedback Perspective2 and their answers, answer the original question again:
{question}

Keep your answer concise, one sentence is enough.
You must choose one of the above choices.
"""

DIALOGUE_SOLVER_REDUCED_QA_PROMPT = """\
The following is the event corresponding to the period of belief communication:
{common_belief}

Based only on the above information, answer the following question:
{reduced_question}

Choices: {choices}

Keep your answer concise, one sentence is enough.
You must choose one of the above choices.
"""

DIALOGUE_SOLVER_FEEDBACK_PROMPT = """\
The following is the belief states chain of character {name}. This is the content known to {name}:
{perspective}

You are {name}.
Based on the above information, answer the following question:
{question}

Choices: {choices}
Answer: {answer1}

Feedback: The event corresponding to the period of belief communication for the belief chain ({belief_chain_text}), focusing on the transition from {question_subject} to {remaining_chain_text}:
{common_belief}

Based on this information, the answer we get to the reduced question:
{reduced_question}
is {answer2}

Considering this feedback, answer the original question again:
{question}

Keep your answer concise, one sentence is enough.
You must choose one of the above choices.
"""
