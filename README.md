# Mind the Perspective: Let's Reason Recursively for Theory of Mind

Official repository for **Mind the Perspective: Let's Reason Recursively for
Theory of Mind**, accepted to **Findings of EMNLP 2026**.

This repository contains the official implementation of **RecToM**, the
recursive perspective-reasoning method introduced in the paper.

## Paper

**Chao Lei, Guang Hu, Meng Yang, Yanbei Jiang, and Nir Lipovetzky**  
*Mind the Perspective: Let's Reason Recursively for Theory of Mind*  
Findings of the 2026 Conference on Empirical Methods in Natural Language
Processing (EMNLP 2026 Findings)

- arXiv: https://arxiv.org/abs/2606.11724

The implementation contains the Hi-ToM, FanToM, and Big-ToM RECTOM pipelines,
including validation, repair, retry, and logging. The
main RECTOM engines use two aligned tracks:

- a persistent state track containing the accumulated symbolic world state;
- a transient action track containing communication, claim, and question
  events.

The two lists are index-aligned and are the code-level representation of the
paper's sequence of `(state_t, event_t)` pairs.  Transient actions retain their
original order and are applied as a final ordered batch after recursive
observability filtering.  This is result-equivalent for the released Hi-ToM
tasks, where belief-relevant persistent transitions do not follow the
communication phase.

## RECTOM pipeline

1. Parse every story into an ordered event sequence.
2. Ask the selected LLM for step-aligned symbolic deltas.
3. Validate indices, exact source text, characters, facts, and state
   consistency; repair invalid model output when possible. Targeted Hi-ToM
   repairs receive the complete ordered story, the complete current delta
   sequence, the initial model output, every prior repair response across all
   repair rounds, and the accumulated state before the target step.
4. Accumulate persistent state deterministically with
   `new_state = (previous_state - removed_facts) union added_facts`.
5. Keep transient communication off the global ontic state.
6. Extract the ordered character chain from the benchmark question.
7. Recursively construct each character's state perspective from the previous
   perspective:
   - retain an observable state;
   - otherwise inherit the preceding completed state;
   - when the state is hidden, apply universally observable Hi-ToM room-entry
     and room-exit deltas;
   - never fill an earlier state from a future observation.
8. Recursively filter the aligned transient-action track.
9. Apply Hi-ToM listener/trust rules deterministically and apply surviving
   transient actions in chronological order.
10. Ask the LLM to reduce a nested question to its zero-order form, validate
    the reduction, and answer from the final constructed belief state.

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Credentials and endpoints

No credentials are stored in the repository.

OpenAI or an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="..."
# Optional for vLLM or another compatible server:
export OPENAI_BASE_URL="http://localhost:8000/v1"
# Optional explicit override for reasoning models (for example, minimal):
export OPENAI_REASONING_EFFORT="minimal"
```

Gemini:

```bash
export GEMINI_API_KEY="..."
```

Token logging is disabled unless a destination is explicitly configured:

```bash
export TOKEN_USAGE_PATH="./outputs/token_usage.csv"
```

Full prompt/response tracing is also opt-in. Each JSONL record contains the
provider, model, public generation parameters, call type, complete prompt,
complete response, retry attempt, and any error. Credentials are never passed
to the trace logger:

```bash
export LLM_TRACE_PATH="./outputs/llm_trace.jsonl"
```

Any credentials that previously appeared in a development copy of this
project must be revoked before publishing that copy.

## Model settings

The release client leaves GPT-5.4 and Gemini sampling parameters at provider
defaults.  For model names containing `qwen`, it uses temperature `1.0`, top-p
`0.95`, and top-k `20`.  For names containing `gemma`, it uses temperature
`1.0`, top-p `0.95`, and top-k `64`.

## Running RECTOM

Hi-ToM:

```bash
rectom run \
  --data-path ./Hi-ToM_data.json \
  --output-dir ./outputs/hitom_gpt54 \
  --provider openai \
  --model gpt-5.4 \
  --dataset-type hitom \
  --max-workers 4
```

FanToM:

```bash
rectom run \
  --data-path ./fantom.json \
  --output-dir ./outputs/fantom_gpt54 \
  --provider openai \
  --model gpt-5.4 \
  --dataset-type fantom \
  --max-workers 4
```

Big-ToM true-belief and false-belief subsets are run separately:

```bash
rectom run \
  --data-path ./conditions/0_forward_belief_true_belief/stories.csv \
  --output-dir ./outputs/bigtom_true_gpt54 \
  --provider openai \
  --model gpt-5.4 \
  --dataset-type bigtom \
  --max-workers 4

rectom run \
  --data-path ./conditions/1_forward_belief_false_belief/stories.csv \
  --output-dir ./outputs/bigtom_false_gpt54 \
  --provider openai \
  --model gpt-5.4 \
  --dataset-type bigtom \
  --max-workers 4
```

Use `--sample-limit` only for debugging.  The publication runs use the full
selected benchmark subsets.

## Tests

The core tests do not call an external model:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

They cover forward-only completion, the Figure 2 room-transition behavior,
transient/global-state separation, nested symbolic predicates, deterministic
Hi-ToM trust, and FanToM participation/communication merging.

## Citation

```bibtex
@inproceedings{lei2026rectom,
  title={Mind the Perspective: Let's Reason Recursively for Theory of Mind},
  author={Lei, Chao and Hu, Guang and Yang, Meng and Jiang, Yanbei and Lipovetzky, Nir},
  booktitle={Findings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026}
}
```

## Outputs

Each run writes:

- `predictions.jsonl`, including symbolic deltas, aligned state/action tracks,
  observation masks, completed perspectives, prediction, and validation
  errors;
- `summary.json`, including processed counts, accuracy, provider, and model.
