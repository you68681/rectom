# Final gpt-5-nano release verification

Date: 2026-09-26  
Model: `gpt-5-nano`  
Provider: OpenAI Chat Completions API  
Dataset path: Hi-ToM  
Sample ID: `927`

## Result

| Directory | Calls | Total tokens | Cached tokens | Prediction | Gold | Result |
|---|---:|---:|---:|---|---|---|
| `final_release` | 8 | 9,665 | 3,072 | `red_box` | `red_box` | correct |

The state visibility mask is `true` for steps 1--5 and `false` for steps
6--15. The final Isla belief state is exactly
`["in(lettuce,red_box)"]`. There were no API errors, no unknown call-stage
labels, and no duplicated source step numbers.

The model's action-observation response still failed validation after two
repairs. The engine therefore used its exact Hi-ToM private/public visibility
fallback. Both `private_tell` and `public_claim` are subsequently filtered by
the deterministic exit-order trust rule before an action may update belief.

The release also passed all 20 unit tests before the temporary virtual
environment was deleted.

## Files

- `final_release/llm_trace.jsonl`: complete prompt and response for every LLM
  call, with public generation parameters and stage labels;
- `final_release/trace.md`: readable rendering of the complete LLM trace;
- `final_release/token_usage.csv`: per-call token usage;
- `final_release/predictions.jsonl`: parsed deltas, accumulated states/actions,
  masks, perspectives, prediction, and errors;
- `final_release/summary.json`: run summary.

No API key or authorization header is recorded.
