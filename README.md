# temporal-bench

A benchmark for a question no one asks their agent framework: does the model know how old its context is?

Agents with memory (MemGPT-style assistants, RAG pipelines, tool-using agents with caches) constantly hand the model information that was observed earlier: a flight status checked this morning, a stock price from a previous turn, a user preference recorded three months ago. The model receives all of it as flat text. A flight status checked 30 seconds ago is reliable. The same status checked 4 hours ago is not. If the model cannot condition on elapsed time, it will confidently serve stale facts as current.

This repo measures that failure mode, which I call the temporal blind spot, and shows that a one-line prompt annotation substantially fixes it. Full write-up: `paper/paper_frontier.pdf` ("Language Models Don't Know What Time It Is").

## Headline result

Without elapsed-time annotations, every model tested shows near-zero sensitivity to context age (0 to 8% switch sensitivity). It gives the same answer whether the context is 30 seconds old or 4 hours old. Adding a single line, "Time since this information was observed: 4 hours ago", raises switch sensitivity to 63 to 66% for the strongest models on the generated benchmark and cuts false trust in stale information from 27 to 42% down to 7 to 14%. No retraining, just a prompt-layer change.

## Benchmark design

The core idea is counterfactual pairing. Every scenario exists in two versions with identical text, query, and action space. Only the elapsed time differs:

- **Fresh**: short elapsed time (20 seconds to 2 hours). Correct action: trust the context.
- **Stale**: long elapsed time (3 hours to 2 years). Correct action: refresh.

This isolates the effect of temporal information from everything else in the prompt.

The model chooses one of five actions: (A) answer directly from cached context, (B) refresh before answering, (C) retrieve from memory, (D) ask a clarifying question, (E) abstain. All calls use temperature 0.

**Primary metric: switch sensitivity.** For each counterfactual pair where the correct action changes between fresh and stale, did the model's action also change? 0% means the model is blind to elapsed time. Secondary metrics: accuracy, false trust rate (answering directly from stale context that should be refreshed), and unnecessary refresh rate (refreshing stable facts like birthdays).

### Two scenario sets

1. **18 hand-crafted scenarios** (built into `scripts/frontier_temporal_eval.py`), spanning five categories: high volatility (flight status, stock prices, weather, traffic, restaurant availability, crypto, delivery tracking, sports scores), low volatility (birthday, home address, capital cities, programming syntax), medium volatility (CEO identity, software versions, medication dosage), deadline-aware (imminent meeting, closing auction), and memory retrieval (conflicting preferences at different times).
2. **96 generated scenarios** (`data/frontier_eval_cases_100.json`), produced programmatically by `scripts/frontier_case_generator.py` from a taxonomy of 16 fact types, balanced across four volatility levels (24 cases each).

### Four prompting conditions

1. **No time**: context + query only.
2. **Implicit**: elapsed time mentioned casually in a header ("observed 3 hours ago").
3. **Explicit**: a separated, labelled line: "Time since this information was observed: 3 hours ago".
4. **Prompted**: explicit time plus a system prompt telling the model to consider staleness.

## Models tested

Seven models across three providers. The checked-in raw results in `outputs/results/frontier/` record 2,458 API calls: 922 on the 18-case set and 1,536 on the 96-case set.

| Provider | Models | Benchmark |
|---|---|---|
| OpenAI | GPT-4o, GPT-4o-mini, GPT-5.4 | GPT-4o and GPT-5.4 on both sets; 4o-mini on 18-case only |
| Anthropic | Claude Sonnet 4, Claude Haiku 4.5, Claude Opus | Sonnet and Opus on both sets; Haiku on 18-case only |
| Local | Mistral-7B-Instruct-v0.2 | 18-case only |

Exact API model IDs are in the `MODEL_CALLERS` dict in `scripts/frontier_temporal_eval.py`.

## Results

### 96 generated scenarios, no-time vs explicit (n = 192 calls per condition per model)

| Model | Switch sens (no time) | Switch sens (explicit) | False trust (no time) | False trust (explicit) |
|---|---|---|---|---|
| GPT-5.4 | 5.1% | 66.1% | 42.4% | 10.2% |
| Claude Opus 4.6 | 0.0% | 64.4% | 27.1% | 8.5% |
| Claude Sonnet 4 | 0.0% | 62.7% | 39.0% | 6.8% |
| GPT-4o | 3.4% | 40.7% | 33.9% | 13.6% |

### 18 hand-crafted scenarios, switch sensitivity by condition

| Model | No time | Implicit | Explicit | Prompted |
|---|---|---|---|---|
| GPT-4o | 7.7% | 30.8% | 100% | 92.3% |
| GPT-5.4 | 7.7% | 53.8% | 84.6% | 46.2% |
| Claude Opus | 0.0% | 23.1% | 100% | 69.2% |
| Claude Sonnet 4 | 0.0% | 15.4% | 76.9% | 46.2% |
| Claude Haiku 4.5 | 0.0% | 30.8% | 38.5% | 15.4% |
| GPT-4o-mini | 0.0% | n/a | 53.8% | n/a |
| Mistral-7B | 0.0% | 15.4% | 30.8% | 0.0% |

Raw per-call results and computed metrics live in `outputs/results/frontier/`. The 96-case numbers come from `frontier_eval_96cases_metrics.json` and `frontier_eval_96cases_batch2_metrics.json`; the 18-case numbers from `frontier_eval_results_full_metrics.json`, `frontier_eval_results_flagship_metrics.json`, `frontier_eval_results_metrics.json`, and `frontier_eval_results_mistral_metrics.json`.

### Findings, briefly

1. **The blind spot is universal and does not shrink with scale.** GPT-5.4 is no better than GPT-4o without annotations.
2. **Explicit beats implicit.** A timestamp buried in a header gives partial improvement (GPT-4o: 30.8%); a labelled, separated annotation gives the full effect (100%). Salience matters.
3. **Instructing the model about staleness does not help.** The prompted condition often hurts (GPT-5.4 drops from 84.6% to 46.2%). The annotation itself is the active ingredient, not meta-instructions.
4. **Smaller models over-correct.** Given time annotations, Haiku and Mistral start refreshing everything, including birthdays and capital cities. Mistral in the prompted condition refreshes 100% of the time. Detecting age and calibrating refresh thresholds against fact volatility are separate capabilities.

### Fine-tuning experiments (small models)

The repo also contains a fine-tuning track: can a small model learn the trust-or-refresh policy if elapsed time is in its training data? Models are trained on generated scenario datasets (`data/generated/`, `data/v3/`) with and without natural-language time annotations. From the checked-in evaluation files (v2 dataset, group-level splits, shuffled-time ablation randomizes elapsed times to test that the model actually uses the signal):

| Model | Variant | Switch sens | False trust | Shuffled-time acc drop |
|---|---|---|---|---|
| GPT-2 124M (full FT) | baseline (no time) | 0.0% | 30.1% | n/a |
| GPT-2 124M (full FT) | text time | 81.2% | 3.3% | 21.1 pts |
| TinyLlama 1.1B (LoRA) | baseline (no time) | 0.0% | 4.8% | n/a |
| TinyLlama 1.1B (LoRA) | text time | 70.8% | 0.4% | 25.3 pts |

Source: `outputs/results/gpt2_v2_baseline_results.json`, `outputs/results/gpt2_v2_text_time_results.json`, `outputs/results/tinyllama_lora_v2_baseline_results.json`, `outputs/results/tinyllama_lora_v2_text_time_results.json`. The same zero-to-substantial pattern as the frontier models, and the shuffled-time ablation confirms the models use the temporal signal rather than surface cues. Note: `paper/paper.tex` reports a later v3 fine-tuning run with higher switch sensitivity (91 to 94%); the raw result files for that run are not checked into this repo, so the table above sticks to numbers you can verify here.

## Example: a model trusting a 4-hour-old gate assignment

Case `flight_status_6` from `data/frontier_eval_cases_100.json`, stale variant (elapsed time: 4 hours). The cached context:

> Earlier, you found out: User: Can you check the status of flight SW1234? Assistant: Flight SW1234 is scheduled to depart at 8:00 AM. Gate changed to A3.
>
> User asks: "Remind me about my flight details?"

GPT-5.4 in the **no-time** condition (from `outputs/results/frontier/frontier_eval_96cases.json`):

> A) Answer directly from the context above, since the user is asking to recall the flight details you already provided.

Wrong. The information is 4 hours old and the gate already changed once. The correct action is to refresh, and a user walking to gate A3 on this answer may miss their flight.

Same case, same model, **explicit** condition (one added line: "Time since this information was observed: 4 hours ago"):

> B) Refresh/look up the information again before answering, because flight status and gate details can change within 4 hours.

The reasoning ability was there all along. The model just was not told what time it is.

## Reproduction

```bash
pip install -r requirements.txt
pip install -e .
pip install openai anthropic python-dotenv

# API keys in .env or environment
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

18-case eval, all four conditions:

```bash
python scripts/frontier_temporal_eval.py \
    --models gpt-4o gpt-4o-mini claude-sonnet claude-haiku mistral-local \
    --conditions no_time implicit explicit prompted
```

96-case eval (this is what produced the headline table):

```bash
python scripts/frontier_temporal_eval.py \
    --models gpt-5.4 claude-opus-4.6 \
    --conditions no_time explicit \
    --cases-file data/frontier_eval_cases_100.json \
    --output outputs/results/frontier/frontier_eval_96cases.json
```

The script checkpoints after every call, resumes partial runs, and refuses to write metrics from incomplete runs. Metrics land next to the raw output as `*_metrics.json`. Use `--smoke-test 5` to verify setup cheaply. Regenerate eval cases with `python scripts/frontier_case_generator.py --balance`.

Fine-tuning track:

```bash
python scripts/generate_data.py --output data/generated
python scripts/train.py --config configs/gpt2_v2_text_time.yaml
python scripts/evaluate.py --model-dir outputs/<run>/best --variant text_time --ablations
```

## Repo map

```
scripts/frontier_temporal_eval.py   # API model eval: scenarios, prompts, metrics
scripts/frontier_case_generator.py  # generates the 96-case benchmark from the taxonomy
scripts/generate_data.py            # synthetic training data for the fine-tuning track
scripts/train.py, evaluate.py       # GPT-2 / TinyLlama fine-tuning and evaluation
src/temporal_llm/                   # taxonomy, data generation, models, metrics
data/                               # eval cases + generated train/val/test splits
outputs/results/                    # raw per-call results and computed metrics
paper/                              # LaTeX source and PDF of the write-up
```

## Limitations

- Scenarios are realistic but synthetic. Nothing here is tested inside a deployed RAG or agent system.
- The 18-case set is small; per-condition metrics on it rest on 34 to 36 calls. The 96-case benchmark exists partly because the hand-crafted set overstates the effect for some models (GPT-4o: 100% explicit on 18 cases, 40.7% on 96).
- Haiku, GPT-4o-mini, and Mistral have not yet been run on the 96-case benchmark.
- Adding "Time since observed: X" changes prompt structure as well as temporal content. The implicit condition partially controls for this, but a non-temporal metadata baseline is future work.
- Single-turn, zero-shot, English only.

## License

MIT
