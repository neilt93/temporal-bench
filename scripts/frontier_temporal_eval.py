"""
Frontier Model Temporal Reasoning Evaluation
=============================================

Tests whether frontier LLMs (GPT-4, Claude, Mistral, etc.) can detect
stale context and make appropriate refresh/trust decisions — with and
without explicit elapsed-time annotations.

Design:
  - Same counterfactual structure as the fine-tuning experiments
  - Each test case has TWO versions: short elapsed time vs long elapsed time
  - Models are tested in 4 conditions:
      1. NO_TIME:   no temporal info at all
      2. IMPLICIT:  timestamps present but no explicit "elapsed time" annotation
      3. EXPLICIT:  "Time since last info: 3 hours ago" added
      4. PROMPTED:  explicit time + system prompt telling model to consider staleness

  - Key metrics:
      * Switch sensitivity: does the model change its answer across the counterfactual pair?
      * Action accuracy: does it pick the right action (trust vs refresh vs retrieve etc.)?
      * False trust rate: how often does it say "answer directly" when info is stale?
      * Calibration: does confidence/hedging language change with elapsed time?

Usage:
  pip install openai anthropic python-dotenv transformers torch
  # Set keys in .env: OPENAI_API_KEY=..., ANTHROPIC_API_KEY=...
  python frontier_temporal_eval.py

  # Quick test with GPT-4o-mini (cheapest, ~$0.50)
  python frontier_temporal_eval.py --models gpt-4o-mini --conditions no_time explicit

  # Full eval across all models and conditions
  python frontier_temporal_eval.py \
    --models gpt-4o gpt-4o-mini claude-sonnet claude-haiku mistral-local \
    --conditions no_time implicit explicit prompted

  # Local Mistral only (no API keys needed)
  python frontier_temporal_eval.py --models mistral-local --conditions no_time explicit
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# TEST CASES — counterfactual pairs
# ============================================================

TEST_CASES = [
    # --- HIGH VOLATILITY: should refresh when stale ---
    {
        "id": "flight_status",
        "category": "high_volatility",
        "context": "You previously looked up the user's flight. Flight AA123 departs at 6:10 PM from Gate B22.",
        "query": "What gate is my flight departing from?",
        "fresh_time": "30 seconds ago",
        "stale_time": "4 hours ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "refresh",
        "fact_type": "flight_status",
    },
    {
        "id": "stock_price",
        "category": "high_volatility",
        "context": "You checked the stock price. AAPL is trading at $187.32, up 1.2% today.",
        "query": "What's Apple's stock price right now?",
        "fresh_time": "2 minutes ago",
        "stale_time": "6 hours ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "refresh",
        "fact_type": "stock_price",
    },
    {
        "id": "weather",
        "category": "high_volatility",
        "context": "You checked the weather. It's currently 72°F and sunny in San Francisco.",
        "query": "Do I need an umbrella today?",
        "fresh_time": "15 minutes ago",
        "stale_time": "2 days ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "refresh",
        "fact_type": "weather",
    },
    {
        "id": "traffic",
        "category": "high_volatility",
        "context": "You checked the route. The drive from your home to the airport takes about 35 minutes with current traffic.",
        "query": "How long will it take me to get to the airport?",
        "fresh_time": "5 minutes ago",
        "stale_time": "3 hours ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "refresh",
        "fact_type": "traffic",
    },
    {
        "id": "restaurant_availability",
        "category": "high_volatility",
        "context": "You called the restaurant. They have a table available for 2 at 7:30 PM tonight.",
        "query": "Can we still get that table tonight?",
        "fresh_time": "1 minute ago",
        "stale_time": "5 hours ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "refresh",
        "fact_type": "reservation",
    },
    {
        "id": "crypto_price",
        "category": "high_volatility",
        "context": "You checked the price. Bitcoin is at $67,234, down 3.4% in the last hour.",
        "query": "What's Bitcoin at?",
        "fresh_time": "45 seconds ago",
        "stale_time": "8 hours ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "refresh",
        "fact_type": "crypto_price",
    },
    {
        "id": "delivery_status",
        "category": "high_volatility",
        "context": "You tracked the package. Your Amazon order is out for delivery and 3 stops away.",
        "query": "Where's my package?",
        "fresh_time": "2 minutes ago",
        "stale_time": "1 day ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "refresh",
        "fact_type": "delivery_status",
    },
    {
        "id": "sports_score",
        "category": "high_volatility",
        "context": "You checked the score. The Lakers are up 87-82 in the 3rd quarter.",
        "query": "What's the score of the Lakers game?",
        "fresh_time": "1 minute ago",
        "stale_time": "3 hours ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "refresh",
        "fact_type": "sports_score",
    },

    # --- LOW VOLATILITY: should trust even when older ---
    {
        "id": "birthday",
        "category": "low_volatility",
        "context": "The user mentioned their birthday is March 15th.",
        "query": "When is my birthday?",
        "fresh_time": "1 hour ago",
        "stale_time": "6 months ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "answer_directly",  # stable fact — should still trust
        "fact_type": "personal_stable",
    },
    {
        "id": "home_address",
        "category": "low_volatility",
        "context": "The user said they live at 742 Evergreen Terrace, Springfield.",
        "query": "What's my home address?",
        "fresh_time": "2 hours ago",
        "stale_time": "3 months ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "answer_directly",
        "fact_type": "personal_stable",
    },
    {
        "id": "capital_city",
        "category": "low_volatility",
        "context": "You looked it up. The capital of France is Paris.",
        "query": "What's the capital of France?",
        "fresh_time": "10 minutes ago",
        "stale_time": "1 year ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "answer_directly",
        "fact_type": "world_knowledge_stable",
    },
    {
        "id": "programming_syntax",
        "category": "low_volatility",
        "context": "You checked the docs. In Python, you use 'len()' to get the length of a list.",
        "query": "How do I get the length of a list in Python?",
        "fresh_time": "5 minutes ago",
        "stale_time": "2 years ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "answer_directly",
        "fact_type": "technical_stable",
    },

    # --- MEDIUM VOLATILITY: judgment calls ---
    {
        "id": "ceo_identity",
        "category": "medium_volatility",
        "context": "You looked it up. The CEO of Twitter is Elon Musk.",
        "query": "Who runs Twitter?",
        "fresh_time": "1 day ago",
        "stale_time": "1 year ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "refresh",
        "fact_type": "role_position",
    },
    {
        "id": "software_version",
        "category": "medium_volatility",
        "context": "You checked. The latest version of Python is 3.12.1.",
        "query": "What's the latest Python version?",
        "fresh_time": "2 days ago",
        "stale_time": "8 months ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "refresh",
        "fact_type": "software_version",
    },
    {
        "id": "medication_dose",
        "category": "medium_volatility",
        "context": "The user's doctor prescribed 10mg of lisinopril daily for blood pressure.",
        "query": "What's my blood pressure medication dose?",
        "fresh_time": "1 week ago",
        "stale_time": "1 year ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "refresh",
        "fact_type": "medical",
    },

    # --- DEADLINE-AWARE ---
    {
        "id": "meeting_in_5min",
        "category": "deadline",
        "context": "The user has a meeting starting in 5 minutes. You checked the calendar and the meeting room is B204.",
        "query": "Which room is my meeting in?",
        "fresh_time": "30 seconds ago",
        "stale_time": "2 hours ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "refresh",
        "fact_type": "calendar",
    },
    {
        "id": "auction_closing",
        "category": "deadline",
        "context": "The eBay auction for the vintage camera closes in 10 minutes. Current bid is $234.",
        "query": "What's the current bid on that camera?",
        "fresh_time": "20 seconds ago",
        "stale_time": "45 minutes ago",
        "fresh_correct": "answer_directly",
        "stale_correct": "refresh",
        "fact_type": "auction",
    },

    # --- MEMORY RETRIEVAL ---
    {
        "id": "user_preference_old",
        "category": "memory",
        "context": "Three months ago, the user mentioned they're vegetarian. Last week, they said they started eating fish again.",
        "query": "Can you recommend a restaurant for me?",
        "fresh_time": "1 day ago",
        "stale_time": "6 months ago",
        "fresh_correct": "retrieve_memory",
        "stale_correct": "retrieve_memory",
        "fact_type": "user_preference",
    },
]


# ============================================================
# PROMPT TEMPLATES — 4 conditions
# ============================================================

def build_prompt_no_time(case: dict, is_stale: bool) -> str:
    """Condition 1: No temporal information at all."""
    return f"""You are a helpful AI assistant. Based on the conversation context, decide what action to take.

Context from previous conversation:
{case['context']}

User asks: "{case['query']}"

What should you do? Choose exactly one:
A) Answer directly from the context above
B) Refresh/look up the information again before answering
C) Retrieve from memory for more context
D) Ask the user a clarifying question
E) Abstain — tell the user you're not confident enough to answer

Respond with just the letter and a one-sentence explanation."""


def build_prompt_implicit(case: dict, is_stale: bool) -> str:
    """Condition 2: Timestamps present but not called out."""
    elapsed = case['stale_time'] if is_stale else case['fresh_time']
    return f"""You are a helpful AI assistant. Based on the conversation context, decide what action to take.

Context from previous conversation (observed {elapsed}):
{case['context']}

User asks now: "{case['query']}"

What should you do? Choose exactly one:
A) Answer directly from the context above
B) Refresh/look up the information again before answering
C) Retrieve from memory for more context
D) Ask the user a clarifying question
E) Abstain — tell the user you're not confident enough to answer

Respond with just the letter and a one-sentence explanation."""


def build_prompt_explicit(case: dict, is_stale: bool) -> str:
    """Condition 3: Explicit elapsed time annotation."""
    elapsed = case['stale_time'] if is_stale else case['fresh_time']
    return f"""You are a helpful AI assistant. Based on the conversation context, decide what action to take.

Context from previous conversation:
{case['context']}

Time since this information was observed: {elapsed}

User asks now: "{case['query']}"

What should you do? Choose exactly one:
A) Answer directly from the context above
B) Refresh/look up the information again before answering
C) Retrieve from memory for more context
D) Ask the user a clarifying question
E) Abstain — tell the user you're not confident enough to answer

Respond with just the letter and a one-sentence explanation."""


def build_prompt_prompted(case: dict, is_stale: bool) -> str:
    """Condition 4: Explicit time + system prompt about staleness."""
    elapsed = case['stale_time'] if is_stale else case['fresh_time']
    return f"""You are a helpful AI assistant that is careful about information freshness. Before answering any question, consider whether the available information might be outdated. If information could be stale, you should refresh it rather than risk giving the user incorrect information.

Context from previous conversation:
{case['context']}

Time since this information was observed: {elapsed}

User asks now: "{case['query']}"

What should you do? Choose exactly one:
A) Answer directly from the context above
B) Refresh/look up the information again before answering
C) Retrieve from memory for more context
D) Ask the user a clarifying question
E) Abstain — tell the user you're not confident enough to answer

Respond with just the letter and a one-sentence explanation."""


CONDITION_BUILDERS = {
    "no_time": build_prompt_no_time,
    "implicit": build_prompt_implicit,
    "explicit": build_prompt_explicit,
    "prompted": build_prompt_prompted,
}


def rephrase_prompt(prompt: str, style: str) -> str:
    """Apply a prompt style transformation."""
    if style == "default":
        return prompt
    elif style == "paraphrased":
        # Swap phrasing while keeping structure
        prompt = prompt.replace("What should you do?", "What is the best course of action?")
        prompt = prompt.replace("Choose exactly one:", "Select one option:")
        prompt = prompt.replace("Respond with just the letter and a one-sentence explanation.",
                                "Reply with the letter of your choice and briefly explain why.")
        prompt = prompt.replace("You are a helpful AI assistant",
                                "You are an intelligent assistant")
        return prompt
    elif style == "noisy_time":
        # Add noise words around time expressions
        prompt = prompt.replace("Time since this information was observed:",
                                "Note: it has been approximately")
        prompt = prompt.replace("(observed ", "(information was gathered roughly ")
        return prompt
    elif style == "minimal":
        # Strip to bare essentials
        lines = prompt.split("\n")
        # Remove empty lines and reduce verbosity
        filtered = [l for l in lines if l.strip()]
        return "\n".join(filtered)
    return prompt


PROMPT_STYLES = ["default", "paraphrased", "noisy_time", "minimal"]


# ============================================================
# MODEL CALLERS
# ============================================================

def call_openai(prompt: str, model: str = "gpt-4o") -> str:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI()
        # GPT-5+ uses max_completion_tokens instead of max_tokens
        token_param = "max_completion_tokens" if model.startswith("gpt-5") else "max_tokens"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            **{token_param: 150},
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def call_anthropic(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call Anthropic API."""
    try:
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"ERROR: {e}"


# --- Local Mistral on GPU ---

_mistral_pipeline = None

def _get_mistral_pipeline():
    """Lazy-load Mistral-7B-Instruct onto the local GPU (RTX 4080 16GB)."""
    global _mistral_pipeline
    if _mistral_pipeline is None:
        import torch
        from transformers import pipeline
        print("Loading Mistral-7B-Instruct locally (first call only)...")
        _mistral_pipeline = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            dtype=torch.float16,
            device_map="auto",
            max_new_tokens=150,
        )
        print("Mistral loaded.")
    return _mistral_pipeline


def call_mistral_local(prompt: str) -> str:
    """Run Mistral-7B-Instruct locally on RTX GPU."""
    try:
        pipe = _get_mistral_pipeline()
        messages = [{"role": "user", "content": prompt}]
        result = pipe(messages, max_new_tokens=150, do_sample=False)
        return result[0]["generated_text"][-1]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"


MODEL_CALLERS = {
    "gpt-4o": lambda p: call_openai(p, "gpt-4o"),
    "gpt-4o-mini": lambda p: call_openai(p, "gpt-4o-mini"),
    "gpt-5.4": lambda p: call_openai(p, "gpt-5.4"),
    "gpt-5.2": lambda p: call_openai(p, "gpt-5.2"),
    "claude-sonnet": lambda p: call_anthropic(p, "claude-sonnet-4-20250514"),
    "claude-haiku": lambda p: call_anthropic(p, "claude-haiku-4-5-20251001"),
    "claude-opus": lambda p: call_anthropic(p, "claude-opus-4-20250514"),
    "claude-opus-4.6": lambda p: call_anthropic(p, "claude-opus-4-6"),
    "mistral-local": call_mistral_local,
}


# ============================================================
# RESPONSE PARSER
# ============================================================

ACTION_MAP = {
    "A": "answer_directly",
    "B": "refresh",
    "C": "retrieve_memory",
    "D": "ask_clarify",
    "E": "abstain",
}

def parse_action(response: str) -> str:
    """Extract action from model response."""
    response = response.strip()
    # Don't parse error messages as actions
    if response.startswith("ERROR"):
        return "error"
    # Only accept leading letter if it looks like a letter choice: "A)", "A.", "A "
    if response and response[0].upper() in ACTION_MAP:
        if len(response) >= 2 and response[1] in "). ":
            return ACTION_MAP[response[0].upper()]
        if len(response) == 1:
            return ACTION_MAP[response[0].upper()]
    # Try to find letter in response
    for letter, action in ACTION_MAP.items():
        if f"{letter})" in response or f"{letter}." in response[:5]:
            return action
    # Keyword fallback
    response_lower = response.lower()
    if "refresh" in response_lower or "look up" in response_lower or "check again" in response_lower:
        return "refresh"
    if "answer directly" in response_lower or "answer from" in response_lower:
        return "answer_directly"
    if "retrieve" in response_lower or "memory" in response_lower:
        return "retrieve_memory"
    if "clarif" in response_lower or "ask" in response_lower:
        return "ask_clarify"
    if "abstain" in response_lower or "not confident" in response_lower:
        return "abstain"
    return "unknown"


# ============================================================
# EVALUATION ENGINE
# ============================================================

@dataclass
class EvalResult:
    model: str
    condition: str
    case_id: str
    is_stale: bool
    correct_action: str
    predicted_action: str
    raw_response: str
    is_correct: bool = False

    def __post_init__(self):
        self.is_correct = self.predicted_action == self.correct_action


def run_eval(models: list[str], conditions: list[str], cases: list[dict],
             checkpoint_file: Optional[str] = None, smoke_test: int = 0) -> list[EvalResult]:
    """Run full evaluation across all models x conditions x cases x fresh/stale.

    Args:
        checkpoint_file: If set, save results after each call and resume from existing file.
        smoke_test: If > 0, run only this many calls per model then stop early.
    """
    # Load checkpoint if it exists
    results = []
    completed_keys = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            raw = json.load(f)
        for r in raw:
            result = EvalResult(**r)
            results.append(result)
            completed_keys.add((r['model'], r['condition'], r['case_id'], r['is_stale']))
        print(f"Resumed from checkpoint: {len(results)} results loaded")

    total = len(models) * len(conditions) * len(cases) * 2
    count = 0
    smoke_count = {}  # per-model call counter for smoke test

    for model_name in models:
        caller = MODEL_CALLERS[model_name]
        smoke_count[model_name] = 0

        for condition_name in conditions:
            builder = CONDITION_BUILDERS[condition_name]

            for case in cases:
                for is_stale in [False, True]:
                    count += 1
                    correct = case['stale_correct'] if is_stale else case['fresh_correct']

                    # Skip if already in checkpoint
                    key = (model_name, condition_name, case['id'], is_stale)
                    if key in completed_keys:
                        continue

                    # Smoke test: stop after N calls per model
                    if smoke_test > 0 and smoke_count[model_name] >= smoke_test:
                        continue

                    prompt = builder(case, is_stale)

                    print(f"[{count}/{total}] {model_name} | {condition_name} | {case['id']} | {'stale' if is_stale else 'fresh'}")

                    response = caller(prompt)
                    action = parse_action(response)

                    # Bail on errors immediately
                    if action == "error":
                        print(f"  ERROR: {response[:120]}")
                        print("  Stopping — fix the error and rerun (checkpoint saved).")
                        if checkpoint_file:
                            _save_checkpoint(results, checkpoint_file)
                        return results

                    result = EvalResult(
                        model=model_name,
                        condition=condition_name,
                        case_id=case['id'],
                        is_stale=is_stale,
                        correct_action=correct,
                        predicted_action=action,
                        raw_response=response,
                    )
                    results.append(result)
                    smoke_count[model_name] += 1

                    # Checkpoint every 10 calls
                    if checkpoint_file and len(results) % 10 == 0:
                        _save_checkpoint(results, checkpoint_file)

                    # Rate limiting (skip for local models)
                    if model_name != "mistral-local":
                        time.sleep(0.5)

    # Final checkpoint save
    if checkpoint_file:
        _save_checkpoint(results, checkpoint_file)

    return results


def _save_checkpoint(results: list[EvalResult], path: str):
    """Save results to checkpoint file."""
    raw = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"  [checkpoint] {len(results)} results saved to {path}")


# ============================================================
# METRICS
# ============================================================

def compute_metrics(results: list[EvalResult]) -> dict:
    """Compute metrics per model per condition."""
    from collections import defaultdict

    metrics = defaultdict(lambda: defaultdict(dict))

    for model_name in set(r.model for r in results):
        for condition_name in set(r.condition for r in results):
            subset = [r for r in results if r.model == model_name and r.condition == condition_name]

            if not subset:
                continue

            # Accuracy
            correct = sum(1 for r in subset if r.is_correct)
            total = len(subset)
            accuracy = correct / total if total > 0 else 0

            # False trust: said "answer_directly" when correct was "refresh"
            stale_cases = [r for r in subset if r.is_stale and r.correct_action == "refresh"]
            false_trust = sum(1 for r in stale_cases if r.predicted_action == "answer_directly")
            false_trust_rate = false_trust / len(stale_cases) if stale_cases else 0

            # Switch sensitivity: for counterfactual pairs, did the action change?
            case_ids = set(r.case_id for r in subset)
            switches = 0
            should_switch = 0
            for cid in case_ids:
                case_results = [r for r in subset if r.case_id == cid]
                fresh = [r for r in case_results if not r.is_stale]
                stale = [r for r in case_results if r.is_stale]
                if fresh and stale:
                    if fresh[0].correct_action != stale[0].correct_action:
                        should_switch += 1
                        if fresh[0].predicted_action != stale[0].predicted_action:
                            switches += 1

            switch_sensitivity = switches / should_switch if should_switch > 0 else 0

            # Unnecessary refresh: said "refresh" for stable facts
            stable_cases = [r for r in subset if r.correct_action == "answer_directly"]
            unnecessary_refresh = sum(1 for r in stable_cases if r.predicted_action == "refresh")
            unnecessary_refresh_rate = unnecessary_refresh / len(stable_cases) if stable_cases else 0

            metrics[model_name][condition_name] = {
                "accuracy": round(accuracy, 3),
                "false_trust_rate": round(false_trust_rate, 3),
                "switch_sensitivity": round(switch_sensitivity, 3),
                "unnecessary_refresh_rate": round(unnecessary_refresh_rate, 3),
                "n_samples": total,
            }

    return dict(metrics)


def print_results_table(metrics: dict):
    """Print a clean results table."""
    print("\n" + "=" * 90)
    print("FRONTIER MODEL TEMPORAL REASONING EVALUATION")
    print("=" * 90)

    conditions = ["no_time", "implicit", "explicit", "prompted"]

    for model_name, model_metrics in metrics.items():
        print(f"\n{'-' * 90}")
        print(f"  MODEL: {model_name}")
        print(f"{'-' * 90}")
        print(f"  {'Condition':<12} {'Accuracy':>10} {'FalseTrust':>12} {'SwitchSens':>12} {'UnnecRefresh':>14}")
        print(f"  {'-' * 62}")

        for cond in conditions:
            if cond in model_metrics:
                m = model_metrics[cond]
                print(f"  {cond:<12} {m['accuracy']:>9.1%} {m['false_trust_rate']:>11.1%} {m['switch_sensitivity']:>11.1%} {m['unnecessary_refresh_rate']:>13.1%}")

    # Summary: the key finding
    print(f"\n{'=' * 90}")
    print("KEY FINDING: Compare 'no_time' vs 'explicit' switch sensitivity.")
    print("If no_time ~ 0% and explicit >> 0%, frontier models CANNOT detect")
    print("staleness without explicit temporal annotations.")
    print("=" * 90)


def compute_robustness_metrics(style_metrics: dict) -> dict:
    """Compute robustness metrics across prompt styles.

    Args:
        style_metrics: {style_name: {model: {condition: metrics_dict}}}

    Returns:
        Per model+condition: mean, std, min, max accuracy across styles.
    """
    from collections import defaultdict
    import numpy as np

    robustness = defaultdict(lambda: defaultdict(list))

    for style, metrics in style_metrics.items():
        for model, conditions in metrics.items():
            for condition, m in conditions.items():
                robustness[model][condition].append(m.get("accuracy", 0))

    result = {}
    for model, conditions in robustness.items():
        result[model] = {}
        for condition, accs in conditions.items():
            arr = np.array(accs)
            result[model][condition] = {
                "mean_accuracy": float(arr.mean()),
                "std_accuracy": float(arr.std()),
                "min_accuracy": float(arr.min()),
                "max_accuracy": float(arr.max()),
                "n_styles": len(accs),
            }

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Frontier model temporal reasoning eval")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"],
                        choices=list(MODEL_CALLERS.keys()),
                        help="Models to evaluate")
    parser.add_argument("--conditions", nargs="+", default=["no_time", "explicit"],
                        choices=list(CONDITION_BUILDERS.keys()),
                        help="Conditions to test")
    parser.add_argument("--output", default="outputs/results/frontier/frontier_eval_results.json",
                        help="Output file for raw results")
    parser.add_argument("--smoke-test", type=int, default=0,
                        help="Run only N calls per model to verify setup, then stop")
    parser.add_argument("--cases-file", type=str, default=None,
                        help="Load cases from JSON file instead of built-in TEST_CASES")
    args = parser.parse_args()

    # Load cases: from file or built-in
    if args.cases_file:
        with open(args.cases_file) as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} cases from {args.cases_file}")
    else:
        cases = TEST_CASES

    print(f"Running eval: models={args.models}, conditions={args.conditions}")
    print(f"Test cases: {len(cases)}")
    print(f"Total API calls: ~{len(args.models) * len(args.conditions) * len(cases) * 2}")
    if args.smoke_test:
        print(f"SMOKE TEST: {args.smoke_test} calls per model only")
    print()

    results = run_eval(args.models, args.conditions, cases,
                       checkpoint_file=args.output, smoke_test=args.smoke_test)

    # Save raw results
    raw = [asdict(r) for r in results]
    with open(args.output, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nRaw results saved to {args.output}")

    # Completeness check: refuse to write metrics from partial runs
    expected = 0
    for model_name in args.models:
        for condition_name in args.conditions:
            for case in cases:
                for is_stale in [False, True]:
                    expected += 1
    if args.smoke_test:
        expected = min(expected, args.smoke_test * len(args.models) * 2)

    actual = len(results)
    if actual < expected and args.smoke_test == 0:
        print(f"\nWARNING: Incomplete run ({actual}/{expected} calls).")
        print("Metrics NOT saved -- rerun to complete all calls.")
        print("Raw checkpoint saved; will resume on next run.")
        sys.exit(1)

    # Compute and print metrics
    metrics = compute_metrics(results)
    print_results_table(metrics)

    # Save metrics
    metrics_file = args.output.replace(".json", "_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")
    print(f"Completed: {actual}/{expected} calls")
