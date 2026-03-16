#!/usr/bin/env python3
"""
Interactive demo for the trained temporal action model.

Usage:
    python scripts/demo.py
    python scripts/demo.py --model-dir outputs/gpt2_time_tokens_lr0.0005/best
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from temporal_llm.models.temporal_model import TemporalActionModel
from temporal_llm.training.config import ModelConfig
from temporal_llm.data.taxonomy import (
    ACTION_LABELS,
    FACT_TYPES,
    FACT_TYPE_MAP,
    seconds_to_token,
    seconds_to_human,
    Volatility,
)


EXAMPLE_SCENARIOS = [
    {
        "name": "Fresh flight info (5 min ago)",
        "conversation": "User: Can you check the status of flight AA123?\nAssistant: Flight AA123 is delayed by 45 minutes. New departure time is 7:45 PM.",
        "query": "Is my flight still delayed?",
        "elapsed": 300,
        "fact_type": "flight_status",
    },
    {
        "name": "Stale stock price (3 hours ago)",
        "conversation": "User: What's NVDA trading at?\nAssistant: NVDA is currently at $495.22, up 5.7% today.",
        "query": "What's NVDA at now?",
        "elapsed": 10800,
        "fact_type": "stock_price",
    },
    {
        "name": "Old meeting info (2 days ago)",
        "conversation": "User: When is the design review?\nAssistant: The design review is scheduled for Friday at 11 AM in the large conference room.",
        "query": "Is the design review still at the same time?",
        "elapsed": 172800,
        "fact_type": "meeting_schedule",
    },
    {
        "name": "Historical fact (1 year ago)",
        "conversation": "User: When did the moon landing happen?\nAssistant: The moon landing occurred in 1969. Neil Armstrong was the first to step on the surface.",
        "query": "When was the moon landing again?",
        "elapsed": 31536000,
        "fact_type": "historical_fact",
    },
    {
        "name": "Very stale traffic (6 hours ago)",
        "conversation": "User: How's the traffic on I-405?\nAssistant: I-405 is backed up due to an accident. Estimated travel time is 45 minutes.",
        "query": "Is the traffic still bad?",
        "elapsed": 21600,
        "fact_type": "traffic",
    },
]


def build_prompt(conversation: str, query: str, elapsed: float, fact_type: str) -> str:
    ft = FACT_TYPE_MAP.get(fact_type)
    vol_str = ft.volatility.value if ft else "unknown"
    dt_token = seconds_to_token(elapsed)
    dt_human = seconds_to_human(elapsed)

    return (
        f"Given the following conversation, decide the best action.\n\n"
        f"Conversation:\n{conversation}\n\n"
        f"Time since last information: {dt_token} ({dt_human})\n"
        f"Fact type: {fact_type} ({vol_str} volatility)\n\n"
        f"User: {query}\n\n"
        f"Actions: {', '.join(ACTION_LABELS)}\n\n"
        f"Action:"
    )


def predict(model: TemporalActionModel, prompt: str) -> tuple[str, dict]:
    device = next(model.model.parameters()).device
    enc = model.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Get action prediction
    actions = model.predict_action(input_ids, attention_mask)
    # Get action probabilities
    probs = model.predict_action_probs(input_ids, attention_mask)
    prob_dict = {a: float(probs[0, i]) for i, a in enumerate(ACTION_LABELS)}

    return actions[0], prob_dict


def run_example(model, scenario):
    prompt = build_prompt(
        scenario["conversation"],
        scenario["query"],
        scenario["elapsed"],
        scenario["fact_type"],
    )
    action, probs = predict(model, prompt)

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario['name']}")
    print(f"  Query: {scenario['query']}")
    print(f"  Elapsed: {seconds_to_human(scenario['elapsed'])}")
    print(f"  Fact type: {scenario['fact_type']}")
    print(f"\n  Predicted action: ** {action} **")
    print(f"\n  Action probabilities:")
    for a in sorted(probs, key=probs.get, reverse=True):
        bar = '#' * int(probs[a] * 40)
        print(f"    {a:<20s} {probs[a]:6.1%} {bar}")


def interactive_mode(model):
    print("\n--- Custom scenario ---")
    print("Enter conversation (e.g. 'User: ... \\nAssistant: ...'):")
    conversation = input("> ").replace("\\n", "\n")

    print("Enter follow-up query:")
    query = input("> ")

    print("Elapsed time in seconds (e.g. 300 for 5 min, 3600 for 1 hour):")
    elapsed = float(input("> "))

    print(f"Fact types: {', '.join(FACT_TYPE_MAP.keys())}")
    print("Enter fact type:")
    fact_type = input("> ").strip()

    scenario = {
        "name": "Custom",
        "conversation": conversation,
        "query": query,
        "elapsed": elapsed,
        "fact_type": fact_type,
    }
    run_example(model, scenario)


def main():
    parser = argparse.ArgumentParser(description="Interactive temporal action demo")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs/gpt2_time_tokens_lr0.0005/best",
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="Base model name (for loading config)",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    config = ModelConfig(name=args.model_name, use_lora=False)
    model = TemporalActionModel.load(args.model_dir, config, variant="time_tokens")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)
    print(f"Model loaded on {device}.\n")

    while True:
        print("\n" + "="*60)
        print("TEMPORAL ACTION DEMO")
        print("="*60)
        for i, s in enumerate(EXAMPLE_SCENARIOS):
            print(f"  [{i+1}] {s['name']}")
        print(f"  [c] Custom scenario")
        print(f"  [a] Run all examples")
        print(f"  [q] Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == "q":
            break
        elif choice == "a":
            for s in EXAMPLE_SCENARIOS:
                run_example(model, s)
        elif choice == "c":
            interactive_mode(model)
        elif choice.isdigit() and 1 <= int(choice) <= len(EXAMPLE_SCENARIOS):
            run_example(model, EXAMPLE_SCENARIOS[int(choice) - 1])
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
