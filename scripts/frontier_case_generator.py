#!/usr/bin/env python3
"""
Programmatically generate frontier eval cases from the taxonomy.

Produces cases in the same dict format as TEST_CASES in frontier_temporal_eval.py,
covering the taxonomy's fact types and volatility levels while preserving
templates that imply clarification or memory-retrieval behavior.

Usage:
    python scripts/frontier_case_generator.py --num-per-type 15 --output data/frontier_eval_cases_v2.json
    python scripts/frontier_case_generator.py --num-per-type 15 --balance --output data/frontier_eval_cases_v2.json
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_llm.data.taxonomy import (
    FACT_TYPES, FACT_TYPE_MAP, Volatility,
    oracle_action,
)
from temporal_llm.data.templates import TEMPLATE_MAP, sample_slots, fill_template


# Time ranges for fresh/stale by volatility
FRESH_TIMES = {
    Volatility.HIGH: ("5 seconds ago", "2 minutes ago", "30 seconds ago"),
    Volatility.MEDIUM: ("10 minutes ago", "1 hour ago", "30 minutes ago"),
    Volatility.LOW: ("2 hours ago", "1 day ago", "6 hours ago"),
    Volatility.STABLE: ("1 hour ago", "5 minutes ago", "10 minutes ago"),
}

STALE_TIMES = {
    Volatility.HIGH: ("4 hours ago", "1 day ago", "6 hours ago", "8 hours ago"),
    Volatility.MEDIUM: ("3 days ago", "1 week ago", "2 weeks ago"),
    Volatility.LOW: ("6 months ago", "1 year ago", "8 months ago"),
    Volatility.STABLE: ("1 year ago", "6 months ago", "2 years ago"),
}

# Map volatility to category strings used in TEST_CASES
VOLATILITY_CATEGORY = {
    Volatility.HIGH: "high_volatility",
    Volatility.MEDIUM: "medium_volatility",
    Volatility.LOW: "low_volatility",
    Volatility.STABLE: "stable",
}


TIME_UNITS_TO_SECONDS = {
    "second": 1,
    "seconds": 1,
    "minute": 60,
    "minutes": 60,
    "hour": 3600,
    "hours": 3600,
    "day": 86400,
    "days": 86400,
    "week": 604800,
    "weeks": 604800,
    "month": 2592000,
    "months": 2592000,
    "year": 31536000,
    "years": 31536000,
}


def _elapsed_text_to_seconds(text: str) -> int:
    """Parse strings like '5 minutes ago' into seconds."""
    amount, unit, *_ = text.split()
    return int(amount) * TIME_UNITS_TO_SECONDS[unit]


def _infer_fixed_action(template) -> str | None:
    """Detect templates that should always map to a non-default action."""
    description = template.description.lower()
    if "contradictory" in description or "conflicting" in description:
        return "ask_clarify"
    if "memory" in description:
        return "retrieve_memory"
    return None


def generate_case(fact_type, template, slots, fresh_time, stale_time, case_idx, rng):
    """Generate a single frontier eval case dict."""
    # Preserve the full dialogue so contradictory and memory-sensitive
    # scenarios stay visible in the generated benchmark.
    context = " ".join(
        f"{'User' if turn.role == 'user' else 'Assistant'}: "
        f"{fill_template(turn.content, slots)}"
        for turn in template.history
    )

    # Add a prefix like "You checked/looked up..."
    prefixes = [
        "You previously looked up: ",
        "You checked earlier. ",
        "Earlier, you found out: ",
        "From a previous check: ",
    ]
    context = rng.choice(prefixes) + context

    # Query
    query = template.sample_query(rng=rng)
    query = fill_template(query, slots)

    # Determine correct actions
    fixed_action = _infer_fixed_action(template)
    if fixed_action is not None:
        fresh_correct = fixed_action
        stale_correct = fixed_action
    else:
        fresh_correct = oracle_action(
            elapsed_seconds=_elapsed_text_to_seconds(fresh_time),
            fact_type=fact_type,
            has_memory=False,
            has_refresh_tool=True,
        ).value
        stale_correct = oracle_action(
            elapsed_seconds=_elapsed_text_to_seconds(stale_time),
            fact_type=fact_type,
            has_memory=False,
            has_refresh_tool=True,
        ).value
    vol = fact_type.volatility

    case_id = f"{fact_type.name}_{case_idx}"

    return {
        "id": case_id,
        "category": VOLATILITY_CATEGORY[vol],
        "context": context,
        "query": query,
        "fresh_time": fresh_time,
        "stale_time": stale_time,
        "fresh_correct": fresh_correct,
        "stale_correct": stale_correct,
        "fact_type": fact_type.name,
    }


def generate_cases(num_per_type=15, balance=False, seed=42):
    """Generate frontier eval cases from all fact types."""
    rng = random.Random(seed)
    cases = []

    for fact_type in FACT_TYPES:
        templates = TEMPLATE_MAP.get(fact_type.name, [])
        if not templates:
            continue

        vol = fact_type.volatility
        fresh_options = FRESH_TIMES[vol]
        stale_options = STALE_TIMES[vol]

        for i in range(num_per_type):
            template = templates[i % len(templates)]
            slots = sample_slots(fact_type.name, rng=rng)
            fresh_time = rng.choice(fresh_options)
            stale_time = rng.choice(stale_options)

            case = generate_case(
                fact_type, template, slots,
                fresh_time, stale_time, i, rng,
            )
            cases.append(case)

    if balance:
        # Ensure even coverage across categories
        by_cat = defaultdict(list)
        for c in cases:
            by_cat[c["category"]].append(c)

        min_count = min(len(v) for v in by_cat.values()) if by_cat else 0
        balanced = []
        for cat, cat_cases in by_cat.items():
            rng.shuffle(cat_cases)
            balanced.extend(cat_cases[:min_count])
        rng.shuffle(balanced)
        cases = balanced

    return cases


def main():
    parser = argparse.ArgumentParser(
        description="Generate frontier eval cases from taxonomy"
    )
    parser.add_argument("--num-per-type", type=int, default=15,
                        help="Cases per fact type (default: 15)")
    parser.add_argument("--balance", action="store_true",
                        help="Balance cases across volatility categories")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="data/frontier_eval_cases_v2.json",
                        help="Output file path")
    args = parser.parse_args()

    cases = generate_cases(
        num_per_type=args.num_per_type,
        balance=args.balance,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(cases, f, indent=2)

    # Print summary
    by_cat = defaultdict(int)
    by_fact = defaultdict(int)
    for c in cases:
        by_cat[c["category"]] += 1
        by_fact[c["fact_type"]] += 1

    print(f"Generated {len(cases)} frontier eval cases -> {args.output}")
    print(f"\nBy category:")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")
    print(f"\nBy fact type:")
    for ft, count in sorted(by_fact.items()):
        print(f"  {ft}: {count}")


if __name__ == "__main__":
    main()
