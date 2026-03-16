#!/usr/bin/env python3
"""
Programmatically generate frontier eval cases from the taxonomy.

Produces cases in the same dict format as TEST_CASES in frontier_temporal_eval.py,
ensuring even coverage across fact types, volatility levels, and oracle actions.

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
    oracle_action, seconds_to_human,
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


def generate_case(fact_type, template, slots, fresh_time, stale_time, case_idx, rng):
    """Generate a single frontier eval case dict."""
    # Build context from template history
    context_parts = []
    for turn in template.history:
        if turn.is_observation:
            context_parts.append(fill_template(turn.content, slots))
        else:
            context_parts.append(fill_template(turn.content, slots))

    context = " ".join(
        fill_template(turn.content, slots)
        for turn in template.history
        if turn.is_observation
    )
    if not context:
        context = " ".join(
            fill_template(turn.content, slots)
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
    vol = fact_type.volatility
    if vol == Volatility.STABLE:
        fresh_correct = "answer_directly"
        stale_correct = "answer_directly"
    else:
        fresh_correct = "answer_directly"
        stale_correct = "refresh"

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

    print(f"Generated {len(cases)} frontier eval cases → {args.output}")
    print(f"\nBy category:")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")
    print(f"\nBy fact type:")
    for ft, count in sorted(by_fact.items()):
        print(f"  {ft}: {count}")


if __name__ == "__main__":
    main()
