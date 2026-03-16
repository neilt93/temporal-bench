#!/usr/bin/env python3
"""
Generate the temporal action classification dataset.

Usage:
    python scripts/generate_data.py --output data/generated --examples-per-type 20
    python scripts/generate_data.py --output data/generated --examples-per-type 50 --seed 123
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_llm.data.generator import DatasetGenerator
from temporal_llm.data.taxonomy import FACT_TYPES, Volatility


def main():
    parser = argparse.ArgumentParser(description="Generate temporal action dataset")
    parser.add_argument("--output", type=str, default="data/generated",
                        help="Output directory")
    parser.add_argument("--examples-per-type", type=int, default=20,
                        help="Counterfactual groups per fact type")
    parser.add_argument("--group-size", type=int, default=6,
                        help="Examples per counterfactual group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--no-memory", action="store_true",
                        help="Exclude memory entries")
    parser.add_argument("--no-deadline", action="store_true",
                        help="Exclude deadline examples")
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    assert test_ratio > 0, "Train + val ratios must be less than 1.0"

    print(f"Generating dataset with {args.examples_per_type} groups per fact type...")
    print(f"  Counterfactual group size: {args.group_size}")
    print(f"  Fact types: {len(FACT_TYPES)}")
    print(f"  Expected total examples: ~{args.examples_per_type * len(FACT_TYPES) * args.group_size}")
    print(f"  Memory: {'yes' if not args.no_memory else 'no'}")
    print(f"  Deadlines: {'yes' if not args.no_deadline else 'no'}")
    print(f"  Split ratios: {args.train_ratio}/{args.val_ratio}/{test_ratio:.2f}")
    print()

    generator = DatasetGenerator(
        seed=args.seed,
        counterfactual_group_size=args.group_size,
        include_memory=not args.no_memory,
        include_deadline=not args.no_deadline,
    )

    counts = generator.save_dataset(
        output_dir=args.output,
        examples_per_fact_type=args.examples_per_type,
        split_ratios=(args.train_ratio, args.val_ratio, test_ratio),
    )

    print("Dataset generated:")
    for split, count in counts.items():
        print(f"  {split}: {count} examples")
    print(f"\nSaved to {args.output}/")

    # Print action distribution
    import json
    from collections import Counter
    for split in ["train", "val", "test"]:
        path = Path(args.output) / f"{split}.jsonl"
        actions = Counter()
        volatilities = Counter()
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                actions[ex["correct_action"]] += 1
                volatilities[ex["volatility"]] += 1

        print(f"\n{split} action distribution:")
        for action, count in sorted(actions.items()):
            print(f"  {action}: {count} ({count/sum(actions.values())*100:.1f}%)")

        print(f"{split} volatility distribution:")
        for vol, count in sorted(volatilities.items()):
            print(f"  {vol}: {count} ({count/sum(volatilities.values())*100:.1f}%)")


if __name__ == "__main__":
    main()
