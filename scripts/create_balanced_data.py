#!/usr/bin/env python3
"""Create class-balanced training data via oversampling, undersampling, or hybrid."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def oversample(examples, target_per_class, rng):
    """Oversample minority classes to match target."""
    by_class = defaultdict(list)
    for ex in examples:
        by_class[ex["correct_action"]].append(ex)

    balanced = []
    for cls, exs in by_class.items():
        if len(exs) >= target_per_class:
            balanced.extend(exs)
        else:
            full_copies = target_per_class // len(exs)
            remainder = target_per_class % len(exs)
            for _ in range(full_copies):
                balanced.extend(exs)
            balanced.extend(rng.sample(exs, remainder))

    rng.shuffle(balanced)
    return balanced


def undersample(examples, target_per_class, rng):
    """Undersample majority classes to match target."""
    by_class = defaultdict(list)
    for ex in examples:
        by_class[ex["correct_action"]].append(ex)

    balanced = []
    for cls, exs in by_class.items():
        if len(exs) <= target_per_class:
            balanced.extend(exs)
        else:
            balanced.extend(rng.sample(exs, target_per_class))

    rng.shuffle(balanced)
    return balanced


def hybrid(examples, rng):
    """Oversample minority to median, undersample majority to median."""
    by_class = defaultdict(list)
    for ex in examples:
        by_class[ex["correct_action"]].append(ex)

    sizes = [len(v) for v in by_class.values()]
    median_size = sorted(sizes)[len(sizes) // 2]

    balanced = []
    for cls, exs in by_class.items():
        if len(exs) <= median_size:
            # Oversample to median
            full_copies = median_size // len(exs)
            remainder = median_size % len(exs)
            for _ in range(full_copies):
                balanced.extend(exs)
            balanced.extend(rng.sample(exs, remainder))
        else:
            # Undersample to median
            balanced.extend(rng.sample(exs, median_size))

    rng.shuffle(balanced)
    return balanced


def balance_split(input_path, output_path, strategy, rng):
    """Balance a single split file."""
    examples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    by_class = defaultdict(list)
    for ex in examples:
        by_class[ex["correct_action"]].append(ex)

    print(f"  Original: {len(examples)} examples")
    for cls, exs in sorted(by_class.items()):
        print(f"    {cls:>20}: {len(exs)}")

    if strategy == "oversample":
        target = max(len(v) for v in by_class.values())
        balanced = oversample(examples, target, rng)
    elif strategy == "undersample":
        target = min(len(v) for v in by_class.values())
        balanced = undersample(examples, target, rng)
    elif strategy == "hybrid":
        balanced = hybrid(examples, rng)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Print result distribution
    result_counts = defaultdict(int)
    for ex in balanced:
        result_counts[ex["correct_action"]] += 1
    print(f"  Balanced ({strategy}): {len(balanced)} examples")
    for cls, count in sorted(result_counts.items()):
        print(f"    {cls:>20}: {count}")

    with open(output_path, "w") as f:
        for ex in balanced:
            f.write(json.dumps(ex) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create class-balanced training data")
    parser.add_argument("--input-dir", type=str, default="data/generated",
                        help="Input data directory")
    parser.add_argument("--output-dir", type=str, default="data/balanced",
                        help="Output data directory")
    parser.add_argument("--strategy", choices=["oversample", "undersample", "hybrid"],
                        default="oversample", help="Balancing strategy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating balanced training data (strategy={args.strategy})...")
    balance_split(input_dir / "train.jsonl", output_dir / "train.jsonl", args.strategy, rng)

    # Copy val and test unchanged
    print("\nCopying val and test (unchanged)...")
    for split in ["val.jsonl", "test.jsonl"]:
        with open(input_dir / split) as f_in, open(output_dir / split, "w") as f_out:
            f_out.write(f_in.read())

    print("Done.")


if __name__ == "__main__":
    main()
