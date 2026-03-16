#!/usr/bin/env python3
"""
Run agent transfer evaluation.

Tests whether models trained on action classification transfer to
a utility-maximizing agent setting. Reports mean utility, normalized
utility (agent/oracle), and regret decomposition.

Usage:
    python scripts/run_agent_eval.py --episodes 500
    python scripts/run_agent_eval.py --episodes 500 --models gpt2_text_time tinyllama_lora
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_llm.agent.environment import TemporalAgentEnv, UtilityConfig, EpisodeResult
from temporal_llm.agent.scenarios import generate_agent_episodes
from temporal_llm.training.config import ModelConfig
from temporal_llm.models.temporal_model import TemporalActionModel
from temporal_llm.eval.evaluator import TemporalEvaluator


# Models to evaluate: (name, model_dir, variant)
MODELS = [
    ("gpt2_baseline", "outputs/gpt2_baseline_lr0.0005/best", "baseline"),
    ("gpt2_text_time", "outputs/gpt2_text_time/best", "text_time"),
    ("tinyllama_fullft_baseline", "outputs/tinyllama_fullft_baseline/best", "baseline"),
    ("tinyllama_fullft_text_time", "outputs/tinyllama_fullft_text_time/best", "text_time"),
    ("tinyllama_lora_midlr", "outputs/tinyllama_lora_midlr_text_time_savemods/best", "text_time"),
]


def evaluate_model(model_name, model_dir, variant, episodes, env):
    """Evaluate a single model on agent episodes."""
    meta_path = Path(model_dir) / "temporal_meta.json"
    if not meta_path.exists():
        print(f"  SKIP: {model_name} — {model_dir} not found")
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    base_model_name = meta.get("base_model_name", model_dir)
    config = ModelConfig(name=base_model_name)
    model = TemporalActionModel.load(model_dir, config, variant=variant)
    evaluator = TemporalEvaluator(model, device="cuda", batch_size=16)

    # Get predictions for all episodes
    input_field = f"input_{variant}"
    texts = [ep[input_field] for ep in episodes]
    preds = evaluator._predict_all(texts)

    # Compute utilities
    results = []
    for ep, pred in zip(episodes, preds):
        agent_util = env.compute_utility(
            action=pred,
            oracle_action=ep["correct_action"],
            elapsed_seconds=ep["elapsed_seconds"],
            staleness_threshold=ep["staleness_threshold"],
            deadline_seconds=ep.get("deadline_seconds"),
        )
        oracle_util = env.compute_oracle_utility(
            oracle_action=ep["correct_action"],
            elapsed_seconds=ep["elapsed_seconds"],
            staleness_threshold=ep["staleness_threshold"],
            deadline_seconds=ep.get("deadline_seconds"),
        )

        results.append(EpisodeResult(
            scenario_id=ep["id"],
            oracle_action=ep["correct_action"],
            agent_action=pred,
            oracle_utility=oracle_util,
            agent_utility=agent_util,
            elapsed_seconds=ep["elapsed_seconds"],
            volatility=ep["volatility"],
            fact_type=ep["fact_type"],
        ))

    # Free GPU
    del model, evaluator
    torch.cuda.empty_cache()

    return results


def compute_agent_metrics(results: list[EpisodeResult]) -> dict:
    """Compute aggregate agent metrics."""
    total_agent = sum(r.agent_utility for r in results)
    total_oracle = sum(r.oracle_utility for r in results)
    n = len(results)

    mean_agent = total_agent / n if n > 0 else 0
    mean_oracle = total_oracle / n if n > 0 else 0
    normalized = total_agent / total_oracle if total_oracle != 0 else 0
    accuracy = sum(1 for r in results if r.is_correct) / n if n > 0 else 0

    # Regret decomposition by error type
    regret = defaultdict(float)
    regret_counts = defaultdict(int)
    for r in results:
        if not r.is_correct:
            key = f"{r.agent_action}_instead_of_{r.oracle_action}"
            regret[key] += r.oracle_utility - r.agent_utility
            regret_counts[key] += 1

    # Per-volatility metrics
    by_vol = defaultdict(list)
    for r in results:
        by_vol[r.volatility].append(r)

    per_vol = {}
    for vol, vol_results in by_vol.items():
        vol_agent = sum(r.agent_utility for r in vol_results)
        vol_oracle = sum(r.oracle_utility for r in vol_results)
        per_vol[vol] = {
            "mean_utility": vol_agent / len(vol_results),
            "normalized_utility": vol_agent / vol_oracle if vol_oracle != 0 else 0,
            "accuracy": sum(1 for r in vol_results if r.is_correct) / len(vol_results),
            "n": len(vol_results),
        }

    return {
        "mean_utility": mean_agent,
        "mean_oracle_utility": mean_oracle,
        "normalized_utility": normalized,
        "accuracy": accuracy,
        "total_regret": total_oracle - total_agent,
        "regret_decomposition": {
            k: {"total_regret": v, "count": regret_counts[k]}
            for k, v in sorted(regret.items(), key=lambda x: -x[1])
        },
        "per_volatility": per_vol,
        "n_episodes": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Agent transfer evaluation")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/results/agent_eval_results.json")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter to specific model names")
    args = parser.parse_args()

    print(f"Generating {args.episodes} agent episodes...")
    episodes = generate_agent_episodes(n_episodes=args.episodes, seed=args.seed)
    print(f"  Generated {len(episodes)} episodes")

    env = TemporalAgentEnv()
    all_results = {}

    models = MODELS
    if args.models:
        models = [(n, d, v) for n, d, v in MODELS if n in args.models]

    for model_name, model_dir, variant in models:
        print(f"\nEvaluating {model_name}...")
        results = evaluate_model(model_name, model_dir, variant, episodes, env)
        if results is None:
            continue

        metrics = compute_agent_metrics(results)
        all_results[model_name] = metrics

        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Mean utility: {metrics['mean_utility']:.3f}")
        print(f"  Normalized utility: {metrics['normalized_utility']:.3f}")
        print(f"  Total regret: {metrics['total_regret']:.1f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {args.output}")

    # Print summary table
    print(f"\n{'Model':<30} {'Acc':>8} {'Utility':>10} {'Norm Util':>10} {'Regret':>10}")
    print("-" * 70)
    for model_name, metrics in all_results.items():
        print(f"{model_name:<30} {metrics['accuracy']:>8.3f} "
              f"{metrics['mean_utility']:>10.3f} "
              f"{metrics['normalized_utility']:>10.3f} "
              f"{metrics['total_regret']:>10.1f}")


if __name__ == "__main__":
    main()
