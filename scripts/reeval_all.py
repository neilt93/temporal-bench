#!/usr/bin/env python3
"""
Re-evaluate ALL saved models with the fixed evaluator (Bug 1 fix).
Also re-parse frontier raw responses with the fixed parse_action (Bug 3 fix).

Outputs go to outputs/results/ with the same filenames, overwriting old results.
Saves old results to outputs/results/pre_bugfix/ for comparison.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_llm.training.config import ModelConfig
from temporal_llm.models.temporal_model import TemporalActionModel
from temporal_llm.eval.evaluator import TemporalEvaluator

RESULTS_DIR = Path("outputs/results")
BACKUP_DIR = RESULTS_DIR / "pre_bugfix"
TEST_DATA = Path("data/generated/test.jsonl")

# (result_filename, model_dir, variant)
MODELS = [
    ("gpt2_baseline_results.json", "outputs/gpt2_baseline_lr0.0005/best", "baseline"),
    ("gpt2_time_tokens_results.json", "outputs/gpt2_time_tokens_lr0.0005/best", "time_tokens"),
    ("gpt2_time_memory_results.json", "outputs/gpt2_time_memory_lr0.0005/best", "time_memory"),
    ("gpt2_text_time_results.json", "outputs/gpt2_text_time/best", "text_time"),
    ("gpt2_balanced_text_time_results.json", "outputs/gpt2_balanced_text_time/best", "text_time"),
    ("gpt2_v2_text_time_results.json", "outputs/gpt2_v2_text_time/best", "text_time"),
    ("gpt2_v2_baseline_results.json", "outputs/gpt2_v2_baseline/best", "baseline"),
    ("tinyllama_baseline_results.json", "outputs/TinyLlama-1.1B-Chat-v1.0_baseline_lr0.0002/best", "baseline"),
    ("tinyllama_time_tokens_results.json", "outputs/TinyLlama-1.1B-Chat-v1.0_time_tokens_lr0.0002/best", "time_tokens"),
    ("tinyllama_time_memory_results.json", "outputs/TinyLlama-1.1B-Chat-v1.0_time_memory_lr0.0002/best", "time_memory"),
    ("tinyllama_fullft_baseline_results.json", "outputs/tinyllama_fullft_baseline/best", "baseline"),
    ("tinyllama_fullft_time_tokens_results.json", "outputs/tinyllama_fullft_time_tokens/best", "time_tokens"),
    ("tinyllama_fullft_text_time_results.json", "outputs/tinyllama_fullft_text_time/best", "text_time"),
    ("tinyllama_lora_r64_time_tokens_results.json", "outputs/tinyllama_lora_r64_time_tokens/best", "time_tokens"),
    ("tinyllama_lora_r64_text_time_results.json", "outputs/tinyllama_lora_r64_text_time/best", "text_time"),
    ("tinyllama_lora_highlr_time_tokens_results.json", "outputs/tinyllama_lora_highlr_time_tokens/best", "time_tokens"),
    ("tinyllama_lora_highlr_text_time_results.json", "outputs/tinyllama_lora_highlr_text_time/best", "text_time"),
    ("tinyllama_lora_highlr_text_time_savemods_results.json", "outputs/tinyllama_lora_highlr_text_time_savemods/best", "text_time"),
    ("tinyllama_lora_midlr_text_time_savemods_results.json", "outputs/tinyllama_lora_midlr_text_time_savemods/best", "text_time"),
    ("tinyllama_lora_balanced_text_time_results.json", "outputs/tinyllama_lora_balanced_text_time/best", "text_time"),
    ("tinyllama_lora_v2_text_time_results.json", "outputs/tinyllama_lora_v2_text_time/best", "text_time"),
    ("tinyllama_lora_v2_baseline_results.json", "outputs/tinyllama_lora_v2_baseline/best", "baseline"),
]


def backup_old_results():
    """Save old results for comparison."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for path in RESULTS_DIR.glob("*_results.json"):
        if path.parent == BACKUP_DIR:
            continue
        dst = BACKUP_DIR / path.name
        if not dst.exists():
            shutil.copy2(path, dst)
            print(f"  Backed up {path.name}")


def evaluate_model(result_name, model_dir, variant):
    """Load model, run all evals + ablations, save results."""
    model_path = Path(model_dir)
    meta_path = model_path / "temporal_meta.json"
    if not meta_path.exists():
        print(f"  SKIP: {model_dir} not found")
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    base_model_name = meta.get("base_model_name", model_dir)

    config = ModelConfig(name=base_model_name)
    model = TemporalActionModel.load(model_dir, config, variant=variant)
    evaluator = TemporalEvaluator(model, device="cuda", batch_size=16)

    # Run all evals
    results = evaluator.run_all(str(TEST_DATA), variant=variant)

    # Run ablations
    results["ablations"] = evaluator.run_all_ablations(str(TEST_DATA), variant=variant)

    # Save
    output_path = RESULTS_DIR / result_name
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Free GPU
    del model, evaluator
    torch.cuda.empty_cache()

    return results


def compare_metrics(old_path, new_results):
    """Compare old vs new false_trust_rate and unnecessary_refresh_rate."""
    if not old_path.exists():
        return None

    with open(old_path) as f:
        old = json.load(f)

    old_sc = old.get("stale_calibration", {})
    new_sc = new_results.get("stale_calibration", {})

    return {
        "false_trust_rate": {
            "old": old_sc.get("false_trust_rate", 0),
            "new": new_sc.get("false_trust_rate", 0),
        },
        "unnecessary_refresh_rate": {
            "old": old_sc.get("unnecessary_refresh_rate", 0),
            "new": new_sc.get("unnecessary_refresh_rate", 0),
        },
    }


def reparse_frontier_results():
    """Re-parse frontier raw responses with fixed parse_action."""
    # Import the fixed parser
    sys.path.insert(0, str(Path(__file__).parent))
    from frontier_temporal_eval import parse_action, compute_metrics, EvalResult

    frontier_dir = RESULTS_DIR / "frontier"
    if not frontier_dir.exists():
        print("\nNo frontier results directory found, skipping")
        return

    frontier_files = list(frontier_dir.glob("frontier_eval_results*.json"))
    frontier_files = [f for f in frontier_files if "metrics" not in f.name]

    misparse_count = 0
    total_count = 0

    for fpath in frontier_files:
        print(f"\n  Re-parsing {fpath.name}...")

        # Backup
        backup = BACKUP_DIR / "frontier"
        backup.mkdir(parents=True, exist_ok=True)
        if not (backup / fpath.name).exists():
            shutil.copy2(fpath, backup / fpath.name)

        with open(fpath) as f:
            raw_results = json.load(f)

        changed = 0
        for r in raw_results:
            total_count += 1
            old_action = r["predicted_action"]
            new_action = parse_action(r["raw_response"])
            if old_action != new_action:
                changed += 1
                misparse_count += 1
                print(f"    CHANGED: {r['case_id']} ({r['model']}/{r['condition']}) "
                      f"{old_action} -> {new_action}")
                print(f"      Raw: {r['raw_response'][:80]}...")
                r["predicted_action"] = new_action
                r["is_correct"] = new_action == r["correct_action"]

        print(f"  {changed} predictions changed out of {len(raw_results)}")

        # Save updated results
        with open(fpath, "w") as f:
            json.dump(raw_results, f, indent=2)

        # Recompute metrics
        eval_results = [EvalResult(**r) for r in raw_results]
        metrics = compute_metrics(eval_results)
        metrics_path = fpath.with_name(fpath.stem + "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Updated metrics saved to {metrics_path.name}")

    print(f"\nFrontier audit: {misparse_count}/{total_count} predictions changed")


def main():
    start = time.time()

    print("=" * 60)
    print("Phase 0: Re-evaluating all models with fixed evaluator")
    print("=" * 60)

    # Step 1: Backup old results
    print("\nBacking up old results...")
    backup_old_results()

    # Step 2: Re-evaluate all fine-tuned models
    comparisons = {}
    for i, (result_name, model_dir, variant) in enumerate(MODELS):
        print(f"\n[{i+1}/{len(MODELS)}] {result_name} ({variant})")
        t0 = time.time()

        results = evaluate_model(result_name, model_dir, variant)
        if results is None:
            continue

        # Compare old vs new
        old_path = BACKUP_DIR / result_name
        diff = compare_metrics(old_path, results)
        if diff:
            comparisons[result_name] = diff
            ftr = diff["false_trust_rate"]
            urr = diff["unnecessary_refresh_rate"]
            print(f"  false_trust_rate:        {ftr['old']:.4f} -> {ftr['new']:.4f}")
            print(f"  unnecessary_refresh_rate: {urr['old']:.4f} -> {urr['new']:.4f}")

        print(f"  Done in {time.time() - t0:.1f}s")

    # Step 3: Re-parse frontier results
    print("\n" + "=" * 60)
    print("Re-parsing frontier results with fixed parse_action")
    print("=" * 60)
    reparse_frontier_results()

    # Step 4: Print comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY: Old vs New Metrics")
    print("=" * 60)
    print(f"\n{'Model':<50} {'Old FTR':>8} {'New FTR':>8} {'Old URR':>8} {'New URR':>8}")
    print("-" * 84)
    for name, diff in comparisons.items():
        ftr = diff["false_trust_rate"]
        urr = diff["unnecessary_refresh_rate"]
        print(f"{name:<50} {ftr['old']:>8.4f} {ftr['new']:>8.4f} "
              f"{urr['old']:>8.4f} {urr['new']:>8.4f}")

    total = time.time() - start
    print(f"\nTotal time: {total/60:.1f} minutes")


if __name__ == "__main__":
    main()
