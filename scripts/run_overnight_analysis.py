#!/usr/bin/env python3
"""
Overnight analysis batch: confusion matrices, balanced slices,
volatile vs stable breakdown, and counterfactual threshold curves.

Runs analysis on all trained models without retraining.
Saves all outputs to outputs/analysis/.
"""

import json
import sys
import time
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_llm.training.config import ModelConfig
from temporal_llm.models.temporal_model import TemporalActionModel
from temporal_llm.eval.evaluator import TemporalEvaluator
from temporal_llm.data.taxonomy import (
    ACTION_LABELS,
    seconds_to_human,
    seconds_to_token,
)
from temporal_llm.data.dataset import TemporalDataset

LOG_PATH = Path("outputs/analysis_log.txt")
ANALYSIS_DIR = Path("outputs/analysis")
TEST_DATA = Path("data/generated/test.jsonl")

# Models to analyze: (name, model_dir, variant)
MODELS = [
    ("gpt2_baseline", "outputs/gpt2_baseline_lr0.0005/best", "baseline"),
    ("gpt2_time_tokens", "outputs/gpt2_time_tokens_lr0.0005/best", "time_tokens"),
    ("gpt2_text_time", "outputs/gpt2_text_time/best", "text_time"),
    ("tinyllama_fullft_baseline", "outputs/tinyllama_fullft_baseline/best", "baseline"),
    ("tinyllama_fullft_time_tokens", "outputs/tinyllama_fullft_time_tokens/best", "time_tokens"),
    ("tinyllama_fullft_text_time", "outputs/tinyllama_fullft_text_time/best", "text_time"),
    ("tinyllama_lora_midlr", "outputs/tinyllama_lora_midlr_text_time_savemods/best", "text_time"),
    ("tinyllama_lora_highlr", "outputs/tinyllama_lora_highlr_text_time_savemods/best", "text_time"),
]


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_examples():
    examples = []
    with open(TEST_DATA) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def get_predictions(model_name, model_dir, variant):
    """Load model, get per-example predictions, save to disk."""
    pred_path = ANALYSIS_DIR / f"{model_name}_predictions.json"
    if pred_path.exists():
        log(f"  Predictions already cached for {model_name}")
        with open(pred_path) as f:
            return json.load(f)

    log(f"  Loading model {model_name} from {model_dir}...")
    meta_path = Path(model_dir) / "temporal_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        base_model_name = meta.get("base_model_name", model_dir)
    else:
        base_model_name = model_dir

    config = ModelConfig(name=base_model_name)
    model = TemporalActionModel.load(model_dir, config, variant=variant)
    evaluator = TemporalEvaluator(model, device="cuda", batch_size=16)

    examples = load_examples()
    texts = [evaluator._get_input_text(ex, variant) for ex in examples]
    preds = evaluator._predict_all(texts)

    pred_data = {
        "model_name": model_name,
        "variant": variant,
        "predictions": preds,
        "oracles": [ex["correct_action"] for ex in examples],
        "volatilities": [ex["volatility"] for ex in examples],
        "fact_types": [ex["fact_type"] for ex in examples],
        "elapsed_seconds": [ex["elapsed_seconds"] for ex in examples],
        "counterfactual_groups": [ex["counterfactual_group"] for ex in examples],
        "ids": [ex["id"] for ex in examples],
    }

    with open(pred_path, "w") as f:
        json.dump(pred_data, f, indent=2)
    log(f"  Saved {len(preds)} predictions for {model_name}")

    # Free GPU memory
    del model, evaluator
    torch.cuda.empty_cache()

    return pred_data


# ===================================================================
# Analysis 1: Per-class confusion matrices
# ===================================================================

def analysis_confusion_matrices(all_preds):
    """Generate confusion matrices for each model."""
    log("Running Analysis 1: Per-class confusion matrices")
    results = {}

    for model_name, data in all_preds.items():
        preds = data["predictions"]
        oracles = data["oracles"]

        cm = confusion_matrix(oracles, preds, labels=ACTION_LABELS)
        report = classification_report(
            oracles, preds, labels=ACTION_LABELS,
            zero_division=0, output_dict=True,
        )

        results[model_name] = {
            "confusion_matrix": cm.tolist(),
            "labels": ACTION_LABELS,
            "per_class": {
                label: {
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"],
                    "support": report[label]["support"],
                }
                for label in ACTION_LABELS
                if label in report
            },
            "accuracy": accuracy_score(oracles, preds),
        }

        # Log the matrix
        log(f"\n  {model_name} confusion matrix:")
        log(f"  {'':>20} " + " ".join(f"{a[:8]:>10}" for a in ACTION_LABELS))
        for i, row_label in enumerate(ACTION_LABELS):
            row = cm[i]
            log(f"  {row_label:>20} " + " ".join(f"{v:>10}" for v in row))

        log(f"  Per-class F1:")
        for label in ACTION_LABELS:
            if label in report:
                log(f"    {label:>20}: P={report[label]['precision']:.3f} "
                    f"R={report[label]['recall']:.3f} F1={report[label]['f1-score']:.3f} "
                    f"(n={report[label]['support']})")

    save_json(results, "confusion_matrices.json")
    return results


# ===================================================================
# Analysis 2: Balanced evaluation slices
# ===================================================================

def analysis_balanced_slices(all_preds):
    """Evaluate on class-balanced subsets to preempt skew criticism."""
    log("Running Analysis 2: Balanced evaluation slices")
    results = {}

    for model_name, data in all_preds.items():
        rng = random.Random(42)  # Fresh seed per model
        preds = data["predictions"]
        oracles = data["oracles"]

        # Group by oracle class
        by_class = defaultdict(list)
        for i, oracle in enumerate(oracles):
            by_class[oracle].append(i)

        # Find min class size
        class_sizes = {k: len(v) for k, v in by_class.items()}
        min_size = min(class_sizes.values())

        # Balanced subsample: take min_size from each class
        balanced_indices = []
        for cls, indices in by_class.items():
            sampled = rng.sample(indices, min(min_size, len(indices)))
            balanced_indices.extend(sampled)

        bal_preds = [preds[i] for i in balanced_indices]
        bal_oracles = [oracles[i] for i in balanced_indices]

        bal_acc = accuracy_score(bal_oracles, bal_preds)
        bal_f1 = f1_score(bal_oracles, bal_preds, average="macro",
                          labels=ACTION_LABELS, zero_division=0)

        results[model_name] = {
            "full_accuracy": accuracy_score(oracles, preds),
            "full_f1_macro": f1_score(oracles, preds, average="macro",
                                       labels=ACTION_LABELS, zero_division=0),
            "balanced_accuracy": bal_acc,
            "balanced_f1_macro": bal_f1,
            "class_sizes": class_sizes,
            "balanced_n_per_class": min_size,
            "balanced_total": len(balanced_indices),
        }

        log(f"  {model_name}: full_acc={accuracy_score(oracles, preds):.3f} "
            f"bal_acc={bal_acc:.3f} bal_f1={bal_f1:.3f} "
            f"(n_per_class={min_size})")

    save_json(results, "balanced_slices.json")
    return results


# ===================================================================
# Analysis 3: Stable vs volatile facts (the "best figure" analysis)
# ===================================================================

def analysis_volatile_vs_stable(all_preds):
    """
    Compare model performance across volatility categories.
    The key hypothesis: time-aware models should show bigger gains
    on volatile facts than stable ones.
    """
    log("Running Analysis 3: Volatile vs stable facts")
    results = {}

    volatility_levels = ["high", "medium", "low", "stable"]

    for model_name, data in all_preds.items():
        preds = data["predictions"]
        oracles = data["oracles"]
        vols = data["volatilities"]

        model_results = {}
        for vol in volatility_levels:
            indices = [i for i, v in enumerate(vols) if v == vol]
            if not indices:
                continue
            vol_preds = [preds[i] for i in indices]
            vol_oracles = [oracles[i] for i in indices]

            model_results[vol] = {
                "accuracy": accuracy_score(vol_oracles, vol_preds),
                "f1_macro": f1_score(vol_oracles, vol_preds, average="macro",
                                     labels=ACTION_LABELS, zero_division=0),
                "n": len(indices),
            }

        results[model_name] = model_results

    # Print comparison table
    log(f"\n  {'Model':<30} {'High':>8} {'Medium':>8} {'Low':>8} {'Stable':>8}")
    log(f"  {'-'*62}")
    for model_name in all_preds:
        r = results[model_name]
        vals = [f"{r.get(v, {}).get('accuracy', 0):.3f}" for v in volatility_levels]
        log(f"  {model_name:<30} " + " ".join(f"{v:>8}" for v in vals))

    # Compute "temporal gain" for each volatility level
    # (time-aware accuracy - baseline accuracy)
    log(f"\n  Temporal gain over baseline (accuracy delta):")
    baselines = {
        "gpt2": "gpt2_baseline",
        "tinyllama": "tinyllama_fullft_baseline",
    }
    for model_name in all_preds:
        if "baseline" in model_name:
            continue
        # Find matching baseline
        base_key = None
        for prefix, bname in baselines.items():
            if model_name.startswith(prefix) or (prefix == "tinyllama" and "tinyllama" in model_name):
                base_key = bname
                break
        if not base_key or base_key not in results:
            continue

        gains = {}
        for vol in volatility_levels:
            model_acc = results[model_name].get(vol, {}).get("accuracy", 0)
            base_acc = results[base_key].get(vol, {}).get("accuracy", 0)
            gains[vol] = model_acc - base_acc

        log(f"  {model_name:<30} " +
            " ".join(f"{gains.get(v, 0):>+8.3f}" for v in volatility_levels))

    save_json(results, "volatile_vs_stable.json")
    return results


# ===================================================================
# Analysis 4: Counterfactual threshold curves
# ===================================================================

def analysis_threshold_curves(all_preds):
    """
    For counterfactual groups, track how predictions change as
    elapsed time increases. Shows monotonic shift from answer_directly
    to refresh/abstain.
    """
    log("Running Analysis 4: Counterfactual threshold curves")
    results = {}

    examples = load_examples()

    for model_name, data in all_preds.items():
        preds = data["predictions"]
        elapsed = data["elapsed_seconds"]
        groups = data["counterfactual_groups"]
        oracles = data["oracles"]

        # Group examples by counterfactual group
        group_data = defaultdict(list)
        for i in range(len(preds)):
            group_data[groups[i]].append({
                "elapsed": elapsed[i],
                "pred": preds[i],
                "oracle": oracles[i],
            })

        # Sort each group by elapsed time
        for gid in group_data:
            group_data[gid].sort(key=lambda x: x["elapsed"])

        # Aggregate: for each elapsed-time bucket, what fraction of
        # predictions are each action?
        time_buckets = [
            (0, 60, "0-1m"),
            (60, 300, "1-5m"),
            (300, 1800, "5-30m"),
            (1800, 3600, "30m-1h"),
            (3600, 21600, "1-6h"),
            (21600, 86400, "6-24h"),
            (86400, 604800, "1-7d"),
            (604800, 2592000, "1w-1mo"),
            (2592000, float("inf"), "1mo+"),
        ]

        bucket_action_counts = {}
        for low, high, label in time_buckets:
            counts = defaultdict(int)
            total = 0
            for i in range(len(preds)):
                if low <= elapsed[i] < high:
                    counts[preds[i]] += 1
                    total += 1

            if total > 0:
                bucket_action_counts[label] = {
                    "counts": dict(counts),
                    "fractions": {a: counts[a] / total for a in ACTION_LABELS},
                    "n": total,
                }

        # Track switch points within groups
        switch_points = []
        for gid, group in group_data.items():
            if len(group) < 2:
                continue
            for k in range(len(group) - 1):
                if group[k]["pred"] != group[k + 1]["pred"]:
                    switch_points.append({
                        "elapsed_from": group[k]["elapsed"],
                        "elapsed_to": group[k + 1]["elapsed"],
                        "action_from": group[k]["pred"],
                        "action_to": group[k + 1]["pred"],
                    })

        # Summarize switch transitions
        transition_counts = defaultdict(int)
        for sp in switch_points:
            key = f"{sp['action_from']} -> {sp['action_to']}"
            transition_counts[key] += 1

        results[model_name] = {
            "action_by_time_bucket": bucket_action_counts,
            "n_switch_points": len(switch_points),
            "transition_counts": dict(transition_counts),
        }

        log(f"\n  {model_name}: {len(switch_points)} switch points")
        log(f"  Action distribution by time bucket:")
        log(f"  {'Bucket':<12} " + " ".join(f"{a[:8]:>10}" for a in ACTION_LABELS) + f" {'n':>6}")
        for low, high, label in time_buckets:
            if label in bucket_action_counts:
                fracs = bucket_action_counts[label]["fractions"]
                n = bucket_action_counts[label]["n"]
                log(f"  {label:<12} " +
                    " ".join(f"{fracs.get(a, 0):>10.3f}" for a in ACTION_LABELS) +
                    f" {n:>6}")
        if transition_counts:
            log(f"  Top transitions:")
            for trans, count in sorted(transition_counts.items(), key=lambda x: -x[1])[:5]:
                log(f"    {trans}: {count}")

    save_json(results, "threshold_curves.json")
    return results


# ===================================================================
# Analysis 5: Per-fact-type breakdown
# ===================================================================

def analysis_fact_type_breakdown(all_preds):
    """Break down accuracy by fact_type for each model."""
    log("Running Analysis 5: Per-fact-type breakdown")
    results = {}

    for model_name, data in all_preds.items():
        preds = data["predictions"]
        oracles = data["oracles"]
        fact_types = data["fact_types"]

        by_type = defaultdict(lambda: {"preds": [], "oracles": []})
        for i in range(len(preds)):
            by_type[fact_types[i]]["preds"].append(preds[i])
            by_type[fact_types[i]]["oracles"].append(oracles[i])

        model_results = {}
        for ft, d in sorted(by_type.items()):
            model_results[ft] = {
                "accuracy": accuracy_score(d["oracles"], d["preds"]),
                "f1_macro": f1_score(d["oracles"], d["preds"], average="macro",
                                     labels=ACTION_LABELS, zero_division=0),
                "n": len(d["preds"]),
            }

        results[model_name] = model_results

    # Print table
    fact_types_all = sorted(set(ft for data in all_preds.values() for ft in data["fact_types"]))
    log(f"\n  Accuracy by fact type:")
    log(f"  {'Model':<30} " + " ".join(f"{ft[:12]:>14}" for ft in fact_types_all))
    log(f"  {'-' * (30 + 14 * len(fact_types_all))}")
    for model_name in all_preds:
        r = results[model_name]
        vals = [f"{r.get(ft, {}).get('accuracy', 0):.3f}" for ft in fact_types_all]
        log(f"  {model_name:<30} " + " ".join(f"{v:>14}" for v in vals))

    save_json(results, "fact_type_breakdown.json")
    return results


def save_json(data, filename):
    path = ANALYSIS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log(f"  Saved {path}")


def check_model_exists(model_dir):
    """Check if a model directory exists and has the required files."""
    p = Path(model_dir)
    return p.exists() and (p / "temporal_meta.json").exists()


def main():
    start = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        f.write(f"Analysis started at {datetime.now()}\n\n")

    # Filter to models that actually exist
    available_models = []
    for name, model_dir, variant in MODELS:
        if check_model_exists(model_dir):
            available_models.append((name, model_dir, variant))
            log(f"Found model: {name} at {model_dir}")
        else:
            log(f"SKIP: {name} — {model_dir} not found")

    # Phase 1: Get predictions for all models
    log(f"\n{'='*70}")
    log(f"  PHASE 1: COLLECTING PREDICTIONS")
    log(f"{'='*70}")

    all_preds = {}
    for name, model_dir, variant in available_models:
        log(f"\nModel: {name}")
        try:
            data = get_predictions(name, model_dir, variant)
            all_preds[name] = data
        except Exception as e:
            log(f"  ERROR: {e}")
            import traceback
            log(traceback.format_exc())

    # Phase 2: Run analyses
    log(f"\n{'='*70}")
    log(f"  PHASE 2: RUNNING ANALYSES")
    log(f"{'='*70}")

    analysis_confusion_matrices(all_preds)
    analysis_balanced_slices(all_preds)
    analysis_volatile_vs_stable(all_preds)
    analysis_threshold_curves(all_preds)
    analysis_fact_type_breakdown(all_preds)

    total = time.time() - start
    log(f"\n{'='*70}")
    log(f"  COMPLETE — Total time: {total/60:.1f} minutes")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
