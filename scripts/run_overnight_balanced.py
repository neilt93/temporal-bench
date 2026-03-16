#!/usr/bin/env python3
"""
Overnight training batch: class-balanced experiments.

Trains on oversampled data to fix 3-class collapse (retrieve_memory,
ask_clarify, abstain are never predicted in current models).

Experiments:
  1. GPT-2 text_time on balanced data
  2. TinyLlama LoRA text_time on balanced data (the key one)
  3. TinyLlama LoRA baseline on balanced data (control)

Then evaluates all three with full ablations.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("outputs/balanced_log.txt")


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_cmd(cmd: list[str], desc: str) -> bool:
    log(f"START: {desc}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log(f"FAILED: {desc} ({elapsed:.0f}s)")
        log(f"  stderr (last 2000 chars): {result.stderr[-2000:]}")
        return False
    log(f"DONE: {desc} ({elapsed:.0f}s)")
    return True


EXPERIMENTS = [
    {
        "name": "gpt2_balanced_text_time",
        "config": "configs/gpt2_balanced_text_time.yaml",
        "variant": "text_time",
    },
    {
        "name": "tinyllama_lora_balanced_text_time",
        "config": "configs/tinyllama_lora_balanced_text_time.yaml",
        "variant": "text_time",
    },
    {
        "name": "tinyllama_lora_balanced_baseline",
        "config": "configs/tinyllama_lora_balanced_baseline.yaml",
        "variant": "baseline",
    },
]


def main():
    start = time.time()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        f.write(f"Balanced training experiments started at {datetime.now()}\n\n")

    for exp in EXPERIMENTS:
        log(f"\n{'='*70}")
        log(f"  {exp['name']}")
        log(f"{'='*70}")

        ok = run_cmd(
            [sys.executable, "scripts/train.py", "--config", exp["config"]],
            f"Train {exp['name']}",
        )
        if not ok:
            continue

        model_dir = f"outputs/{exp['name']}/best"
        output_path = str(results_dir / f"{exp['name']}_results.json")
        run_cmd(
            [
                sys.executable, "scripts/evaluate.py",
                "--model-dir", model_dir,
                "--test-data", "data/generated/test.jsonl",
                "--variant", exp["variant"],
                "--output", output_path,
                "--ablations",
            ],
            f"Evaluate {exp['name']}",
        )

    # Summary table
    log(f"\n{'='*70}")
    log(f"  BALANCED TRAINING RESULTS")
    log(f"{'='*70}")

    # Include both original and balanced results for comparison
    all_names = [
        "gpt2_text_time", "gpt2_balanced_text_time",
        "tinyllama_lora_midlr_text_time_savemods", "tinyllama_lora_balanced_text_time",
        "tinyllama_lora_balanced_baseline",
    ]

    log(f"{'Experiment':<45} {'Acc':>7} {'F1':>7} {'FalseT':>7} {'SwSens':>7} {'ShufDr':>7}")
    log(f"{'-'*80}")

    for name in all_names:
        path = results_dir / f"{name}_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            r = json.load(f)
        sc = r.get("stale_calibration", {})
        cs = r.get("counterfactual_sensitivity", {})
        abl = r.get("ablations", {})
        ft = abl.get("fake_time", {})
        log(f"{name:<45} {sc.get('overall_accuracy', 0):>7.3f} "
            f"{sc.get('f1_macro', 0):>7.3f} "
            f"{sc.get('false_trust_rate', 0):>7.3f} "
            f"{cs.get('switch_sensitivity', 0):>7.3f} "
            f"{ft.get('accuracy_drop', 0) if ft else 0:>7.3f}")

    # Per-class F1 comparison for balanced vs original
    log(f"\n  Per-class comparison (balanced vs original):")
    for name in all_names:
        path = results_dir / f"{name}_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            r = json.load(f)
        # We need to re-run predictions to get per-class metrics
        # The confusion_matrices.json from analysis has this data
        cm_path = Path("outputs/analysis/confusion_matrices.json")
        if cm_path.exists():
            with open(cm_path) as f:
                cm_data = json.load(f)
            if name in cm_data:
                pc = cm_data[name].get("per_class", {})
                log(f"\n  {name}:")
                for cls in ["answer_directly", "refresh", "retrieve_memory", "ask_clarify", "abstain"]:
                    if cls in pc:
                        log(f"    {cls:>20}: F1={pc[cls]['f1']:.3f} (n={pc[cls]['support']})")

    total = time.time() - start
    log(f"\n{'='*70}")
    log(f"  COMPLETE -- Total time: {total/60:.1f} minutes")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
