#!/usr/bin/env python3
"""
Overnight experiment runner.

Runs GPT-2 (fast) and TinyLlama (full) experiments back-to-back,
with evaluation and comparison.

Usage:
    python scripts/run_overnight.py
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

LOG_PATH = Path("outputs/overnight_log.txt")

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
        log(f"  stderr: {result.stderr[-500:]}")
        return False
    log(f"DONE: {desc} ({elapsed:.0f}s)")
    return True


def train(variant: str, model: str, lr: float, epochs: int,
          batch_size: int, no_lora: bool = False):
    cmd = [
        sys.executable, "scripts/train.py",
        "--variant", variant,
        "--model", model,
        "--data-dir", "data/generated",
        "--output-dir", "outputs",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--no-wandb",
        "--seed", "42",
    ]
    if no_lora:
        cmd.append("--no-lora")
    model_short = model.split("/")[-1]
    return run_cmd(cmd, f"Train {model_short} {variant}")


def evaluate(model_dir: str, variant: str, output_path: str):
    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--model-dir", model_dir,
        "--test-data", "data/generated/test.jsonl",
        "--variant", variant,
        "--output", output_path,
        "--ablations",
    ]
    return run_cmd(cmd, f"Evaluate {variant} ({model_dir})")


def get_run_name(model: str, variant: str, lr: float) -> str:
    model_short = model.split("/")[-1]
    return f"{model_short}_{variant}_lr{lr}"


def print_comparison(results_dir: Path, prefix: str, variants: list[str]):
    log(f"\n{'='*70}")
    log(f"  {prefix} COMPARISON")
    log(f"{'='*70}")
    log(f"{'Variant':<15} {'Accuracy':>10} {'F1':>10} {'False Trust':>12} {'Switch Sens':>12}")
    log(f"{'-'*60}")
    for variant in variants:
        path = results_dir / f"{prefix}_{variant}_results.json"
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            sc = r.get("stale_calibration", {})
            cs = r.get("counterfactual_sensitivity", {})
            log(f"{variant:<15} {sc.get('overall_accuracy', 0):>10.4f} "
                f"{sc.get('f1_macro', 0):>10.4f} "
                f"{sc.get('false_trust_rate', 0):>12.4f} "
                f"{cs.get('switch_sensitivity', 0):>12.4f}")
        else:
            log(f"{variant:<15} {'(no results)':>10}")


def main():
    start = time.time()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Clear log
    with open(LOG_PATH, "w") as f:
        f.write(f"Overnight experiment run started at {datetime.now()}\n\n")

    variants = ["baseline", "time_tokens", "time_memory"]
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # PHASE 1: GPT-2 experiments (fast, ~2-3 min each)
    # ================================================================
    log("\n" + "="*70)
    log("  PHASE 1: GPT-2 experiments")
    log("="*70)

    gpt2_model = "gpt2"
    gpt2_lr = 5e-4
    gpt2_epochs = 10
    gpt2_batch = 16

    for variant in variants:
        train(variant, gpt2_model, gpt2_lr, gpt2_epochs, gpt2_batch, no_lora=True)

    # Evaluate GPT-2 models
    for variant in variants:
        run_name = get_run_name(gpt2_model, variant, gpt2_lr)
        model_dir = f"outputs/{run_name}/best"
        output_path = str(results_dir / f"gpt2_{variant}_results.json")
        if Path(model_dir).exists():
            evaluate(model_dir, variant, output_path)
        else:
            log(f"SKIP eval: {model_dir} not found")

    print_comparison(results_dir, "gpt2", variants)

    # ================================================================
    # PHASE 2: TinyLlama experiments (slower, ~10-20 min each)
    # ================================================================
    log("\n" + "="*70)
    log("  PHASE 2: TinyLlama experiments")
    log("="*70)

    llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    llama_lr = 2e-4
    llama_epochs = 10
    llama_batch = 8

    for variant in variants:
        train(variant, llama_model, llama_lr, llama_epochs, llama_batch, no_lora=False)

    # Evaluate TinyLlama models
    for variant in variants:
        run_name = get_run_name(llama_model, variant, llama_lr)
        model_dir = f"outputs/{run_name}/best"
        output_path = str(results_dir / f"tinyllama_{variant}_results.json")
        if Path(model_dir).exists():
            evaluate(model_dir, variant, output_path)
        else:
            log(f"SKIP eval: {model_dir} not found")

    print_comparison(results_dir, "tinyllama", variants)

    # ================================================================
    # PHASE 3: Summary
    # ================================================================
    total = time.time() - start
    log(f"\n{'='*70}")
    log(f"  ALL EXPERIMENTS COMPLETE")
    log(f"  Total time: {total/60:.1f} minutes")
    log(f"  Results: {results_dir}")
    log(f"  Log: {LOG_PATH}")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
