#!/usr/bin/env python3
"""
Re-run TinyLlama LoRA experiments after fixing frozen embeddings bug.

Only re-trains time_tokens and time_memory (baseline unaffected by fix).
Re-evaluates all three variants with the left-padding fix.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

LOG_PATH = Path("outputs/rerun_tinyllama_log.txt")


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


def train(variant: str, model: str, lr: float, epochs: int, batch_size: int):
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


def main():
    start = time.time()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        f.write(f"TinyLlama re-run started at {datetime.now()}\n")
        f.write(f"Fix: modules_to_save for LoRA + left-padding for generation\n\n")

    llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    llama_lr = 2e-4
    llama_epochs = 10
    llama_batch = 8
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Only re-train time_tokens and time_memory (baseline unaffected by embed fix)
    for variant in ["time_tokens", "time_memory"]:
        train(variant, llama_model, llama_lr, llama_epochs, llama_batch)

    # Re-evaluate all three (baseline benefits from left-padding fix)
    for variant in ["baseline", "time_tokens", "time_memory"]:
        model_short = llama_model.split("/")[-1]
        run_name = f"{model_short}_{variant}_lr{llama_lr}"
        model_dir = f"outputs/{run_name}/best"
        output_path = str(results_dir / f"tinyllama_{variant}_results.json")
        if Path(model_dir).exists():
            evaluate(model_dir, variant, output_path)
        else:
            log(f"SKIP eval: {model_dir} not found")

    # Print comparison
    log(f"\n{'='*70}")
    log(f"  TinyLlama COMPARISON (post-fix)")
    log(f"{'='*70}")
    log(f"{'Variant':<15} {'Accuracy':>10} {'F1':>10} {'False Trust':>12} {'Switch Sens':>12}")
    log(f"{'-'*60}")

    for variant in ["baseline", "time_tokens", "time_memory"]:
        path = results_dir / f"tinyllama_{variant}_results.json"
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            sc = r.get("stale_calibration", {})
            cs = r.get("counterfactual_sensitivity", {})
            log(f"{variant:<15} {sc.get('overall_accuracy', 0):>10.4f} "
                f"{sc.get('f1_macro', 0):>10.4f} "
                f"{sc.get('false_trust_rate', 0):>12.4f} "
                f"{cs.get('switch_sensitivity', 0):>12.4f}")

    total = time.time() - start
    log(f"\n{'='*70}")
    log(f"  COMPLETE — Total time: {total/60:.1f} minutes")
    log(f"  Results: {results_dir}")
    log(f"  Log: {LOG_PATH}")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
