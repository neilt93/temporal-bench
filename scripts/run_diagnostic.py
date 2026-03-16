#!/usr/bin/env python3
"""
Phase 1 diagnostic experiments: identify why TinyLlama LoRA fails.

4 experiments, all time_tokens variant (except one baseline control):
  1. Full fine-tune + time_tokens  — does full FT work like GPT-2?
  2. Full fine-tune + baseline     — control (does FT help without time?)
  3. LoRA r=64, all linear layers  — is it LoRA capacity?
  4. LoRA r=16, LR=1e-3            — is it learning rate?

Estimated time: ~40-60 min on RTX 4080.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("outputs/diagnostic_log.txt")


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
        log(f"  stderr: {result.stderr[-1000:]}")
        return False
    log(f"DONE: {desc} ({elapsed:.0f}s)")
    return True


def train_config(config_path: str, desc: str) -> bool:
    return run_cmd(
        [sys.executable, "scripts/train.py", "--config", config_path],
        desc,
    )


def evaluate(model_dir: str, variant: str, output_path: str) -> bool:
    return run_cmd(
        [
            sys.executable, "scripts/evaluate.py",
            "--model-dir", model_dir,
            "--test-data", "data/generated/test.jsonl",
            "--variant", variant,
            "--output", output_path,
            "--ablations",
        ],
        f"Evaluate {Path(model_dir).parent.name} ({variant})",
    )


EXPERIMENTS = [
    {
        "name": "tinyllama_fullft_time_tokens",
        "config": "configs/tinyllama_fullft_time_tokens.yaml",
        "variant": "time_tokens",
        "question": "Does full fine-tune work like GPT-2?",
    },
    {
        "name": "tinyllama_fullft_baseline",
        "config": "configs/tinyllama_fullft_baseline.yaml",
        "variant": "baseline",
        "question": "Control — does full FT help without time info?",
    },
    {
        "name": "tinyllama_lora_r64_time_tokens",
        "config": "configs/tinyllama_lora_r64_time_tokens.yaml",
        "variant": "time_tokens",
        "question": "Is LoRA capacity (r=16 to r=64) the bottleneck?",
    },
    {
        "name": "tinyllama_lora_highlr_time_tokens",
        "config": "configs/tinyllama_lora_highlr_time_tokens.yaml",
        "variant": "time_tokens",
        "question": "Is the learning rate too low for LoRA?",
    },
]


def main():
    start = time.time()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        f.write(f"Diagnostic experiments started at {datetime.now()}\n\n")

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train and evaluate each experiment
    for exp in EXPERIMENTS:
        log(f"\n{'='*70}")
        log(f"  Experiment: {exp['name']}")
        log(f"  Question: {exp['question']}")
        log(f"{'='*70}")

        ok = train_config(exp["config"], f"Train {exp['name']}")
        if not ok:
            log(f"  SKIPPING eval due to training failure")
            continue

        model_dir = f"outputs/{exp['name']}/best"
        output_path = str(results_dir / f"{exp['name']}_results.json")
        evaluate(model_dir, exp["variant"], output_path)

    # Summary table
    log(f"\n{'='*70}")
    log(f"  DIAGNOSTIC RESULTS")
    log(f"{'='*70}")
    log(f"{'Experiment':<35} {'Acc':>7} {'F1':>7} {'FalseT':>7} {'SwSens':>7} {'ShufDrop':>8}")
    log(f"{'-'*75}")

    for exp in EXPERIMENTS:
        path = results_dir / f"{exp['name']}_results.json"
        if not path.exists():
            log(f"{exp['name']:<35} {'FAILED':>7}")
            continue

        with open(path) as f:
            r = json.load(f)

        sc = r.get("stale_calibration", {})
        cs = r.get("counterfactual_sensitivity", {})
        abl = r.get("ablations", {})
        ft = abl.get("fake_time", {})

        acc = sc.get("overall_accuracy", 0)
        f1 = sc.get("f1_macro", 0)
        false_t = sc.get("false_trust_rate", 0)
        sw_sens = cs.get("switch_sensitivity", 0)
        shuf_drop = ft.get("accuracy_drop", 0) if ft else 0

        log(f"{exp['name']:<35} {acc:>7.3f} {f1:>7.3f} {false_t:>7.3f} {sw_sens:>7.3f} {shuf_drop:>8.3f}")

    # Add GPT-2 reference rows
    log(f"{'-'*75}")
    log(f"  Reference (prior results):")
    for ref_name in ["gpt2_baseline", "gpt2_time_tokens", "gpt2_time_memory"]:
        path = results_dir / f"{ref_name}_results.json"
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            sc = r.get("stale_calibration", {})
            cs = r.get("counterfactual_sensitivity", {})
            abl = r.get("ablations", {})
            ft = abl.get("fake_time", {})
            log(f"{ref_name:<35} {sc.get('overall_accuracy', 0):>7.3f} "
                f"{sc.get('f1_macro', 0):>7.3f} {sc.get('false_trust_rate', 0):>7.3f} "
                f"{cs.get('switch_sensitivity', 0):>7.3f} "
                f"{ft.get('accuracy_drop', 0) if ft else 0:>8.3f}")

    total = time.time() - start
    log(f"\n{'='*70}")
    log(f"  COMPLETE — Total time: {total/60:.1f} minutes")
    log(f"  Results: {results_dir}")
    log(f"  Log: {LOG_PATH}")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
