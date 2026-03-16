#!/usr/bin/env python3
"""
V2 pipeline batch: train and evaluate baselines + time_tokens with
the fixed pipeline (class-weighted loss, F1 model selection).

GPT-2 and TinyLlama text_time v2 are already done.
This script runs: baselines + GPT-2 time_tokens v2.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("outputs/v2_batch_log.txt")


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
        log(f"  stderr: {result.stderr[-2000:]}")
        return False
    log(f"DONE: {desc} ({elapsed:.0f}s)")
    return True


EXPERIMENTS = [
    {
        "name": "gpt2_v2_baseline",
        "config": "configs/gpt2_v2_baseline.yaml",
        "variant": "baseline",
    },
    {
        "name": "gpt2_v2_time_tokens",
        "config": "configs/gpt2_v2_time_tokens.yaml",
        "variant": "time_tokens",
        "skip_if_no_config": True,
    },
    {
        "name": "tinyllama_lora_v2_baseline",
        "config": "configs/tinyllama_lora_v2_baseline.yaml",
        "variant": "baseline",
    },
]


def main():
    start = time.time()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        f.write(f"V2 batch started at {datetime.now()}\n\n")

    for exp in EXPERIMENTS:
        if exp.get("skip_if_no_config") and not Path(exp["config"]).exists():
            log(f"SKIP: {exp['name']} (config not found)")
            continue

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

    # Summary
    log(f"\n{'='*70}")
    log(f"  V2 RESULTS SUMMARY")
    log(f"{'='*70}")

    all_names = [
        "gpt2_v2_baseline", "gpt2_v2_text_time",
        "tinyllama_lora_v2_baseline", "tinyllama_lora_v2_text_time",
    ]

    log(f"{'Experiment':<40} {'Acc':>7} {'F1':>7} {'FalseT':>7} {'SwSens':>7} {'ShufDr':>7}")
    log(f"{'-'*75}")

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
        log(f"{name:<40} {sc.get('overall_accuracy', 0):>7.3f} "
            f"{sc.get('f1_macro', 0):>7.3f} "
            f"{sc.get('false_trust_rate', 0):>7.3f} "
            f"{cs.get('switch_sensitivity', 0):>7.3f} "
            f"{ft.get('accuracy_drop', 0) if ft else 0:>7.3f}")

    total = time.time() - start
    log(f"\n{'='*70}")
    log(f"  COMPLETE -- Total time: {total/60:.1f} minutes")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
