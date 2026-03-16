#!/usr/bin/env python3
"""
Text-time experiments: natural language time representation.

Tests whether plain text ("3 hours ago") works better than special tokens
(<dt_3h>) across model sizes. This is the key experiment for practical
contribution: text_time requires no tokenizer changes.

Experiments:
  1. GPT-2 text_time       — compare to GPT-2 time_tokens (82%)
  2. TinyLlama text_time   — the critical test (time_tokens failed)

Plus new ablations: paraphrase, raw seconds, coarse category.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("outputs/text_time_log.txt")


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


EXPERIMENTS = [
    {
        "name": "gpt2_text_time",
        "config": "configs/gpt2_text_time.yaml",
        "variant": "text_time",
    },
    {
        "name": "tinyllama_fullft_text_time",
        "config": "configs/tinyllama_fullft_text_time.yaml",
        "variant": "text_time",
    },
]


def main():
    start = time.time()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        f.write(f"Text-time experiments started at {datetime.now()}\n\n")

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

    # Summary
    log(f"\n{'='*70}")
    log(f"  TEXT-TIME RESULTS")
    log(f"{'='*70}")
    log(f"{'Experiment':<30} {'Acc':>7} {'F1':>7} {'FalseT':>7} {'SwSens':>7} {'ShufDr':>7}")
    log(f"{'-'*65}")

    all_names = [
        "gpt2_baseline", "gpt2_time_tokens", "gpt2_text_time",
        "tinyllama_fullft_baseline", "tinyllama_fullft_time_tokens",
        "tinyllama_fullft_text_time",
    ]
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
        log(f"{name:<30} {sc.get('overall_accuracy', 0):>7.3f} "
            f"{sc.get('f1_macro', 0):>7.3f} "
            f"{sc.get('false_trust_rate', 0):>7.3f} "
            f"{cs.get('switch_sensitivity', 0):>7.3f} "
            f"{ft.get('accuracy_drop', 0) if ft else 0:>7.3f}")

    # Print text-time specific ablations
    for name in ["gpt2_text_time", "tinyllama_fullft_text_time"]:
        path = results_dir / f"{name}_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            r = json.load(f)
        abl = r.get("ablations", {})
        log(f"\n  {name} ablations:")
        if "paraphrase" in abl:
            p = abl["paraphrase"]
            log(f"    Paraphrase:  orig={p['original_accuracy']:.3f}  para={p['paraphrase_accuracy']:.3f}  agreement={p['prediction_agreement']:.3f}")
        if "raw_seconds" in abl:
            rs = abl["raw_seconds"]
            log(f"    Raw seconds: orig={rs['original_accuracy']:.3f}  raw={rs['raw_seconds_accuracy']:.3f}  drop={rs['accuracy_drop']:.3f}")
        if "category" in abl:
            c = abl["category"]
            log(f"    Category:    orig={c['original_accuracy']:.3f}  cat={c['category_accuracy']:.3f}  drop={c['accuracy_drop']:.3f}")

    total = time.time() - start
    log(f"\n{'='*70}")
    log(f"  COMPLETE -- Total time: {total/60:.1f} minutes")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
