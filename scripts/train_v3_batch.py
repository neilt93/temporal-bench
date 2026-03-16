#!/usr/bin/env python3
"""
Train all v3 headline models and evaluate each one.

Runs 4 models sequentially:
1. GPT-2 baseline (v3 data)
2. GPT-2 text_time (v3 data)
3. TinyLlama LoRA baseline (v3 data)
4. TinyLlama LoRA text_time (v3 data)

After each training run, evaluates on v3 test set with full ablations.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CONFIGS = [
    ("gpt2_v3_baseline", "configs/gpt2_v3_baseline.yaml", "baseline"),
    ("gpt2_v3_text_time", "configs/gpt2_v3_text_time.yaml", "text_time"),
    ("tinyllama_lora_v3_baseline", "configs/tinyllama_lora_v3_baseline.yaml", "baseline"),
    ("tinyllama_lora_v3_text_time", "configs/tinyllama_lora_v3_text_time.yaml", "text_time"),
]

TEST_DATA = "data/v3/test.jsonl"
RESULTS_DIR = Path("outputs/results")


def run_cmd(cmd, description):
    """Run a command, streaming output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    result = subprocess.run(
        cmd, shell=True,
        stdout=sys.stdout, stderr=sys.stderr,
    )
    return result.returncode


def main():
    start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, config, variant in CONFIGS:
        t0 = time.time()

        # Train
        rc = run_cmd(
            f"python scripts/train.py --config {config}",
            f"Training {name}",
        )
        if rc != 0:
            print(f"ERROR: Training {name} failed with exit code {rc}")
            continue

        train_time = time.time() - t0
        print(f"\nTraining {name} completed in {train_time/60:.1f} minutes")

        # Find the best model directory
        model_dir = f"outputs/{name}/best"
        if not Path(model_dir).exists():
            # Try alternative naming
            alt_dir = None
            for d in Path("outputs").iterdir():
                if d.is_dir() and name.replace("_", "") in d.name.replace("_", "").replace("-", ""):
                    best = d / "best"
                    if best.exists():
                        alt_dir = str(best)
                        break
            if alt_dir:
                model_dir = alt_dir
            else:
                print(f"WARNING: Could not find best model for {name}, skipping eval")
                continue

        # Evaluate
        output_file = str(RESULTS_DIR / f"{name}_results.json")
        rc = run_cmd(
            f"python scripts/evaluate.py --model-dir {model_dir} "
            f"--test-data {TEST_DATA} --variant {variant} --ablations "
            f"--output {output_file}",
            f"Evaluating {name}",
        )
        if rc != 0:
            print(f"WARNING: Evaluation of {name} failed")
            continue

        # Print key metrics
        try:
            with open(output_file) as f:
                results = json.load(f)
            sc = results.get("stale_calibration", {})
            cs = results.get("counterfactual_sensitivity", {})
            ft = results.get("ablations", {}).get("fake_time", {})
            print(f"\n  {name} results:")
            print(f"    Accuracy:          {sc.get('overall_accuracy', 0):.3f}")
            print(f"    Switch sensitivity: {cs.get('switch_sensitivity', 0):.3f}")
            print(f"    False trust rate:   {sc.get('false_trust_rate', 0):.3f}")
            print(f"    Shuffled delta:     {ft.get('accuracy_drop', 0):.3f}")
        except Exception as e:
            print(f"  Could not read results: {e}")

    total = time.time() - start
    print(f"\n{'='*60}")
    print(f"  ALL DONE — Total time: {total/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
