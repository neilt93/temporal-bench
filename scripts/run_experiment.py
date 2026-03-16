#!/usr/bin/env python3
"""
End-to-end experiment runner.

Runs the full pipeline: generate data -> train all variants -> evaluate -> plot.

Usage:
    python scripts/run_experiment.py                    # Full experiment
    python scripts/run_experiment.py --quick             # Quick run with GPT-2 + fewer examples
    python scripts/run_experiment.py --skip-train        # Only evaluate + plot
    python scripts/run_experiment.py --model gpt2 --epochs 3  # Custom model
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_cmd(cmd: list[str], desc: str):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with return code {result.returncode}")
        sys.exit(1)
    return result


def get_run_name(model: str, variant: str, lr: float) -> str:
    """Compute the run name the same way TrainingConfig.run_name does."""
    model_short = model.split("/")[-1]
    return f"{model_short}_{variant}_lr{lr}"


def main():
    parser = argparse.ArgumentParser(description="Run full temporal LLM experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: GPT-2, fewer examples, fewer epochs")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name")
    parser.add_argument("--examples-per-type", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set defaults based on mode
    if args.quick:
        model = args.model or "gpt2"
        examples_per_type = args.examples_per_type or 10
        epochs = args.epochs or 3
        batch_size = args.batch_size or 4
        lr = args.lr or 5e-4
    else:
        model = args.model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        examples_per_type = args.examples_per_type or 20
        epochs = args.epochs or 5
        batch_size = args.batch_size or 8
        lr = args.lr or 2e-4

    # Quick mode uses full finetune (no LoRA) for GPT-2
    no_lora = args.quick and model == "gpt2"

    variants = ["baseline", "time_tokens", "time_memory"]
    results_dir = Path(args.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Step 1: Generate data
    if not args.skip_data:
        run_cmd([
            sys.executable, "scripts/generate_data.py",
            "--output", args.data_dir,
            "--examples-per-type", str(examples_per_type),
            "--seed", str(args.seed),
        ], "Step 1: Generate dataset")

    # Step 2: Train all variants
    if not args.skip_train:
        for variant in variants:
            train_cmd = [
                sys.executable, "scripts/train.py",
                "--variant", variant,
                "--model", model,
                "--data-dir", args.data_dir,
                "--output-dir", args.output_dir,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--lr", str(lr),
                "--seed", str(args.seed),
            ]
            if args.no_wandb:
                train_cmd.append("--no-wandb")
            if no_lora:
                train_cmd.append("--no-lora")

            run_cmd(train_cmd, f"Step 2: Train {variant} variant")

    # Step 3: Evaluate all variants
    if not args.skip_eval:
        test_data = f"{args.data_dir}/test.jsonl"

        for variant in variants:
            run_name = get_run_name(model, variant, lr)
            model_dir = f"{args.output_dir}/{run_name}/best"
            output_path = str(results_dir / f"{variant}_results.json")

            if not Path(model_dir).exists():
                print(f"WARNING: Model not found at {model_dir}, skipping {variant}")
                continue

            run_cmd([
                sys.executable, "scripts/evaluate.py",
                "--model-dir", model_dir,
                "--test-data", test_data,
                "--variant", variant,
                "--output", output_path,
                "--ablations",
            ], f"Step 3: Evaluate {variant} variant")

    # Step 4: Generate plots
    print(f"\n{'='*60}")
    print(f"  Step 4: Generate plots")
    print(f"{'='*60}\n")

    try:
        from temporal_llm.analysis.plots import plot_results, generate_latex_table
        plot_results(str(results_dir))

        latex = generate_latex_table(str(results_dir))
        latex_path = results_dir / "main_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to {latex_path}")
    except Exception as e:
        print(f"Plot generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Results: {results_dir}")
    print(f"{'='*60}")

    # Print comparison table
    print("\nQuick comparison:")
    print(f"{'Variant':<15} {'Accuracy':>10} {'F1':>10} {'False Trust':>12} {'Switch Sens':>12}")
    print("-" * 60)
    for variant in variants:
        path = results_dir / f"{variant}_results.json"
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            sc = r.get("stale_calibration", {})
            cs = r.get("counterfactual_sensitivity", {})
            print(f"{variant:<15} {sc.get('overall_accuracy', 0):>10.4f} "
                  f"{sc.get('f1_macro', 0):>10.4f} "
                  f"{sc.get('false_trust_rate', 0):>12.4f} "
                  f"{cs.get('switch_sensitivity', 0):>12.4f}")


if __name__ == "__main__":
    main()
