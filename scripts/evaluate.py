#!/usr/bin/env python3
"""
Evaluate a trained model on all evals and ablations.

Usage:
    python scripts/evaluate.py --model-dir outputs/TinyLlama-1.1B-Chat-v1.0_time_tokens/best \
        --test-data data/generated/test.jsonl --variant time_tokens
    python scripts/evaluate.py --model-dir outputs/baseline/best --variant baseline --ablations
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_llm.training.config import ModelConfig
from temporal_llm.models.temporal_model import TemporalActionModel
from temporal_llm.eval.evaluator import TemporalEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate temporal action model")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to saved model directory")
    parser.add_argument("--test-data", type=str, default="data/generated/test.jsonl")
    parser.add_argument("--variant", type=str, default="time_tokens",
                        choices=["baseline", "text_time", "time_tokens", "time_memory"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: model-dir/eval_results.json)")
    parser.add_argument("--ablations", action="store_true",
                        help="Also run ablation studies")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model metadata to get the base model name
    meta_path = Path(args.model_dir) / "temporal_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        base_model_name = meta.get("base_model_name", args.model_dir)
    else:
        base_model_name = args.model_dir

    print(f"Loading model from {args.model_dir}...")
    config = ModelConfig(name=base_model_name)
    model = TemporalActionModel.load(args.model_dir, config, variant=args.variant)

    print(f"Running evaluations on {args.test_data}...")
    evaluator = TemporalEvaluator(model, device=args.device, batch_size=args.batch_size)

    results = evaluator.run_all(args.test_data, variant=args.variant)

    if args.ablations:
        print("\nRunning ablation studies...")
        results["ablations"] = evaluator.run_all_ablations(args.test_data, variant=args.variant)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    if "stale_calibration" in results:
        sc = results["stale_calibration"]
        print(f"\nStale-Context Calibration:")
        print(f"  Accuracy:              {sc['overall_accuracy']:.4f}")
        print(f"  Macro F1:              {sc['f1_macro']:.4f}")
        print(f"  Refresh F1:            {sc['refresh_f1']:.4f}")
        print(f"  False trust rate:      {sc['false_trust_rate']:.4f}")
        print(f"  Unnecessary refresh:   {sc['unnecessary_refresh_rate']:.4f}")

    if "counterfactual_sensitivity" in results:
        cs = results["counterfactual_sensitivity"]
        print(f"\nCounterfactual Sensitivity:")
        print(f"  Switch sensitivity:    {cs['switch_sensitivity']:.4f}")
        print(f"  Switch specificity:    {cs['switch_specificity']:.4f}")
        print(f"  Switch accuracy:       {cs['switch_accuracy']:.4f}")

    if "memory_retrieval" in results:
        mr = results["memory_retrieval"]
        if "error" not in mr:
            print(f"\nMemory Retrieval:")
            print(f"  Accuracy:              {mr['accuracy']:.4f}")
            print(f"  Retrieve F1:           {mr['retrieve_f1']:.4f}")

    if "deadline_awareness" in results:
        da = results["deadline_awareness"]
        if "error" not in da:
            print(f"\nDeadline Awareness:")
            print(f"  Overall accuracy:      {da['overall_accuracy']:.4f}")
            print(f"  Tight deadline acc:    {da['tight_deadline_accuracy']:.4f}")
            print(f"  Relaxed deadline acc:  {da['relaxed_deadline_accuracy']:.4f}")

    if "ablations" in results:
        abl = results["ablations"]
        if "fake_time" in abl:
            ft = abl["fake_time"]
            print(f"\nAblation - Real vs Fake Time:")
            print(f"  Real time accuracy:    {ft['real_time_accuracy']:.4f}")
            print(f"  Shuffled accuracy:     {ft['shuffled_time_accuracy']:.4f}")
            print(f"  Accuracy drop:         {ft['accuracy_drop']:.4f}")

    # Save results
    output_path = args.output or str(Path(args.model_dir) / "eval_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
