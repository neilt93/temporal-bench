#!/usr/bin/env python3
"""
Train a temporal action classification model.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/time_tokens.yaml
    python scripts/train.py --config configs/time_memory.yaml
    python scripts/train.py --variant time_tokens --model gpt2  # quick override
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_llm.training.config import TrainingConfig, ModelConfig
from temporal_llm.training.trainer import TemporalTrainer


def main():
    parser = argparse.ArgumentParser(description="Train temporal action model")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    # Quick overrides (used when no config file)
    parser.add_argument("--variant", type=str, default="baseline",
                        choices=["baseline", "text_time", "time_tokens", "time_memory"])
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (e.g., gpt2, TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        model_name = args.model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        config = TrainingConfig(
            model=ModelConfig(
                name=model_name,
                use_lora=not args.no_lora,
            ),
            data_dir=args.data_dir,
            variant=args.variant,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            output_dir=args.output_dir,
            use_wandb=not args.no_wandb,
            seed=args.seed,
        )

    print(f"Training configuration:")
    print(f"  Model: {config.model.name}")
    print(f"  Variant: {config.variant}")
    print(f"  LoRA: {config.model.use_lora}")
    print(f"  Batch size: {config.effective_batch_size} (micro={config.batch_size})")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Run name: {config.run_name}")
    print()

    trainer = TemporalTrainer(config)
    trainer.setup()
    best_path = trainer.train()

    print(f"\nTraining complete. Best model at: {best_path}")


if __name__ == "__main__":
    main()
