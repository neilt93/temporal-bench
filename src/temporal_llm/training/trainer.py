"""
Training loop for temporal action classification.

Uses HuggingFace Trainer with custom evaluation.
"""

import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ..data.dataset import TemporalDataset
from ..data.taxonomy import ACTION_LABELS
from ..models.temporal_model import TemporalActionModel
from .config import TrainingConfig


class WeightedCETrainer(Trainer):
    """Trainer subclass that applies per-sample class weights to the loss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self._class_weights = class_weights  # dict: first_token_id -> weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Standard causal LM loss with class weighting
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if self._class_weights is not None:
            # Build per-token weight tensor: default=1.0, but action tokens
            # get their class weight
            weight_tensor = torch.ones_like(shift_labels, dtype=shift_logits.dtype)
            for token_id, w in self._class_weights.items():
                weight_tensor[shift_labels == token_id] = w

            # Compute per-token CE loss
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            # Apply weights and mask
            loss = loss.view(shift_labels.shape)
            mask = shift_labels != -100
            loss = (loss * weight_tensor * mask).sum() / mask.sum()
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return (loss, outputs) if return_outputs else loss


class TemporalTrainer:
    """
    Manages the full training pipeline for a single model variant.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.temporal_model = None
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        """Initialize model, tokenizer, and datasets."""
        self.temporal_model = TemporalActionModel(
            self.config.model, variant=self.config.variant
        )
        self.model, self.tokenizer = self.temporal_model.build()

        data_dir = Path(self.config.data_dir)
        self.train_dataset = TemporalDataset(
            data_dir / "train.jsonl",
            self.tokenizer,
            variant=self.config.variant,
            max_length=self.config.model.max_length,
        )
        self.val_dataset = TemporalDataset(
            data_dir / "val.jsonl",
            self.tokenizer,
            variant=self.config.variant,
            max_length=self.config.model.max_length,
        )

        # Build first-token-to-action mapping for eval metrics
        self._token_to_action = {}
        self._action_to_first_token = {}
        for action in ACTION_LABELS:
            token_ids = self.tokenizer.encode(" " + action, add_special_tokens=False)
            if token_ids:
                self._token_to_action[token_ids[0]] = action
                self._action_to_first_token[action] = token_ids[0]

        # Compute class weights from training data label distribution
        action_counts = Counter(
            ex["correct_action"] for ex in self.train_dataset.examples
        )
        total = sum(action_counts.values())
        n_classes = len(ACTION_LABELS)
        self._class_weights = {}
        for action in ACTION_LABELS:
            count = action_counts.get(action, 1)
            first_tok = self._action_to_first_token.get(action)
            if first_tok is not None:
                # Inverse frequency weighting: weight = total / (n_classes * count)
                self._class_weights[first_tok] = total / (n_classes * count)

        print(f"Train: {len(self.train_dataset)} examples")
        print(f"Val:   {len(self.val_dataset)} examples")
        print(f"Class weights: " + ", ".join(
            f"{a}={self._class_weights.get(self._action_to_first_token.get(a, -1), 0):.2f}"
            for a in ACTION_LABELS
        ))

    @staticmethod
    def _preprocess_logits_for_eval(logits, labels):
        """Reduce full vocab logits to argmax indices to avoid OOM during eval."""
        return logits.argmax(dim=-1)

    def _compute_metrics(self, eval_pred):
        """Compute classification metrics from generative outputs."""
        preds, labels = eval_pred

        pred_actions = []
        true_actions = []

        for i in range(labels.shape[0]):
            valid_mask = labels[i] != -100
            if not valid_mask.any():
                continue
            first_valid = int(valid_mask.nonzero()[0][0])
            if first_valid == 0:
                continue
            pred_token = int(preds[i, first_valid - 1])
            true_token = int(labels[i, first_valid])

            # Map first token back to action name
            pred_action = self._token_to_action.get(pred_token, "unknown")
            true_action = self._token_to_action.get(true_token, "unknown")
            pred_actions.append(pred_action)
            true_actions.append(true_action)

        if not true_actions:
            return {"accuracy": 0.0, "f1_macro": 0.0}

        accuracy = accuracy_score(true_actions, pred_actions)
        f1_mac = f1_score(
            true_actions, pred_actions,
            average="macro", labels=ACTION_LABELS, zero_division=0,
        )
        return {"accuracy": accuracy, "f1_macro": f1_mac}

    def train(self) -> str:
        """Run training. Returns path to best checkpoint."""
        output_dir = Path(self.config.output_dir) / self.config.run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        self.config.to_yaml(output_dir / "config.yaml")

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler,
            optim=self.config.optimizer,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.model.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if self.config.model.gradient_checkpointing else None,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.eval_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            seed=self.config.seed,
            report_to="wandb" if self.config.use_wandb else "none",
            run_name=self.config.run_name if self.config.use_wandb else None,
            dataloader_num_workers=0,  # Windows compatibility
            remove_unused_columns=False,
        )

        trainer = WeightedCETrainer(
            class_weights=self._class_weights,
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=TemporalDataset.collate_fn,
            compute_metrics=self._compute_metrics,
            preprocess_logits_for_metrics=self._preprocess_logits_for_eval,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        # Resume from latest checkpoint if one exists
        last_ckpt = None
        ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            last_ckpt = str(ckpts[-1])
            print(f"Resuming from {last_ckpt}")

        trainer.train(resume_from_checkpoint=last_ckpt)

        best_dir = str(output_dir / "best")
        self.temporal_model.save(best_dir)
        print(f"Best model saved to {best_dir}")

        return best_dir

    def evaluate_detailed(self, test_path: str | Path) -> dict:
        """
        Run detailed evaluation on a test set.
        Returns per-action and per-volatility metrics.
        """
        # Load raw examples (not the training dataset — we need metadata + prompt-only text)
        examples = []
        with open(test_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        device = next(self.model.parameters()).device
        self.model.eval()

        input_field = f"input_{self.config.variant}"
        all_preds = []
        all_labels = []
        all_volatility = []
        all_elapsed = []
        all_fact_types = []

        # Process in batches — tokenize prompt-only text for generation
        bs = self.config.batch_size
        for i in range(0, len(examples), bs):
            batch_exs = examples[i : i + bs]
            texts = [ex[input_field] for ex in batch_exs]

            # Left-pad for generation (set on tokenizer, not as kwarg)
            self.tokenizer.padding_side = "left"
            enc = self.tokenizer(
                texts,
                max_length=self.config.model.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            self.tokenizer.padding_side = "right"
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            preds = self.temporal_model.predict_action(input_ids, attention_mask)

            all_preds.extend(preds)
            all_labels.extend(ex["correct_action"] for ex in batch_exs)
            all_volatility.extend(ex["volatility"] for ex in batch_exs)
            all_elapsed.extend(ex["elapsed_seconds"] for ex in batch_exs)
            all_fact_types.extend(ex["fact_type"] for ex in batch_exs)

        results = {
            "overall": {
                "accuracy": accuracy_score(all_labels, all_preds),
                "f1_macro": f1_score(all_labels, all_preds, average="macro",
                                     labels=ACTION_LABELS, zero_division=0),
                "f1_weighted": f1_score(all_labels, all_preds, average="weighted",
                                        labels=ACTION_LABELS, zero_division=0),
                "classification_report": classification_report(
                    all_labels, all_preds, labels=ACTION_LABELS, zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(
                    all_labels, all_preds, labels=ACTION_LABELS,
                ).tolist(),
            },
        }

        # Per-volatility metrics
        volatility_results = {}
        for vol in set(all_volatility):
            mask = [v == vol for v in all_volatility]
            vol_preds = [p for p, m in zip(all_preds, mask) if m]
            vol_labels = [la for la, m in zip(all_labels, mask) if m]
            if vol_preds:
                volatility_results[vol] = {
                    "accuracy": accuracy_score(vol_labels, vol_preds),
                    "f1_macro": f1_score(vol_labels, vol_preds, average="macro",
                                         labels=ACTION_LABELS, zero_division=0),
                    "n": len(vol_preds),
                }
        results["by_volatility"] = volatility_results

        # Per-action metrics
        per_action = {}
        for action in ACTION_LABELS:
            tp = sum(1 for p, la in zip(all_preds, all_labels) if p == action and la == action)
            fp = sum(1 for p, la in zip(all_preds, all_labels) if p == action and la != action)
            fn = sum(1 for p, la in zip(all_preds, all_labels) if p != action and la == action)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            per_action[action] = {
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
                "support": tp + fn,
            }
        results["by_action"] = per_action

        return results
