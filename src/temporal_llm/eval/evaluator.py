"""
Evaluation harness for temporal action classification.

Implements all four evaluations from the research plan:
1. Counterfactual time sensitivity
2. Stale-context calibration
3. Memory retrieval by age
4. Deadline-aware action selection

Plus ablation support:
A. Position vs time (real vs shuffled/fake time)
B. Coarse vs fine time bins
C. Time at turn level vs memory-entry level
D. Stable facts vs volatile facts
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from ..data.taxonomy import (
    Action, Volatility, ACTION_LABELS, FACT_TYPE_MAP,
    seconds_to_bucket, seconds_to_token, seconds_to_human,
    seconds_to_raw_text, seconds_to_paraphrase, seconds_to_category,
)
from ..data.dataset import TemporalDataset, TemporalEvalDataset
from ..models.temporal_model import TemporalActionModel


class TemporalEvaluator:
    """
    Runs all evaluations and ablations for a trained model.
    """

    def __init__(
        self,
        model: TemporalActionModel,
        device: str = "cuda",
        batch_size: int = 16,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.model.model.to(device)
        self.model.model.eval()

    def _predict_batch(self, texts: list[str]) -> list[str]:
        """Tokenize a batch of input texts and predict actions."""
        # Left-padding is required for correct batched generation
        prev_side = self.model.tokenizer.padding_side
        self.model.tokenizer.padding_side = "left"
        enc = self.model.tokenizer(
            texts,
            max_length=self.model.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        self.model.tokenizer.padding_side = prev_side
        return self.model.predict_action(enc["input_ids"], enc["attention_mask"])

    def _predict_all(self, texts: list[str]) -> list[str]:
        """Predict actions for all texts, batched."""
        all_preds = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            preds = self._predict_batch(batch)
            all_preds.extend(preds)
        return all_preds

    # -------------------------------------------------------------------
    # Eval 1: Counterfactual time sensitivity
    # -------------------------------------------------------------------

    def eval_counterfactual_sensitivity(
        self,
        test_path: str | Path,
        variant: str = "time_tokens",
    ) -> dict:
        """
        Hold text fixed, change only elapsed time.
        Measure whether the model's action changes appropriately.

        Key metric: for counterfactual pairs where the oracle action differs,
        does the model also change its prediction?
        """
        examples = self._load_examples(test_path)

        # Group by counterfactual group
        groups = defaultdict(list)
        for ex in examples:
            groups[ex["counterfactual_group"]].append(ex)

        total_pairs = 0
        correct_switches = 0  # model switched when oracle switched
        incorrect_static = 0  # model didn't switch when oracle did
        correct_static = 0    # model correctly didn't switch when oracle didn't
        incorrect_switches = 0  # model switched when oracle didn't

        # For each group, compare all pairs
        for group_id, group_exs in groups.items():
            # Get predictions for all examples in the group
            texts = [self._get_input_text(ex, variant) for ex in group_exs]
            preds = self._predict_all(texts)
            oracles = [ex["correct_action"] for ex in group_exs]

            # Compare all pairs within the group
            for i in range(len(group_exs)):
                for j in range(i + 1, len(group_exs)):
                    oracle_changed = oracles[i] != oracles[j]
                    pred_changed = preds[i] != preds[j]
                    total_pairs += 1

                    if oracle_changed and pred_changed:
                        correct_switches += 1
                    elif oracle_changed and not pred_changed:
                        incorrect_static += 1
                    elif not oracle_changed and not pred_changed:
                        correct_static += 1
                    else:
                        incorrect_switches += 1

        # Metrics
        switch_sensitivity = (
            correct_switches / (correct_switches + incorrect_static)
            if (correct_switches + incorrect_static) > 0 else 0
        )
        switch_specificity = (
            correct_static / (correct_static + incorrect_switches)
            if (correct_static + incorrect_switches) > 0 else 0
        )

        return {
            "total_pairs": total_pairs,
            "correct_switches": correct_switches,
            "incorrect_static": incorrect_static,
            "correct_static": correct_static,
            "incorrect_switches": incorrect_switches,
            "switch_sensitivity": switch_sensitivity,  # P(model switches | oracle switches)
            "switch_specificity": switch_specificity,  # P(model static | oracle static)
            "switch_accuracy": (
                (correct_switches + correct_static) / total_pairs
                if total_pairs > 0 else 0
            ),
        }

    # -------------------------------------------------------------------
    # Eval 2: Stale-context calibration
    # -------------------------------------------------------------------

    def eval_stale_calibration(
        self,
        test_path: str | Path,
        variant: str = "time_tokens",
    ) -> dict:
        """
        Test whether the model knows when not to trust prior context.

        Metrics:
        - Refresh precision/recall
        - False trust rate (answered directly when should have refreshed)
        - Unnecessary refresh rate (refreshed when could have answered)
        """
        examples = self._load_examples(test_path)

        texts = [self._get_input_text(ex, variant) for ex in examples]
        preds = self._predict_all(texts)
        oracles = [ex["correct_action"] for ex in examples]

        # Core metrics
        overall_acc = accuracy_score(oracles, preds)
        f1_macro = f1_score(oracles, preds, average="macro",
                            labels=ACTION_LABELS, zero_division=0)

        # Refresh-specific metrics
        refresh_tp = sum(1 for p, o in zip(preds, oracles)
                        if p == "refresh" and o == "refresh")
        refresh_fp = sum(1 for p, o in zip(preds, oracles)
                        if p == "refresh" and o != "refresh")
        refresh_fn = sum(1 for p, o in zip(preds, oracles)
                        if p != "refresh" and o == "refresh")
        refresh_prec = refresh_tp / (refresh_tp + refresh_fp) if (refresh_tp + refresh_fp) > 0 else 0
        refresh_recall = refresh_tp / (refresh_tp + refresh_fn) if (refresh_tp + refresh_fn) > 0 else 0

        # False trust: model said answer_directly but should have refreshed/abstained
        false_trust = sum(
            1 for p, o in zip(preds, oracles)
            if p == "answer_directly" and o in ("refresh", "abstain")
        )
        should_not_trust = sum(1 for o in oracles if o in ("refresh", "abstain"))
        false_trust_rate = false_trust / should_not_trust if should_not_trust > 0 else 0

        # Unnecessary refresh: model said refresh but answer_directly was fine
        unnecessary_refresh = sum(
            1 for p, o in zip(preds, oracles)
            if p == "refresh" and o == "answer_directly"
        )
        could_answer = sum(1 for o in oracles if o == "answer_directly")
        unnecessary_refresh_rate = unnecessary_refresh / could_answer if could_answer > 0 else 0

        # Confusion matrix
        cm = confusion_matrix(oracles, preds, labels=ACTION_LABELS)

        # Per-class metrics
        per_class = {}
        for idx, label in enumerate(ACTION_LABELS):
            tp = cm[idx, idx]
            fp = cm[:, idx].sum() - tp
            fn = cm[idx, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            support = int(cm[idx, :].sum())
            per_class[label] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": support,
            }

        # Per-volatility confusion matrices
        vol_confusion = {}
        for ex, pred in zip(examples, preds):
            vol = ex.get("volatility", "unknown")
            if vol not in vol_confusion:
                vol_confusion[vol] = {"oracles": [], "preds": []}
            vol_confusion[vol]["oracles"].append(ex["correct_action"])
            vol_confusion[vol]["preds"].append(pred)

        per_volatility_cm = {}
        for vol, data in vol_confusion.items():
            vcm = confusion_matrix(data["oracles"], data["preds"], labels=ACTION_LABELS)
            per_volatility_cm[vol] = vcm.tolist()

        return {
            "overall_accuracy": overall_acc,
            "f1_macro": f1_macro,
            "refresh_precision": refresh_prec,
            "refresh_recall": refresh_recall,
            "refresh_f1": (
                2 * refresh_prec * refresh_recall / (refresh_prec + refresh_recall)
                if (refresh_prec + refresh_recall) > 0 else 0
            ),
            "false_trust_rate": false_trust_rate,
            "false_trust_count": false_trust,
            "unnecessary_refresh_rate": unnecessary_refresh_rate,
            "unnecessary_refresh_count": unnecessary_refresh,
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
            "per_volatility_confusion_matrix": per_volatility_cm,
            "n": len(preds),
        }

    # -------------------------------------------------------------------
    # Eval 3: Memory retrieval by age
    # -------------------------------------------------------------------

    def eval_memory_retrieval(
        self,
        test_path: str | Path,
        variant: str = "time_memory",
    ) -> dict:
        """
        Test whether the model uses memory age appropriately.

        Filters to examples that have memory entries and evaluates
        whether the model retrieves memory vs answering directly
        based on memory freshness.
        """
        examples = self._load_examples(test_path)
        # Filter to examples with memory
        mem_examples = [ex for ex in examples if ex.get("has_memory") and ex.get("memories")]
        if not mem_examples:
            return {"error": "no examples with memory entries", "n": 0}

        texts = [self._get_input_text(ex, variant) for ex in mem_examples]
        preds = self._predict_all(texts)
        oracles = [ex["correct_action"] for ex in mem_examples]

        # Overall accuracy on memory examples
        acc = accuracy_score(oracles, preds)

        # Memory retrieval specific
        retrieve_tp = sum(1 for p, o in zip(preds, oracles)
                         if p == "retrieve_memory" and o == "retrieve_memory")
        retrieve_fp = sum(1 for p, o in zip(preds, oracles)
                         if p == "retrieve_memory" and o != "retrieve_memory")
        retrieve_fn = sum(1 for p, o in zip(preds, oracles)
                         if p != "retrieve_memory" and o == "retrieve_memory")
        retrieve_prec = retrieve_tp / (retrieve_tp + retrieve_fp) if (retrieve_tp + retrieve_fp) > 0 else 0
        retrieve_recall = retrieve_tp / (retrieve_tp + retrieve_fn) if (retrieve_tp + retrieve_fn) > 0 else 0

        # Analyze by memory age
        age_buckets = defaultdict(lambda: {"correct": 0, "total": 0})
        for ex, pred in zip(mem_examples, preds):
            if ex["memories"]:
                age = ex["memories"][0].get("age_seconds", 0)
                bucket = seconds_to_bucket(age)
                age_buckets[bucket]["total"] += 1
                if pred == ex["correct_action"]:
                    age_buckets[bucket]["correct"] += 1

        age_accuracy = {
            bucket: data["correct"] / data["total"]
            for bucket, data in age_buckets.items()
            if data["total"] > 0
        }

        return {
            "accuracy": acc,
            "retrieve_precision": retrieve_prec,
            "retrieve_recall": retrieve_recall,
            "retrieve_f1": (
                2 * retrieve_prec * retrieve_recall / (retrieve_prec + retrieve_recall)
                if (retrieve_prec + retrieve_recall) > 0 else 0
            ),
            "accuracy_by_memory_age": age_accuracy,
            "n": len(mem_examples),
        }

    # -------------------------------------------------------------------
    # Eval 4: Deadline-aware action selection
    # -------------------------------------------------------------------

    def eval_deadline_awareness(
        self,
        test_path: str | Path,
        variant: str = "time_tokens",
    ) -> dict:
        """
        Test whether the model shifts strategy when deadlines are tight.

        Filters to examples with deadline_seconds set.
        """
        examples = self._load_examples(test_path)
        deadline_exs = [ex for ex in examples if ex.get("deadline_seconds") is not None]
        if not deadline_exs:
            return {"error": "no examples with deadlines", "n": 0}

        texts = [self._get_input_text(ex, variant) for ex in deadline_exs]
        preds = self._predict_all(texts)
        oracles = [ex["correct_action"] for ex in deadline_exs]

        acc = accuracy_score(oracles, preds)

        # Break down by deadline urgency
        tight = []  # deadline < 60s
        relaxed = []  # deadline >= 60s
        for ex, pred in zip(deadline_exs, preds):
            entry = {"pred": pred, "oracle": ex["correct_action"],
                     "deadline": ex["deadline_seconds"]}
            if ex["deadline_seconds"] < 60:
                tight.append(entry)
            else:
                relaxed.append(entry)

        tight_acc = (
            accuracy_score([e["oracle"] for e in tight], [e["pred"] for e in tight])
            if tight else 0
        )
        relaxed_acc = (
            accuracy_score([e["oracle"] for e in relaxed], [e["pred"] for e in relaxed])
            if relaxed else 0
        )

        # Does the model avoid refresh when deadline is tight?
        tight_refresh_rate = (
            sum(1 for e in tight if e["pred"] == "refresh") / len(tight)
            if tight else 0
        )
        relaxed_refresh_rate = (
            sum(1 for e in relaxed if e["pred"] == "refresh") / len(relaxed)
            if relaxed else 0
        )

        return {
            "overall_accuracy": acc,
            "tight_deadline_accuracy": tight_acc,
            "relaxed_deadline_accuracy": relaxed_acc,
            "tight_refresh_rate": tight_refresh_rate,
            "relaxed_refresh_rate": relaxed_refresh_rate,
            "n_tight": len(tight),
            "n_relaxed": len(relaxed),
            "n": len(deadline_exs),
        }

    # -------------------------------------------------------------------
    # Ablations
    # -------------------------------------------------------------------

    def ablation_fake_time(
        self,
        test_path: str | Path,
        variant: str = "time_tokens",
        seed: int = 42,
    ) -> dict:
        """
        Ablation A: Replace real elapsed times with shuffled/random times.
        If the model is actually using time, performance should drop.
        """
        examples = self._load_examples(test_path)
        rng = random.Random(seed)

        # Real-time predictions
        real_texts = [self._get_input_text(ex, variant) for ex in examples]
        real_preds = self._predict_all(real_texts)
        oracles = [ex["correct_action"] for ex in examples]

        real_acc = accuracy_score(oracles, real_preds)
        real_f1 = f1_score(oracles, real_preds, average="macro",
                           labels=ACTION_LABELS, zero_division=0)

        # Shuffled-time predictions: shuffle elapsed times across examples
        shuffled_times = [ex["elapsed_seconds"] for ex in examples]
        rng.shuffle(shuffled_times)

        shuffled_texts = []
        for ex, new_time in zip(examples, shuffled_times):
            text = self._get_input_text(ex, variant)
            if variant == "text_time":
                old_time = f"{seconds_to_human(ex['elapsed_seconds'])} ago"
                new_time_str = f"{seconds_to_human(new_time)} ago"
                shuffled_texts.append(text.replace(old_time, new_time_str))
            else:
                old_token = ex["elapsed_token"]
                new_token = seconds_to_token(new_time)
                shuffled_texts.append(text.replace(old_token, new_token))

        shuffled_preds = self._predict_all(shuffled_texts)
        shuffled_acc = accuracy_score(oracles, shuffled_preds)
        shuffled_f1 = f1_score(oracles, shuffled_preds, average="macro",
                               labels=ACTION_LABELS, zero_division=0)

        return {
            "real_time_accuracy": real_acc,
            "real_time_f1": real_f1,
            "shuffled_time_accuracy": shuffled_acc,
            "shuffled_time_f1": shuffled_f1,
            "accuracy_drop": real_acc - shuffled_acc,
            "f1_drop": real_f1 - shuffled_f1,
            "n": len(examples),
        }

    def ablation_swapped_time(
        self,
        test_path: str | Path,
        variant: str = "time_tokens",
    ) -> dict:
        """
        Ablation: Systematically swap fresh↔stale times.
        Fresh examples get stale times, stale get fresh.
        Stronger than random shuffle — directly inverts the temporal signal.
        """
        examples = self._load_examples(test_path)

        # Group by counterfactual group to identify fresh/stale pairs
        groups = defaultdict(list)
        for ex in examples:
            groups[ex["counterfactual_group"]].append(ex)

        # Real-time predictions
        real_texts = [self._get_input_text(ex, variant) for ex in examples]
        real_preds = self._predict_all(real_texts)
        oracles = [ex["correct_action"] for ex in examples]

        real_acc = accuracy_score(oracles, real_preds)
        real_f1 = f1_score(oracles, real_preds, average="macro",
                           labels=ACTION_LABELS, zero_division=0)

        # Swapped-time: within each group, sort by elapsed time,
        # then reverse the time assignments
        # Build a map from example id to swapped elapsed seconds
        swapped_times = {}
        for group_id, group_exs in groups.items():
            sorted_exs = sorted(group_exs, key=lambda x: x["elapsed_seconds"])
            times = [ex["elapsed_seconds"] for ex in sorted_exs]
            reversed_times = list(reversed(times))
            for ex, new_time in zip(sorted_exs, reversed_times):
                swapped_times[ex["id"]] = new_time

        # Build swapped texts
        swapped_texts = []
        for ex in examples:
            text = self._get_input_text(ex, variant)
            new_time = swapped_times.get(ex["id"], ex["elapsed_seconds"])
            if variant == "text_time":
                old_time = f"{seconds_to_human(ex['elapsed_seconds'])} ago"
                new_time_str = f"{seconds_to_human(new_time)} ago"
                swapped_texts.append(text.replace(old_time, new_time_str))
            else:
                old_token = ex["elapsed_token"]
                new_token = seconds_to_token(new_time)
                swapped_texts.append(text.replace(old_token, new_token))

        swapped_preds = self._predict_all(swapped_texts)
        swapped_acc = accuracy_score(oracles, swapped_preds)
        swapped_f1 = f1_score(oracles, swapped_preds, average="macro",
                              labels=ACTION_LABELS, zero_division=0)

        return {
            "real_time_accuracy": real_acc,
            "real_time_f1": real_f1,
            "swapped_time_accuracy": swapped_acc,
            "swapped_time_f1": swapped_f1,
            "accuracy_drop": real_acc - swapped_acc,
            "f1_drop": real_f1 - swapped_f1,
            "n": len(examples),
        }

    def ablation_volatility_split(
        self,
        test_path: str | Path,
        variant: str = "time_tokens",
    ) -> dict:
        """
        Ablation D: Compare performance on stable vs volatile facts.
        Time-aware models should show bigger gains on volatile facts.
        """
        examples = self._load_examples(test_path)

        results_by_vol = {}
        for vol in [v.value for v in Volatility]:
            vol_exs = [ex for ex in examples if ex["volatility"] == vol]
            if not vol_exs:
                continue

            texts = [self._get_input_text(ex, variant) for ex in vol_exs]
            preds = self._predict_all(texts)
            oracles = [ex["correct_action"] for ex in vol_exs]

            results_by_vol[vol] = {
                "accuracy": accuracy_score(oracles, preds),
                "f1_macro": f1_score(oracles, preds, average="macro",
                                     labels=ACTION_LABELS, zero_division=0),
                "n": len(vol_exs),
            }

        return results_by_vol

    def ablation_time_granularity(
        self,
        test_path: str | Path,
        variant: str = "time_tokens",
    ) -> dict:
        """
        Ablation B: Compare coarse vs fine time bins.
        Remap fine time tokens to coarser buckets and see impact.
        """
        examples = self._load_examples(test_path)

        # Fine (default): all 18 buckets
        fine_texts = [self._get_input_text(ex, variant) for ex in examples]
        fine_preds = self._predict_all(fine_texts)
        oracles = [ex["correct_action"] for ex in examples]

        fine_acc = accuracy_score(oracles, fine_preds)

        # Coarse: collapse to 6 bins (seconds, minutes, hours, days, weeks, months+)
        coarse_map = {
            "<dt_5s>": "<dt_1m>", "<dt_30s>": "<dt_1m>", "<dt_1m>": "<dt_1m>",
            "<dt_5m>": "<dt_15m>", "<dt_15m>": "<dt_15m>", "<dt_30m>": "<dt_15m>",
            "<dt_1h>": "<dt_3h>", "<dt_3h>": "<dt_3h>", "<dt_6h>": "<dt_3h>",
            "<dt_12h>": "<dt_1d>", "<dt_1d>": "<dt_1d>", "<dt_3d>": "<dt_1d>",
            "<dt_1w>": "<dt_1w>", "<dt_2w>": "<dt_1w>",
            "<dt_1mo>": "<dt_3mo>", "<dt_3mo>": "<dt_3mo>",
            "<dt_6mo>": "<dt_1y>", "<dt_1y>": "<dt_1y>",
        }

        coarse_texts = []
        for ex in examples:
            text = self._get_input_text(ex, variant)
            for fine_tok, coarse_tok in coarse_map.items():
                text = text.replace(fine_tok, coarse_tok)
            coarse_texts.append(text)

        coarse_preds = self._predict_all(coarse_texts)
        coarse_acc = accuracy_score(oracles, coarse_preds)

        return {
            "fine_accuracy": fine_acc,
            "coarse_accuracy": coarse_acc,
            "accuracy_diff": fine_acc - coarse_acc,
            "n": len(examples),
        }

    # -------------------------------------------------------------------
    # Eval 5: Temporal policy curves
    # -------------------------------------------------------------------

    def eval_temporal_policy_curves(
        self,
        variant: str = "text_time",
        n_per_volatility: int = 3,
        seed: int = 42,
    ) -> dict:
        """
        Evaluate how action probabilities change as elapsed time varies.

        For each volatility level, generates fixed scenarios and sweeps
        through all time buckets, recording the model's action probability
        distribution at each point.
        """
        from .policy_curves import generate_policy_scenarios, vary_time_for_scenario
        from ..data.taxonomy import TIME_BUCKET_SECONDS, ACTION_LABELS as LABELS

        scenarios = generate_policy_scenarios(
            n_per_volatility=n_per_volatility, seed=seed,
        )

        # Results: {volatility: {elapsed_seconds_str: {action: avg_prob}}}
        results = {}

        for scenario in scenarios:
            vol = scenario["volatility"]
            time_inputs = vary_time_for_scenario(scenario, variant=variant)

            texts = [ti["input_text"] for ti in time_inputs]
            # Use predict_action_probs if available, else fall back to hard predictions
            if hasattr(self.model, "predict_action_probs"):
                prev_side = self.model.tokenizer.padding_side
                self.model.tokenizer.padding_side = "left"
                enc = self.model.tokenizer(
                    texts,
                    max_length=self.model.config.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                self.model.tokenizer.padding_side = prev_side
                probs_list = self.model.predict_action_probs(
                    enc["input_ids"], enc["attention_mask"],
                )
            else:
                # Fall back to hard predictions (uniform prob on predicted class)
                preds = self._predict_all(texts)
                probs_list = []
                for pred in preds:
                    probs = {a: 0.0 for a in LABELS}
                    probs[pred] = 1.0
                    probs_list.append(probs)

            if vol not in results:
                results[vol] = {}

            for ti, probs in zip(time_inputs, probs_list):
                t_key = str(ti["elapsed_seconds"])
                if t_key not in results[vol]:
                    results[vol][t_key] = {a: [] for a in LABELS}
                for action in LABELS:
                    p = probs[action] if isinstance(probs, dict) else 0.0
                    results[vol][t_key][action].append(p)

        # Average probabilities across scenarios
        averaged = {}
        for vol, time_data in results.items():
            averaged[vol] = {}
            for t_key, action_probs in time_data.items():
                averaged[vol][t_key] = {
                    action: sum(vals) / len(vals) if vals else 0.0
                    for action, vals in action_probs.items()
                }

        return averaged

    # -------------------------------------------------------------------
    # Run all evaluations
    # -------------------------------------------------------------------

    def run_all(
        self,
        test_path: str | Path,
        variant: str = "time_tokens",
    ) -> dict:
        """Run all four evaluations and return combined results."""
        results = {}

        print("Running Eval 1: Counterfactual time sensitivity...")
        results["counterfactual_sensitivity"] = self.eval_counterfactual_sensitivity(
            test_path, variant
        )

        print("Running Eval 2: Stale-context calibration...")
        results["stale_calibration"] = self.eval_stale_calibration(test_path, variant)

        print("Running Eval 3: Memory retrieval by age...")
        results["memory_retrieval"] = self.eval_memory_retrieval(
            test_path, variant=variant
        )

        print("Running Eval 4: Deadline-aware action selection...")
        results["deadline_awareness"] = self.eval_deadline_awareness(test_path, variant)

        return results

    def run_all_ablations(
        self,
        test_path: str | Path,
        variant: str = "time_tokens",
    ) -> dict:
        """Run all ablation studies."""
        results = {}

        # Fake time and granularity ablations only make sense for time-aware variants
        if variant != "baseline":
            print("Running Ablation A: Real vs fake time...")
            results["fake_time"] = self.ablation_fake_time(test_path, variant)

            if variant in ("time_tokens", "time_memory"):
                print("Running Ablation B: Time granularity...")
                results["time_granularity"] = self.ablation_time_granularity(test_path, variant)

            if variant == "text_time":
                print("Running Ablation: Paraphrase...")
                results["paraphrase"] = self.ablation_paraphrase(test_path, variant)

                print("Running Ablation: Raw seconds...")
                results["raw_seconds"] = self.ablation_raw_seconds(test_path, variant)

                print("Running Ablation: Coarse category...")
                results["category"] = self.ablation_category(test_path, variant)
        else:
            print("Skipping time ablations for baseline variant")

        print("Running Ablation D: Volatility split...")
        results["volatility_split"] = self.ablation_volatility_split(test_path, variant)

        return results

    # -------------------------------------------------------------------
    # Text-time ablations (paraphrase, raw seconds, coarse category)
    # -------------------------------------------------------------------

    def ablation_paraphrase(
        self,
        test_path: str | Path,
        variant: str = "text_time",
    ) -> dict:
        """
        Ablation: Replace human time with alternative phrasing (different unit).
        Tests whether the model learned temporal magnitude or surface patterns.
        E.g., "3 hours ago" -> "180 minutes ago"
        """
        examples = self._load_examples(test_path)
        oracles = [ex["correct_action"] for ex in examples]

        # Original predictions
        orig_texts = [self._get_input_text(ex, variant) for ex in examples]
        orig_preds = self._predict_all(orig_texts)
        orig_acc = accuracy_score(oracles, orig_preds)

        # Paraphrased predictions
        para_texts = []
        for ex in examples:
            text = self._get_input_text(ex, variant)
            orig_time = f"{seconds_to_human(ex['elapsed_seconds'])} ago"
            para_time = seconds_to_paraphrase(ex["elapsed_seconds"])
            para_texts.append(text.replace(orig_time, para_time))
        para_preds = self._predict_all(para_texts)
        para_acc = accuracy_score(oracles, para_preds)

        # Agreement: how often do orig and paraphrase give the same answer?
        agreement = sum(1 for a, b in zip(orig_preds, para_preds) if a == b) / len(orig_preds)

        return {
            "original_accuracy": orig_acc,
            "paraphrase_accuracy": para_acc,
            "accuracy_drop": orig_acc - para_acc,
            "prediction_agreement": agreement,
            "n": len(examples),
        }

    def ablation_raw_seconds(
        self,
        test_path: str | Path,
        variant: str = "text_time",
    ) -> dict:
        """
        Ablation: Replace human time with raw seconds ("10800 seconds ago").
        Preserves magnitude, removes human-interpretable scale.
        If model handles this: it's doing math. If it breaks: relies on "hours"/"days".
        """
        examples = self._load_examples(test_path)
        oracles = [ex["correct_action"] for ex in examples]

        orig_texts = [self._get_input_text(ex, variant) for ex in examples]
        orig_preds = self._predict_all(orig_texts)
        orig_acc = accuracy_score(oracles, orig_preds)

        raw_texts = []
        for ex in examples:
            text = self._get_input_text(ex, variant)
            orig_time = f"{seconds_to_human(ex['elapsed_seconds'])} ago"
            raw_time = seconds_to_raw_text(ex["elapsed_seconds"])
            raw_texts.append(text.replace(orig_time, raw_time))
        raw_preds = self._predict_all(raw_texts)
        raw_acc = accuracy_score(oracles, raw_preds)

        return {
            "original_accuracy": orig_acc,
            "raw_seconds_accuracy": raw_acc,
            "accuracy_drop": orig_acc - raw_acc,
            "n": len(examples),
        }

    def ablation_category(
        self,
        test_path: str | Path,
        variant: str = "text_time",
    ) -> dict:
        """
        Ablation: Replace exact time with coarse category.
        "3 hours ago" -> "a moderate time ago"
        Tests whether the model needs granularity or just coarse bins.
        """
        examples = self._load_examples(test_path)
        oracles = [ex["correct_action"] for ex in examples]

        orig_texts = [self._get_input_text(ex, variant) for ex in examples]
        orig_preds = self._predict_all(orig_texts)
        orig_acc = accuracy_score(oracles, orig_preds)

        cat_texts = []
        for ex in examples:
            text = self._get_input_text(ex, variant)
            orig_time = f"{seconds_to_human(ex['elapsed_seconds'])} ago"
            cat_time = seconds_to_category(ex["elapsed_seconds"])
            cat_texts.append(text.replace(orig_time, cat_time))
        cat_preds = self._predict_all(cat_texts)
        cat_acc = accuracy_score(oracles, cat_preds)

        return {
            "original_accuracy": orig_acc,
            "category_accuracy": cat_acc,
            "accuracy_drop": orig_acc - cat_acc,
            "n": len(examples),
        }

    def ablation_time_format_comparison(
        self,
        test_path: str | Path,
        variant: str = "text_time",
    ) -> dict:
        """
        Consolidated ablation comparing all time format representations.
        Replaces the original time string with each alternative format
        and measures accuracy/agreement.
        """
        from ..data.taxonomy import (
            seconds_to_absolute_timestamp,
            seconds_to_relative_elapsed,
        )

        examples = self._load_examples(test_path)
        oracles = [ex["correct_action"] for ex in examples]

        # Original predictions
        orig_texts = [self._get_input_text(ex, variant) for ex in examples]
        orig_preds = self._predict_all(orig_texts)
        orig_acc = accuracy_score(oracles, orig_preds)

        formats = {
            "original": {"accuracy": orig_acc, "agreement": 1.0},
        }

        # Define format converters: (name, conversion_fn)
        converters = [
            ("paraphrase", seconds_to_paraphrase),
            ("raw_seconds", seconds_to_raw_text),
            ("coarse_category", seconds_to_category),
            ("absolute_timestamp", seconds_to_absolute_timestamp),
            ("relative_elapsed", seconds_to_relative_elapsed),
        ]

        for fmt_name, convert_fn in converters:
            fmt_texts = []
            for ex in examples:
                text = self._get_input_text(ex, variant)
                orig_time = f"{seconds_to_human(ex['elapsed_seconds'])} ago"
                new_time = convert_fn(ex["elapsed_seconds"])
                fmt_texts.append(text.replace(orig_time, new_time))

            fmt_preds = self._predict_all(fmt_texts)
            fmt_acc = accuracy_score(oracles, fmt_preds)
            agreement = sum(1 for a, b in zip(orig_preds, fmt_preds) if a == b) / len(orig_preds)

            formats[fmt_name] = {
                "accuracy": fmt_acc,
                "accuracy_drop": orig_acc - fmt_acc,
                "agreement": agreement,
            }

        return {
            "formats": formats,
            "n": len(examples),
        }

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _get_input_text(ex: dict, variant: str) -> str:
        """Get input text for a variant, computing text_time on the fly if needed."""
        field = f"input_{variant}"
        if field in ex:
            return ex[field]
        if variant == "text_time":
            return TemporalDataset._compute_text_time_input(ex)
        raise KeyError(f"Missing field {field} and cannot compute on the fly")

    @staticmethod
    def _load_examples(path: str | Path) -> list[dict]:
        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples
