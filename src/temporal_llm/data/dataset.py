"""
PyTorch Dataset for temporal action classification.

Loads JSONL examples and tokenizes them for each model variant.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .taxonomy import seconds_to_human


class TemporalDataset(Dataset):
    """
    Dataset for temporal action classification.

    Each example has three input formats (baseline, time_tokens, time_memory)
    and a correct action label. The `variant` parameter selects which input
    format to use during training.
    """

    ACTION_TO_IDX = {
        "answer_directly": 0,
        "refresh": 1,
        "retrieve_memory": 2,
        "ask_clarify": 3,
        "abstain": 4,
    }
    IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        variant: str = "baseline",
        max_length: int = 512,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.variant = variant
        self.max_length = max_length

        self.examples = []
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        self.input_field = f"input_{variant}"

    @staticmethod
    def _compute_text_time_input(ex: dict) -> str:
        """Compute text_time input on the fly for datasets missing the field."""
        elapsed_human = seconds_to_human(ex["elapsed_seconds"])
        return (
            f"Given the following conversation, decide the best action.\n\n"
            f"Conversation:\n{ex['conversation_text']}\n\n"
            f"Time since last information: {elapsed_human} ago\n"
            f"Fact type: {ex['fact_type']} ({ex['volatility']} volatility)\n\n"
            f"User: {ex['query_text']}\n\n"
            f"Actions: answer_directly, refresh, retrieve_memory, ask_clarify, abstain\n\n"
            f"Action:"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]

        # On-the-fly computation for text_time if field is missing
        if self.input_field not in ex and self.variant == "text_time":
            input_text = self._compute_text_time_input(ex)
        else:
            input_text = ex[self.input_field]
        action = ex["correct_action"]

        # Tokenize prompt and action SEPARATELY then concatenate.
        # This avoids BPE boundary issues from concatenating strings.
        prompt_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        action_ids = self.tokenizer.encode(" " + action, add_special_tokens=False)

        # Concatenate
        full_ids = prompt_ids + action_ids

        # Truncate if needed (truncate prompt, keep action intact)
        max_total = self.max_length
        if len(full_ids) > max_total:
            prompt_ids = prompt_ids[: max_total - len(action_ids)]
            full_ids = prompt_ids + action_ids

        prompt_len = len(prompt_ids)
        seq_len = len(full_ids)

        # Build labels: -100 for prompt tokens, real IDs for action tokens
        labels = [-100] * prompt_len + action_ids

        # Right-pad to max_length
        pad_id = self.tokenizer.pad_token_id
        pad_len = max_total - seq_len
        input_ids = full_ids + [pad_id] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Custom collate — only returns model-compatible keys."""
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }


class TemporalEvalDataset(Dataset):
    """
    Evaluation dataset that returns all three variants for each example.
    Used for comparing model variants on the same examples.
    """

    ACTION_TO_IDX = TemporalDataset.ACTION_TO_IDX
    IDX_TO_ACTION = TemporalDataset.IDX_TO_ACTION

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_length: int = 512,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.examples = []
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]
