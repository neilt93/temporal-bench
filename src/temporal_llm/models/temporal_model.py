"""
Model variants for temporal action classification.

All variants share the same base architecture (causal LM) but differ in:
- Baseline: no temporal features in input
- Time tokens: special <dt_*> tokens added to vocabulary and input
- Time + memory: time tokens plus memory-age metadata in input

The model is trained generatively — it outputs the action name as text.
At inference, we extract the predicted action from the generated tokens.
"""

import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from ..data.taxonomy import TIME_TOKENS, ACTION_LABELS
from ..training.config import ModelConfig


def build_tokenizer(config: ModelConfig) -> AutoTokenizer:
    """Build tokenizer with temporal special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Right-padding for training; inference methods set left-padding locally
    tokenizer.padding_side = "right"

    return tokenizer


def add_time_tokens(tokenizer) -> list[str]:
    """Add temporal special tokens to the tokenizer (extends, does not replace)."""
    existing = list(tokenizer.additional_special_tokens or [])
    new_tokens = [t for t in TIME_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing + new_tokens}
        )
    return new_tokens


class TemporalActionModel:
    """
    Wrapper that builds and manages model variants.

    Usage:
        model = TemporalActionModel(config, variant="time_tokens")
        model.build()
        predicted_actions = model.predict(input_ids, attention_mask)
    """

    def __init__(self, config: ModelConfig, variant: str = "baseline"):
        self.config = config
        self.variant = variant
        self.tokenizer = None
        self.model = None
        self._action_token_ids = {}
        self._is_peft = False

    def build(self) -> tuple:
        """Build tokenizer and model. Returns (model, tokenizer)."""
        self.tokenizer = build_tokenizer(self.config)

        if self.variant in ("time_tokens", "time_memory"):
            add_time_tokens(self.tokenizer)

        # Quantization config
        bnb_config = None
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Use bfloat16 for large models, auto for small (GPT-2)
        is_large = "gpt2" not in self.config.name.lower()
        dtype = torch.bfloat16 if is_large else "auto"
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.name, **model_kwargs
        )

        if self.variant in ("time_tokens", "time_memory"):
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.config.use_lora:
            if bnb_config:
                self.model = prepare_model_for_kbit_training(self.model)

            # When new tokens are added (time_tokens, time_memory), the embedding
            # and LM head layers must remain trainable so the model can learn
            # meaningful representations for <dt_*> tokens.
            modules_to_save = None
            if self.variant in ("time_tokens", "time_memory") or self.config.lora_save_embeddings:
                modules_to_save = ["embed_tokens", "lm_head"]

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                modules_to_save=modules_to_save,
            )
            self.model = get_peft_model(self.model, lora_config)
            self._is_peft = True
            self.model.print_trainable_parameters()

        self._cache_action_tokens()

        return self.model, self.tokenizer

    def _cache_action_tokens(self):
        """Cache the token IDs for each action label."""
        for action in ACTION_LABELS:
            tokens = self.tokenizer.encode(
                " " + action, add_special_tokens=False
            )
            self._action_token_ids[action] = tokens

    def predict_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 10,
    ) -> list[str]:
        """Generate and extract predicted actions."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated part
        generated = outputs[:, input_ids.shape[1]:]
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        # Sort actions longest-first to avoid prefix collisions
        # (e.g., "answer" matching before "ask_clarify" when text starts with "a")
        sorted_actions = sorted(ACTION_LABELS, key=len, reverse=True)
        sorted_prefixes = sorted(
            [(a, a.split("_")[0]) for a in ACTION_LABELS],
            key=lambda x: len(x[1]), reverse=True,
        )

        actions = []
        for text in decoded:
            text = text.strip().lower()
            matched = False
            for action in sorted_actions:
                if text.startswith(action):
                    actions.append(action)
                    matched = True
                    break
            if not matched:
                # Substring match fallback (longest match wins)
                best_action = None
                best_len = 0
                for action in sorted_actions:
                    if action in text and len(action) > best_len:
                        best_action = action
                        best_len = len(action)
                if best_action is None:
                    # Prefix match fallback (longest prefix first)
                    for action, prefix in sorted_prefixes:
                        if text.startswith(prefix):
                            best_action = action
                            break
                actions.append(best_action or "abstain")

        return actions

    def predict_action_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get probability distribution over actions using first-token logits.

        Returns:
            (batch, num_actions) tensor of probabilities
        """
        self.model.eval()
        device = input_ids.device
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        batch_size = input_ids.shape[0]
        last_positions = attention_mask.sum(dim=1) - 1
        last_logits = outputs.logits[
            torch.arange(batch_size, device=device), last_positions
        ]

        action_scores = torch.zeros(batch_size, len(ACTION_LABELS), device=device)
        log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)

        for i, action in enumerate(ACTION_LABELS):
            token_ids = self._action_token_ids[action]
            if token_ids:
                action_scores[:, i] = log_probs[:, token_ids[0]]

        probs = torch.nn.functional.softmax(action_scores, dim=-1)
        return probs

    def save(self, output_dir: str):
        """Save model, tokenizer, and metadata."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        # Save metadata so load() knows whether this is a PEFT model
        meta = {
            "is_peft": self._is_peft,
            "variant": self.variant,
            "base_model_name": self.config.name,
        }
        with open(Path(output_dir) / "temporal_meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, model_dir: str, config: ModelConfig, variant: str = "baseline"):
        """Load a saved model (handles both LoRA and full-finetune checkpoints)."""
        model_dir = Path(model_dir)
        instance = cls(config, variant)
        instance.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # Check if this is a PEFT/LoRA checkpoint
        meta_path = model_dir / "temporal_meta.json"
        is_peft = False
        base_model_name = config.name
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            is_peft = meta.get("is_peft", False)
            base_model_name = meta.get("base_model_name", config.name)

        if is_peft:
            # Load base model first, then apply LoRA adapter
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            # Resize embeddings if time tokens were added
            base_model.resize_token_embeddings(len(instance.tokenizer))
            instance.model = PeftModel.from_pretrained(base_model, str(model_dir))
            instance._is_peft = True
        else:
            instance.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype="auto",
                trust_remote_code=True,
            )

        instance._cache_action_tokens()
        return instance
