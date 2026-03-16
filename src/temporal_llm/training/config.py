"""
Training and experiment configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    load_in_4bit: bool = False  # QLoRA — enable if VRAM is tight
    load_in_8bit: bool = False
    max_length: int = 512
    gradient_checkpointing: bool = False
    lora_save_embeddings: bool = False  # Force modules_to_save for embed_tokens/lm_head


@dataclass
class TrainingConfig:
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # Data
    data_dir: str = "data/generated"
    variant: str = "baseline"  # "baseline", "time_tokens", "time_memory"

    # Training hyperparameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    num_epochs: int = 5
    max_steps: int = -1  # -1 = use num_epochs

    # Optimization
    optimizer: str = "adamw_torch"
    lr_scheduler: str = "cosine"
    fp16: bool = False
    bf16: bool = True  # Use bf16 on Ampere+ (RTX 4080)

    # Logging & checkpointing
    output_dir: str = "outputs"
    logging_steps: int = 10
    eval_steps: int = 25
    save_steps: int = 25
    save_total_limit: int = 3
    eval_strategy: str = "steps"

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "temporal-llm"
    wandb_run_name: Optional[str] = None

    # Reproducibility
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        model_cfg = ModelConfig(**raw.pop("model", {}))
        config = cls(model=model_cfg, **raw)
        return config

    def to_yaml(self, path: str | Path):
        from dataclasses import asdict
        d = asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def run_name(self) -> str:
        if self.wandb_run_name:
            return self.wandb_run_name
        model_short = self.model.name.split("/")[-1]
        return f"{model_short}_{self.variant}_lr{self.learning_rate}"
