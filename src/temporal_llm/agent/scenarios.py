"""
Agent episode scenarios for temporal transfer evaluation.

Reuses DatasetGenerator to produce agent episodes with utility scoring.
"""

import json
from pathlib import Path
from typing import Optional

from ..data.generator import DatasetGenerator
from ..data.taxonomy import FACT_TYPE_MAP


def generate_agent_episodes(
    n_episodes: int = 500,
    seed: int = 42,
) -> list[dict]:
    """
    Generate agent evaluation episodes.

    Each episode is a scenario with context, query, elapsed time,
    and oracle action — ready for utility-based scoring.
    """
    gen = DatasetGenerator(
        seed=seed,
        counterfactual_group_size=1,  # One episode per scenario
        include_memory=True,
        include_deadline=True,
    )

    episodes = []
    splits = gen.generate_dataset(
        examples_per_fact_type=max(1, n_episodes // 16),
        split_ratios=(1.0, 0.0, 0.0),
    )

    for ex in splits["train"][:n_episodes]:
        fact_type = FACT_TYPE_MAP.get(ex.fact_type)
        episodes.append({
            "id": ex.id,
            "input_baseline": ex.input_baseline,
            "input_text_time": ex.input_text_time,
            "input_time_tokens": ex.input_time_tokens,
            "elapsed_seconds": ex.elapsed_seconds,
            "correct_action": ex.correct_action,
            "fact_type": ex.fact_type,
            "volatility": ex.volatility,
            "staleness_threshold": fact_type.staleness_threshold if fact_type else 3600,
            "deadline_seconds": ex.deadline_seconds,
        })

    return episodes


def load_episodes_from_jsonl(path: str | Path) -> list[dict]:
    """Load pre-generated episodes from a JSONL file."""
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes
