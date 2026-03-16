"""
External benchmark loader and formatter.

Loads timestamped data from external sources (news, Wikipedia edit history)
and formats it into the same schema as the synthetic dataset for evaluation
with the existing TemporalEvaluator.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .taxonomy import (
    Volatility, FACT_TYPE_MAP,
    seconds_to_bucket, seconds_to_token, seconds_to_human,
    oracle_action,
)


@dataclass
class ExternalExample:
    """A single external benchmark example."""
    id: str
    source: str              # e.g., "wikipedia_edit", "news_headline"
    context: str             # The observed fact/information
    query: str               # Follow-up question
    elapsed_seconds: float   # Time since observation
    fact_type: str           # Mapped to taxonomy fact type
    volatility: str          # From taxonomy
    correct_action: str      # Ground truth
    # Formatted inputs
    input_baseline: str = ""
    input_text_time: str = ""


def load_external_benchmark(path: str | Path) -> list[dict]:
    """Load external benchmark from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_external_example(
    raw: dict,
    fact_type_name: Optional[str] = None,
) -> dict:
    """
    Convert a raw external example into the standard evaluation format.

    Expected raw keys: id, source, context, query, elapsed_seconds,
    fact_type (optional), volatility (optional), correct_action (optional).
    """
    fact_type_name = fact_type_name or raw.get("fact_type", "weather_now")
    fact_type = FACT_TYPE_MAP.get(fact_type_name)

    elapsed = raw["elapsed_seconds"]
    volatility = raw.get("volatility", fact_type.volatility.value if fact_type else "medium")

    # Compute correct action if not provided
    if "correct_action" in raw:
        correct = raw["correct_action"]
    elif fact_type:
        correct = oracle_action(
            elapsed_seconds=elapsed,
            fact_type=fact_type,
            has_memory=False,
            has_refresh_tool=True,
        ).value
    else:
        correct = "refresh" if elapsed > 3600 else "answer_directly"

    # Format inputs
    context = raw["context"]
    query = raw["query"]

    input_baseline = (
        f"Given the following conversation, decide the best action.\n\n"
        f"Conversation:\nAssistant: {context}\n\n"
        f"User: {query}\n\n"
        f"Actions: answer_directly, refresh, retrieve_memory, ask_clarify, abstain\n\n"
        f"Action:"
    )

    elapsed_human = seconds_to_human(elapsed)
    input_text_time = (
        f"Given the following conversation, decide the best action.\n\n"
        f"Conversation:\nAssistant: {context}\n\n"
        f"Time since last information: {elapsed_human} ago\n"
        f"Fact type: {fact_type_name} ({volatility} volatility)\n\n"
        f"User: {query}\n\n"
        f"Actions: answer_directly, refresh, retrieve_memory, ask_clarify, abstain\n\n"
        f"Action:"
    )

    return {
        "id": raw["id"],
        "source": raw.get("source", "external"),
        "conversation_text": f"Assistant: {context}",
        "query_text": query,
        "elapsed_seconds": elapsed,
        "elapsed_bucket": seconds_to_bucket(elapsed),
        "elapsed_token": seconds_to_token(elapsed),
        "elapsed_human": elapsed_human,
        "fact_type": fact_type_name,
        "volatility": volatility,
        "has_memory": False,
        "has_refresh_tool": True,
        "deadline_seconds": None,
        "memories": [],
        "correct_action": correct,
        "counterfactual_group": f"ext_{raw['id']}",
        "input_baseline": input_baseline,
        "input_text_time": input_text_time,
        "input_time_tokens": input_text_time,  # Use text_time format
        "input_time_memory": input_text_time,
    }


def build_external_benchmark(
    raw_examples: list[dict],
    output_path: str | Path,
) -> int:
    """
    Convert raw external examples to evaluation format and save as JSONL.

    Returns: number of examples written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    formatted = []
    for raw in raw_examples:
        try:
            ex = format_external_example(raw)
            formatted.append(ex)
        except Exception as e:
            print(f"Warning: skipping {raw.get('id', '?')}: {e}")

    with open(output_path, "w") as f:
        for ex in formatted:
            f.write(json.dumps(ex) + "\n")

    return len(formatted)
