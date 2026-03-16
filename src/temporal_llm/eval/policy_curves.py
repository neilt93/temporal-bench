"""
Monotonic temporal policy curves.

Generates fixed scenarios and varies elapsed time to produce
action probability distributions at each time point. The expected
pattern: answer_directly probability decreases with time while
refresh probability increases, with the crossover earlier for
volatile facts.
"""

import random
from typing import Optional

from ..data.taxonomy import (
    FACT_TYPES, FACT_TYPE_MAP, Volatility,
    TIME_BUCKETS, TIME_BUCKET_SECONDS,
    seconds_to_human, seconds_to_token, seconds_to_bucket,
)
from ..data.templates import TEMPLATE_MAP, sample_slots, fill_template


def generate_policy_scenarios(
    n_per_volatility: int = 3,
    seed: int = 42,
) -> list[dict]:
    """
    Generate fixed scenarios for policy curve evaluation.

    Each scenario is a (conversation, query, fact_type, volatility) tuple
    that can be evaluated at multiple time points.
    """
    rng = random.Random(seed)
    scenarios = []

    for vol in Volatility:
        fact_types = [ft for ft in FACT_TYPES if ft.volatility == vol]
        if not fact_types:
            continue

        for i in range(n_per_volatility):
            ft = fact_types[i % len(fact_types)]
            templates = TEMPLATE_MAP.get(ft.name, [])
            if not templates:
                continue

            template = templates[i % len(templates)]
            slots = sample_slots(ft.name, rng=rng)

            # Build conversation text
            lines = []
            for turn in template.history:
                role = "User" if turn.role == "user" else "Assistant"
                content = fill_template(turn.content, slots)
                lines.append(f"{role}: {content}")
            conversation = "\n".join(lines)

            # Build query
            query = template.sample_query(rng=rng)
            query = fill_template(query, slots)

            scenarios.append({
                "id": f"{ft.name}_{vol.value}_{i}",
                "conversation": conversation,
                "query": query,
                "fact_type": ft.name,
                "volatility": vol.value,
            })

    return scenarios


def vary_time_for_scenario(
    scenario: dict,
    time_buckets: Optional[list[float]] = None,
    variant: str = "text_time",
) -> list[dict]:
    """
    Produce one input per time bucket for a given scenario.

    Returns a list of dicts with keys: elapsed_seconds, input_text.
    """
    if time_buckets is None:
        time_buckets = sorted(TIME_BUCKET_SECONDS)

    inputs = []
    for elapsed in time_buckets:
        elapsed_human = seconds_to_human(elapsed)
        elapsed_token = seconds_to_token(elapsed)

        if variant == "text_time":
            text = (
                f"Given the following conversation, decide the best action.\n\n"
                f"Conversation:\n{scenario['conversation']}\n\n"
                f"Time since last information: {elapsed_human} ago\n"
                f"Fact type: {scenario['fact_type']} ({scenario['volatility']} volatility)\n\n"
                f"User: {scenario['query']}\n\n"
                f"Actions: answer_directly, refresh, retrieve_memory, ask_clarify, abstain\n\n"
                f"Action:"
            )
        elif variant == "time_tokens":
            text = (
                f"Given the following conversation, decide the best action.\n\n"
                f"Conversation:\n{scenario['conversation']}\n\n"
                f"Time since last information: {elapsed_token} ({elapsed_human})\n"
                f"Fact type: {scenario['fact_type']} ({scenario['volatility']} volatility)\n\n"
                f"User: {scenario['query']}\n\n"
                f"Actions: answer_directly, refresh, retrieve_memory, ask_clarify, abstain\n\n"
                f"Action:"
            )
        else:
            # baseline — no time info
            text = (
                f"Given the following conversation, decide the best action.\n\n"
                f"Conversation:\n{scenario['conversation']}\n\n"
                f"User: {scenario['query']}\n\n"
                f"Actions: answer_directly, refresh, retrieve_memory, ask_clarify, abstain\n\n"
                f"Action:"
            )

        inputs.append({
            "elapsed_seconds": elapsed,
            "input_text": text,
        })

    return inputs
