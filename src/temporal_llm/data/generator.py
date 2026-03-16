"""
Counterfactual dataset generator.

Core idea: generate pairs/groups of examples that share the same text
but differ in elapsed time, producing different correct actions.
This gives causal control over the temporal variable.
"""

import json
import random
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .taxonomy import (
    Action, Volatility, FACT_TYPES, FACT_TYPE_MAP,
    oracle_action, sample_elapsed_time, seconds_to_bucket,
    seconds_to_token, seconds_to_human, TIME_BUCKETS,
    seconds_to_category,
)
from .templates import (
    TEMPLATE_MAP, ConversationTemplate, fill_template, sample_slots,
)


@dataclass
class MemoryEntry:
    content: str
    age_seconds: float
    source: str        # "api", "user", "observation"
    volatility: str    # matches Volatility enum value
    age_bucket: str = ""

    def __post_init__(self):
        self.age_bucket = seconds_to_bucket(self.age_seconds)


@dataclass
class Example:
    id: str
    # Conversation text (same across counterfactuals)
    conversation_text: str
    query_text: str
    # Temporal features
    elapsed_seconds: float
    elapsed_bucket: str
    elapsed_token: str
    elapsed_human: str
    # Fact metadata
    fact_type: str
    volatility: str
    # Context
    has_memory: bool
    has_refresh_tool: bool
    deadline_seconds: Optional[float]
    memories: list[dict] = field(default_factory=list)
    # Label
    correct_action: str = ""
    # Counterfactual grouping
    counterfactual_group: str = ""
    # Formatted inputs for each model variant
    input_baseline: str = ""
    input_text_time: str = ""
    input_time_tokens: str = ""
    input_time_memory: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class DatasetGenerator:
    """Generates counterfactual temporal decision-making datasets."""

    def __init__(
        self,
        seed: int = 42,
        counterfactual_group_size: int = 6,
        include_memory: bool = True,
        include_deadline: bool = True,
    ):
        self.rng = random.Random(seed)
        self.cf_group_size = counterfactual_group_size
        self.include_memory = include_memory
        self.include_deadline = include_deadline
        self._group_counter = 0

    def _make_group_id(self, fact_type: str, template_idx: int, slot_hash: str) -> str:
        self._group_counter += 1
        return f"{fact_type}_{template_idx}_{slot_hash}_{self._group_counter}"

    def _hash_slots(self, slots: dict) -> str:
        s = json.dumps(slots, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()[:8]

    def _format_conversation(self, template: ConversationTemplate, slots: dict) -> str:
        """Format the conversation history as text."""
        lines = []
        for turn in template.history:
            role = "User" if turn.role == "user" else "Assistant"
            content = fill_template(turn.content, slots)
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _format_query(self, template: ConversationTemplate, slots: dict) -> str:
        """Sample and format the follow-up query."""
        query = template.sample_query(rng=self.rng)
        return fill_template(query, slots)

    def _scenario_signature(self, fact_type: str, conversation: str, query: str) -> str:
        """Visible prompt signature used to prevent split leakage."""
        payload = {
            "fact_type": fact_type,
            "conversation": conversation,
            "query": query,
        }
        return json.dumps(payload, sort_keys=True)

    def _generate_memories(
        self, template: ConversationTemplate, slots: dict,
        elapsed_seconds: float, fact_type_obj,
    ) -> list[MemoryEntry]:
        """Generate memory entries for this example."""
        memories = []
        if not self.include_memory or not template.memory_templates:
            return memories

        # Primary memory: the observed fact, aged by elapsed time
        # Memory age can differ from elapsed time (e.g., memory was refreshed more recently)
        primary_age_factor = self.rng.choice([0.5, 0.8, 1.0, 1.5, 2.0])
        primary_age = elapsed_seconds * primary_age_factor

        content = fill_template(template.memory_templates[0], slots)
        memories.append(MemoryEntry(
            content=content,
            age_seconds=primary_age,
            source=self.rng.choice(["api", "observation", "user"]),
            volatility=fact_type_obj.volatility.value,
        ))

        # Sometimes add a distractor memory (different volatility/topic)
        if self.rng.random() < 0.4:
            distractor_facts = [ft for ft in FACT_TYPES if ft.name != fact_type_obj.name]
            dist_fact = self.rng.choice(distractor_facts)
            dist_templates = TEMPLATE_MAP.get(dist_fact.name, [])
            if dist_templates:
                dist_slots = sample_slots(dist_fact.name, rng=self.rng)
                dist_template = self.rng.choice(dist_templates)
                if dist_template.memory_templates:
                    dist_content = fill_template(dist_template.memory_templates[0], dist_slots)
                    dist_age = sample_elapsed_time(dist_fact.volatility, rng=self.rng)
                    memories.append(MemoryEntry(
                        content=dist_content,
                        age_seconds=dist_age,
                        source=self.rng.choice(["api", "observation", "user"]),
                        volatility=dist_fact.volatility.value,
                    ))

        return memories

    def _format_input_baseline(self, conversation: str, query: str) -> str:
        """Format input for the baseline model (no temporal features)."""
        return (
            f"Given the following conversation, decide the best action.\n\n"
            f"Conversation:\n{conversation}\n\n"
            f"User: {query}\n\n"
            f"Actions: answer_directly, refresh, retrieve_memory, ask_clarify, abstain\n\n"
            f"Action:"
        )

    def _format_input_text_time(
        self, conversation: str, query: str,
        elapsed_seconds: float, fact_type: str, volatility: str,
    ) -> str:
        """Format input for the text-time model (natural language, no special tokens)."""
        elapsed_human = seconds_to_human(elapsed_seconds)
        return (
            f"Given the following conversation, decide the best action.\n\n"
            f"Conversation:\n{conversation}\n\n"
            f"Time since last information: {elapsed_human} ago\n"
            f"Fact type: {fact_type} ({volatility} volatility)\n\n"
            f"User: {query}\n\n"
            f"Actions: answer_directly, refresh, retrieve_memory, ask_clarify, abstain\n\n"
            f"Action:"
        )

    def _format_input_time_tokens(
        self, conversation: str, query: str,
        elapsed_seconds: float, fact_type: str, volatility: str,
    ) -> str:
        """Format input for the time-token model."""
        elapsed_token = seconds_to_token(elapsed_seconds)
        elapsed_human = seconds_to_human(elapsed_seconds)
        return (
            f"Given the following conversation, decide the best action.\n\n"
            f"Conversation:\n{conversation}\n\n"
            f"Time since last information: {elapsed_token} ({elapsed_human})\n"
            f"Fact type: {fact_type} ({volatility} volatility)\n\n"
            f"User: {query}\n\n"
            f"Actions: answer_directly, refresh, retrieve_memory, ask_clarify, abstain\n\n"
            f"Action:"
        )

    def _format_input_time_memory(
        self, conversation: str, query: str,
        elapsed_seconds: float, fact_type: str, volatility: str,
        memories: list[MemoryEntry],
    ) -> str:
        """Format input for the time + memory model."""
        elapsed_token = seconds_to_token(elapsed_seconds)
        elapsed_human = seconds_to_human(elapsed_seconds)

        mem_lines = []
        for m in memories:
            age_token = f"<dt_{m.age_bucket}>"
            age_human = seconds_to_human(m.age_seconds)
            mem_lines.append(
                f"  - [age: {age_token} ({age_human})] {m.content} "
                f"(source: {m.source}, volatility: {m.volatility})"
            )
        mem_block = "\n".join(mem_lines) if mem_lines else "  (no memories available)"

        return (
            f"Given the following conversation and memory, decide the best action.\n\n"
            f"Conversation:\n{conversation}\n\n"
            f"Time since last information: {elapsed_token} ({elapsed_human})\n"
            f"Fact type: {fact_type} ({volatility} volatility)\n\n"
            f"Memory entries:\n{mem_block}\n\n"
            f"User: {query}\n\n"
            f"Actions: answer_directly, refresh, retrieve_memory, ask_clarify, abstain\n\n"
            f"Action:"
        )

    def generate_counterfactual_group(
        self,
        fact_type_name: str,
        template_idx: int = 0,
        slots: Optional[dict] = None,
    ) -> list[Example]:
        """
        Generate a counterfactual group: same conversation, different elapsed times.
        Returns `cf_group_size` examples with varying temporal features.
        """
        fact_type = FACT_TYPE_MAP[fact_type_name]
        templates = TEMPLATE_MAP.get(fact_type_name, [])
        if not templates:
            return []

        template = templates[template_idx % len(templates)]
        if slots is None:
            slots = sample_slots(fact_type_name, rng=self.rng)

        slot_hash = self._hash_slots(slots)
        group_id = self._make_group_id(fact_type_name, template_idx, slot_hash)

        conversation = self._format_conversation(template, slots)
        query = self._format_query(template, slots)

        # Sample diverse elapsed times spanning fresh → very stale
        # Bias toward stale to counteract answer_directly dominance
        biases = ["fresh", "borderline", "borderline", "stale", "stale", "stale"]
        if self.cf_group_size != 6:
            biases = [self.rng.choice(["fresh", "borderline", "stale"])
                      for _ in range(self.cf_group_size)]

        # Fix: sample context variables ONCE per group so only elapsed time varies
        group_has_memory = self.include_memory and self.rng.random() < 0.7
        group_has_refresh = self.rng.random() < 0.6
        group_deadline = None
        if self.include_deadline and self.rng.random() < 0.2:
            group_deadline = self.rng.choice([10, 30, 60, 300, 3600, 86400])

        # Generate memories once for the group (content is fixed, age will scale with elapsed)
        # Use a representative elapsed time to generate memory content
        representative_elapsed = sample_elapsed_time(fact_type.volatility, bias="borderline", rng=self.rng)
        group_memories_template = self._generate_memories(template, slots, representative_elapsed, fact_type)

        examples = []
        for i, bias in enumerate(biases):
            elapsed = sample_elapsed_time(fact_type.volatility, bias=bias, rng=self.rng)

            # Keep memory ages fixed within a counterfactual group so the
            # only changing temporal variable is the top-level elapsed time.
            memories = []
            for mem_tmpl in group_memories_template:
                memories.append(MemoryEntry(
                    content=mem_tmpl.content,
                    age_seconds=mem_tmpl.age_seconds,
                    source=mem_tmpl.source,
                    volatility=mem_tmpl.volatility,
                ))
            memory_age = memories[0].age_seconds if memories else None

            correct = oracle_action(
                elapsed_seconds=elapsed,
                fact_type=fact_type,
                has_memory=group_has_memory and len(memories) > 0,
                has_refresh_tool=group_has_refresh,
                deadline_seconds=group_deadline,
                memory_age_seconds=memory_age,
            )

            ex = Example(
                id=f"{group_id}_{i}",
                conversation_text=conversation,
                query_text=query,
                elapsed_seconds=elapsed,
                elapsed_bucket=seconds_to_bucket(elapsed),
                elapsed_token=seconds_to_token(elapsed),
                elapsed_human=seconds_to_human(elapsed),
                fact_type=fact_type_name,
                volatility=fact_type.volatility.value,
                has_memory=group_has_memory and len(memories) > 0,
                has_refresh_tool=group_has_refresh,
                deadline_seconds=group_deadline,
                memories=[asdict(m) for m in memories] if group_has_memory else [],
                correct_action=correct.value,
                counterfactual_group=group_id,
            )

            # Format inputs for each model variant
            ex.input_baseline = self._format_input_baseline(conversation, query)
            ex.input_text_time = self._format_input_text_time(
                conversation, query, elapsed, fact_type_name, fact_type.volatility.value,
            )
            ex.input_time_tokens = self._format_input_time_tokens(
                conversation, query, elapsed, fact_type_name, fact_type.volatility.value,
            )
            ex.input_time_memory = self._format_input_time_memory(
                conversation, query, elapsed, fact_type_name, fact_type.volatility.value,
                [MemoryEntry(**m) for m in ex.memories] if ex.memories else [],
            )

            examples.append(ex)

        return examples

    def generate_dataset(
        self,
        examples_per_fact_type: int = 20,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> dict[str, list[Example]]:
        """
        Generate the full dataset with train/val/test splits.

        Splits are done at the counterfactual-group level to prevent
        data leakage (all examples from one group go to the same split).

        Stable fact types get fewer groups since they always produce
        answer_directly regardless of elapsed time.
        """
        all_groups: list[list[Example]] = []
        seen_signatures: set[str] = set()

        for fact_type in FACT_TYPES:
            templates = TEMPLATE_MAP.get(fact_type.name, [])
            if not templates:
                continue

            # Fewer groups for stable facts (no temporal signal to learn)
            if fact_type.volatility == Volatility.STABLE:
                groups_needed = max(5, examples_per_fact_type // 4)
            else:
                groups_needed = examples_per_fact_type

            groups_created = 0
            attempts = 0
            max_attempts = max(groups_needed * 50, 100)

            while groups_created < groups_needed:
                template_idx = attempts % len(templates)
                group = self.generate_counterfactual_group(
                    fact_type.name, template_idx=template_idx,
                )
                attempts += 1
                if group:
                    signature = self._scenario_signature(
                        fact_type.name,
                        group[0].conversation_text,
                        group[0].query_text,
                    )
                    if signature in seen_signatures:
                        if attempts >= max_attempts:
                            raise RuntimeError(
                                f"Could not generate {groups_needed} unique groups for "
                                f"{fact_type.name}; only produced {groups_created}."
                            )
                        continue
                    seen_signatures.add(signature)
                    all_groups.append(group)
                    groups_created += 1

        # Shuffle groups
        self.rng.shuffle(all_groups)

        # Split at group level
        n = len(all_groups)
        train_end = int(n * split_ratios[0])
        val_end = train_end + int(n * split_ratios[1])

        splits = {
            "train": [],
            "val": [],
            "test": [],
        }

        for i, group in enumerate(all_groups):
            if i < train_end:
                splits["train"].extend(group)
            elif i < val_end:
                splits["val"].extend(group)
            else:
                splits["test"].extend(group)

        return splits

    def save_dataset(
        self,
        output_dir: str | Path,
        examples_per_fact_type: int = 20,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> dict[str, int]:
        """Generate and save dataset to disk as JSONL files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = self.generate_dataset(examples_per_fact_type, split_ratios)
        counts = {}

        for split_name, examples in splits.items():
            path = output_dir / f"{split_name}.jsonl"
            with open(path, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex.to_dict()) + "\n")
            counts[split_name] = len(examples)

        # Save metadata
        meta = {
            "fact_types": [ft.name for ft in FACT_TYPES],
            "volatility_levels": [v.value for v in Volatility],
            "actions": [a.value for a in Action],
            "counterfactual_group_size": self.cf_group_size,
            "examples_per_fact_type": examples_per_fact_type,
            "split_ratios": list(split_ratios),
            "counts": counts,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return counts
