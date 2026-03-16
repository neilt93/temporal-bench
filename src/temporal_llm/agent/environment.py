"""
Temporal Agent Environment for transfer evaluation.

Implements a utility-based evaluation framework where models
receive rewards/penalties based on their temporal decisions.
This tests whether models trained on action classification
transfer to a utility-maximizing agent setting.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UtilityConfig:
    """Reward/penalty configuration for the temporal agent."""
    correct_answer_reward: float = 1.0
    refresh_cost: float = -0.2
    stale_answer_penalty: float = -1.5
    missed_deadline_penalty: float = -2.0
    abstain_penalty: float = -0.3
    ask_clarify_cost: float = -0.1
    retrieve_memory_cost: float = -0.05


@dataclass
class EpisodeResult:
    """Result of a single agent episode."""
    scenario_id: str
    oracle_action: str
    agent_action: str
    oracle_utility: float
    agent_utility: float
    elapsed_seconds: float
    volatility: str
    fact_type: str
    is_correct: bool = False

    def __post_init__(self):
        self.is_correct = self.agent_action == self.oracle_action


class TemporalAgentEnv:
    """
    Environment that scores temporal action decisions by utility.

    The key insight: raw accuracy doesn't capture the asymmetric costs
    of different errors. Trusting stale info (false trust) is much
    worse than unnecessarily refreshing (wasted API call).
    """

    def __init__(self, config: Optional[UtilityConfig] = None):
        self.config = config or UtilityConfig()

    def compute_utility(
        self,
        action: str,
        oracle_action: str,
        elapsed_seconds: float,
        staleness_threshold: float,
        deadline_seconds: Optional[float] = None,
    ) -> float:
        """
        Compute utility for a single action decision.

        The utility function encodes the asymmetric costs:
        - Correct answer from fresh info: +1.0
        - Refreshing when needed: +1.0 - 0.2 (refresh cost) = +0.8
        - Refreshing when unnecessary: -0.2 (wasted cost)
        - Answering with stale info: -1.5 (dangerous)
        - Abstaining: -0.3 (missed opportunity but safe)
        - Missing deadline by refreshing: -2.0
        """
        cfg = self.config
        is_stale = elapsed_seconds > staleness_threshold

        # Check deadline constraint
        if deadline_seconds is not None and deadline_seconds < 30:
            if action == "refresh":
                return cfg.missed_deadline_penalty

        if action == oracle_action:
            # Correct action
            if action == "answer_directly":
                return cfg.correct_answer_reward
            elif action == "refresh":
                return cfg.correct_answer_reward + cfg.refresh_cost
            elif action == "retrieve_memory":
                return cfg.correct_answer_reward + cfg.retrieve_memory_cost
            elif action == "ask_clarify":
                return cfg.correct_answer_reward + cfg.ask_clarify_cost
            elif action == "abstain":
                return cfg.abstain_penalty  # Correct but still a cost
        else:
            # Incorrect action
            if action == "answer_directly" and is_stale:
                return cfg.stale_answer_penalty
            elif action == "answer_directly" and not is_stale:
                # Answered directly but oracle wanted something else
                # (e.g., retrieve_memory) — mild penalty
                return 0.0
            elif action == "refresh" and oracle_action == "answer_directly":
                return cfg.refresh_cost  # Unnecessary but not dangerous
            elif action == "refresh":
                # Refreshed when should have done something else
                return cfg.refresh_cost
            elif action == "abstain":
                return cfg.abstain_penalty
            elif action == "ask_clarify":
                return cfg.ask_clarify_cost
            elif action == "retrieve_memory":
                return cfg.retrieve_memory_cost

        return 0.0

    def compute_oracle_utility(
        self,
        oracle_action: str,
        elapsed_seconds: float,
        staleness_threshold: float,
        deadline_seconds: Optional[float] = None,
    ) -> float:
        """Compute the maximum possible utility (oracle's utility)."""
        return self.compute_utility(
            action=oracle_action,
            oracle_action=oracle_action,
            elapsed_seconds=elapsed_seconds,
            staleness_threshold=staleness_threshold,
            deadline_seconds=deadline_seconds,
        )
