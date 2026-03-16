"""
Fact taxonomy, volatility levels, time deltas, and action space.

This module defines the core ontology for the temporal decision-making task:
- What kinds of facts exist and how quickly they go stale
- What time scales matter
- What actions the model can take
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class Action(str, Enum):
    ANSWER_DIRECTLY = "answer_directly"
    REFRESH = "refresh"
    RETRIEVE_MEMORY = "retrieve_memory"
    ASK_CLARIFY = "ask_clarify"
    ABSTAIN = "abstain"

ALL_ACTIONS = list(Action)
ACTION_LABELS = [a.value for a in Action]
NUM_ACTIONS = len(ALL_ACTIONS)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

class Volatility(str, Enum):
    HIGH = "high"        # changes within minutes-hours  (flight status, stock prices, traffic)
    MEDIUM = "medium"    # changes within hours-days      (meeting times, project status, weather forecast)
    LOW = "low"          # changes within months-years    (company HQ, country capitals, birthdays)
    STABLE = "stable"    # effectively never changes      (historical dates, physical constants, math)


# ---------------------------------------------------------------------------
# Time deltas (seconds)
# ---------------------------------------------------------------------------

# Canonical time buckets — used both for time tokens and for sampling
TIME_BUCKETS = {
    "5s":   5,
    "30s":  30,
    "1m":   60,
    "5m":   300,
    "15m":  900,
    "30m":  1800,
    "1h":   3600,
    "3h":   10800,
    "6h":   21600,
    "12h":  43200,
    "1d":   86400,
    "3d":   259200,
    "1w":   604800,
    "2w":   1209600,
    "1mo":  2592000,
    "3mo":  7776000,
    "6mo":  15552000,
    "1y":   31536000,
}

TIME_BUCKET_NAMES = list(TIME_BUCKETS.keys())
TIME_BUCKET_SECONDS = list(TIME_BUCKETS.values())

# Special tokens for time deltas
TIME_TOKENS = [f"<dt_{name}>" for name in TIME_BUCKET_NAMES]


def seconds_to_bucket(seconds: float) -> str:
    """Map a raw seconds value to the nearest time bucket (log-scale distance)."""
    import math
    log_s = math.log(max(seconds, 1e-6))
    best = TIME_BUCKET_NAMES[0]
    best_dist = abs(log_s - math.log(TIME_BUCKETS[best]))
    for name, secs in TIME_BUCKETS.items():
        dist = abs(log_s - math.log(secs))
        if dist < best_dist:
            best = name
            best_dist = dist
    return best


def seconds_to_token(seconds: float) -> str:
    """Map a raw seconds value to the corresponding time token."""
    return f"<dt_{seconds_to_bucket(seconds)}>"


def seconds_to_human(seconds: float) -> str:
    """Convert seconds to a human-readable string."""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        m = seconds / 60
        return f"{int(m)} minute{'s' if m >= 2 else ''}"
    elif seconds < 86400:
        h = seconds / 3600
        return f"{int(h)} hour{'s' if h >= 2 else ''}"
    elif seconds < 604800:
        d = seconds / 86400
        return f"{int(d)} day{'s' if d >= 2 else ''}"
    elif seconds < 2592000:
        w = seconds / 604800
        return f"{int(w)} week{'s' if w >= 2 else ''}"
    elif seconds < 31536000:
        mo = seconds / 2592000
        return f"{int(mo)} month{'s' if mo >= 2 else ''}"
    else:
        y = seconds / 31536000
        return f"{int(y)} year{'s' if y >= 2 else ''}"


def seconds_to_raw_text(seconds: float) -> str:
    """Convert to raw seconds string, e.g. '10800 seconds ago'."""
    return f"{int(seconds)} seconds ago"


_WORD_NUMBERS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty", 45: "forty-five",
}


def seconds_to_paraphrase(seconds: float) -> str:
    """Alternative phrasing: different unit or words instead of digits."""
    # Convert to a different unit than seconds_to_human would pick
    if seconds < 60:
        n = int(seconds)
        word = _WORD_NUMBERS.get(n, str(n))
        return f"{word} seconds ago"
    elif seconds < 3600:
        # Use seconds instead of minutes
        return f"{int(seconds)} seconds ago"
    elif seconds < 86400:
        # Use minutes instead of hours
        return f"{int(seconds / 60)} minutes ago"
    elif seconds < 604800:
        # Use hours instead of days
        return f"{int(seconds / 3600)} hours ago"
    elif seconds < 2592000:
        # Use days instead of weeks
        return f"{int(seconds / 86400)} days ago"
    else:
        # Use days for months/years
        return f"{int(seconds / 86400)} days ago"


def seconds_to_category(seconds: float) -> str:
    """Coarse temporal category with no magnitude."""
    if seconds < 300:           # < 5 min
        return "a very short time ago"
    elif seconds < 3600:        # < 1 hour
        return "a short time ago"
    elif seconds < 86400:       # < 1 day
        return "a moderate time ago"
    elif seconds < 604800:      # < 1 week
        return "a long time ago"
    else:
        return "a very long time ago"


def seconds_to_absolute_timestamp(seconds: float) -> str:
    """Convert elapsed seconds to an absolute timestamp string.
    Assumes 'now' is a fixed reference point for reproducibility."""
    from datetime import datetime, timedelta
    # Use a fixed reference time for reproducibility
    reference = datetime(2026, 3, 15, 12, 0, 0)
    observed_at = reference - timedelta(seconds=seconds)
    return f"observed at {observed_at.strftime('%Y-%m-%d %H:%M')}"


def seconds_to_relative_elapsed(seconds: float) -> str:
    """Convert to relative elapsed with 'about' hedging.
    E.g., 'about 3 hours have passed since then'."""
    human = seconds_to_human(seconds)
    return f"about {human} have passed since then"


# ---------------------------------------------------------------------------
# Fact categories
# ---------------------------------------------------------------------------

@dataclass
class FactType:
    name: str
    volatility: Volatility
    description: str
    examples: list[str] = field(default_factory=list)
    # Typical staleness threshold in seconds — after this, info should be refreshed
    staleness_threshold: float = 3600.0


FACT_TYPES = [
    # HIGH volatility
    FactType("flight_status",   Volatility.HIGH,   "Airline flight status, gates, delays",
             ["Flight AA123 departs at 6:10 PM from gate B12",
              "Flight UA456 is delayed by 45 minutes",
              "Flight DL789 has been cancelled"],
             staleness_threshold=1800),  # 30 min

    FactType("stock_price",     Volatility.HIGH,   "Stock or crypto prices",
             ["AAPL is trading at $187.50",
              "Bitcoin is at $43,200",
              "The S&P 500 is up 1.2% today"],
             staleness_threshold=300),   # 5 min

    FactType("traffic",         Volatility.HIGH,   "Road traffic and commute times",
             ["The I-405 is backed up, expect 45 minutes",
              "Route 101 is clear, about 20 minutes",
              "There's an accident on Highway 1 causing delays"],
             staleness_threshold=900),   # 15 min

    FactType("live_score",      Volatility.HIGH,   "Live sports scores",
             ["Lakers lead 87-82 in the third quarter",
              "Manchester United 2 - Liverpool 1, 67th minute",
              "The match is currently tied 3-3"],
             staleness_threshold=300),   # 5 min

    FactType("weather_now",     Volatility.HIGH,   "Current weather conditions",
             ["It's currently 72°F and sunny",
              "Light rain right now, 58°F",
              "Winds at 25 mph with gusts to 40"],
             staleness_threshold=3600),  # 1 hour

    # MEDIUM volatility
    FactType("meeting_schedule", Volatility.MEDIUM, "Meeting times and calendar events",
             ["Your standup is at 10 AM in room 3B",
              "The design review was moved to 2 PM",
              "Tomorrow's all-hands is cancelled"],
             staleness_threshold=21600),  # 6 hours

    FactType("project_status",  Volatility.MEDIUM, "Project progress and task status",
             ["The API migration is 80% complete",
              "We're blocked on the database upgrade",
              "QA signed off on the release candidate"],
             staleness_threshold=86400),  # 1 day

    FactType("weather_forecast", Volatility.MEDIUM, "Weather forecasts (not current conditions)",
             ["Rain expected Thursday afternoon",
              "The weekend should be clear and warm",
              "A cold front is moving in Wednesday"],
             staleness_threshold=43200),  # 12 hours

    FactType("package_tracking", Volatility.MEDIUM, "Shipping and delivery status",
             ["Your package is out for delivery",
              "Shipment arrived at the local facility",
              "Expected delivery is Thursday by 5 PM"],
             staleness_threshold=21600),  # 6 hours

    FactType("restaurant_availability", Volatility.MEDIUM, "Restaurant reservations and wait times",
             ["There's a 30-minute wait at the restaurant",
              "A table for 4 is available at 7:30 PM",
              "The restaurant is fully booked tonight"],
             staleness_threshold=7200),  # 2 hours

    # LOW volatility
    FactType("company_info",    Volatility.LOW,    "Company details like HQ, CEO, employee count",
             ["Apple is headquartered in Cupertino",
              "Microsoft's CEO is Satya Nadella",
              "The company has about 3,000 employees"],
             staleness_threshold=2592000),  # 30 days

    FactType("personal_preference", Volatility.LOW, "User preferences and habits",
             ["You mentioned you prefer window seats",
              "You usually take your coffee black",
              "You said you're allergic to shellfish"],
             staleness_threshold=7776000),  # 90 days

    FactType("contact_info",    Volatility.LOW,    "Phone numbers, emails, addresses",
             ["Your dentist's number is 555-0123",
              "Sarah's email is sarah@example.com",
              "The office address is 100 Main St"],
             staleness_threshold=2592000),  # 30 days

    # STABLE
    FactType("historical_fact", Volatility.STABLE, "Historical dates and events",
             ["World War II ended in 1945",
              "The Declaration of Independence was signed in 1776",
              "The Berlin Wall fell in 1989"],
             staleness_threshold=float("inf")),

    FactType("math_constant",   Volatility.STABLE, "Mathematical and physical constants",
             ["Pi is approximately 3.14159",
              "The speed of light is about 3 × 10⁸ m/s",
              "Avogadro's number is 6.022 × 10²³"],
             staleness_threshold=float("inf")),

    FactType("geography",       Volatility.STABLE, "Stable geographic facts",
             ["The capital of France is Paris",
              "Mount Everest is the tallest mountain",
              "The Amazon is the largest river by volume"],
             staleness_threshold=float("inf")),
]

FACT_TYPE_MAP = {ft.name: ft for ft in FACT_TYPES}


def get_fact_types_by_volatility(volatility: Volatility) -> list[FactType]:
    return [ft for ft in FACT_TYPES if ft.volatility == volatility]


# ---------------------------------------------------------------------------
# Correct-action oracle
# ---------------------------------------------------------------------------

def oracle_action(
    elapsed_seconds: float,
    fact_type: FactType,
    has_memory: bool = False,
    has_refresh_tool: bool = True,
    deadline_seconds: Optional[float] = None,
    memory_age_seconds: Optional[float] = None,
) -> Action:
    """
    Deterministic oracle that returns the correct action given temporal context.

    This encodes the "ground truth" policy that the model should learn.
    The logic is intentionally simple and interpretable.
    """
    threshold = fact_type.staleness_threshold
    vol = fact_type.volatility

    # Stable facts: always answer directly
    if vol == Volatility.STABLE:
        return Action.ANSWER_DIRECTLY

    # If there's a deadline and we're very close, prefer fast actions
    if deadline_seconds is not None and deadline_seconds < 30:
        # No time to refresh — answer with what we have or abstain
        if elapsed_seconds < threshold * 2:
            return Action.ANSWER_DIRECTLY
        return Action.ABSTAIN

    # Very stale + no tools: abstain even if memory exists but is also stale
    if elapsed_seconds > threshold * 5 and not has_refresh_tool:
        if has_memory and memory_age_seconds is not None and memory_age_seconds < threshold * 0.8:
            return Action.RETRIEVE_MEMORY
        return Action.ABSTAIN

    # Fresh information: answer directly
    if elapsed_seconds < threshold * 0.3:
        return Action.ANSWER_DIRECTLY

    # Somewhat stale: check memory first if available
    if elapsed_seconds < threshold:
        if has_memory and memory_age_seconds is not None:
            if memory_age_seconds < threshold * 0.8:
                return Action.RETRIEVE_MEMORY
        # Borderline stale + no refresh tool + no memory: ask clarify
        if not has_refresh_tool and elapsed_seconds >= threshold * 0.5:
            if not has_memory or memory_age_seconds is None:
                return Action.ASK_CLARIFY
        # Still within threshold, cautiously answer
        return Action.ANSWER_DIRECTLY

    # Beyond staleness threshold
    if elapsed_seconds < threshold * 3:
        # Moderately stale — refresh if possible
        if has_refresh_tool:
            return Action.REFRESH
        if has_memory and memory_age_seconds is not None and memory_age_seconds < threshold:
            return Action.RETRIEVE_MEMORY
        return Action.ASK_CLARIFY

    # Very stale — definitely need to refresh or abstain
    if has_refresh_tool:
        return Action.REFRESH
    if has_memory and memory_age_seconds is not None and memory_age_seconds < threshold:
        return Action.RETRIEVE_MEMORY
    return Action.ABSTAIN


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_elapsed_time(
    volatility: Volatility,
    bias: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> float:
    """
    Sample an elapsed time in seconds, optionally biased toward
    'fresh', 'stale', or 'borderline' relative to the volatility.
    """
    rng = rng or random
    if volatility == Volatility.HIGH:
        ranges = {"fresh": (5, 300), "borderline": (300, 3600), "stale": (3600, 604800)}
    elif volatility == Volatility.MEDIUM:
        ranges = {"fresh": (60, 7200), "borderline": (7200, 86400), "stale": (86400, 2592000)}
    elif volatility == Volatility.LOW:
        ranges = {"fresh": (3600, 604800), "borderline": (604800, 2592000), "stale": (2592000, 31536000)}
    else:  # STABLE
        ranges = {"fresh": (5, 31536000), "borderline": (5, 31536000), "stale": (5, 31536000)}

    if bias and bias in ranges:
        lo, hi = ranges[bias]
    else:
        all_lo = min(r[0] for r in ranges.values())
        all_hi = max(r[1] for r in ranges.values())
        lo, hi = all_lo, all_hi

    import math
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return math.exp(rng.uniform(log_lo, log_hi))
