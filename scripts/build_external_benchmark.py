#!/usr/bin/env python3
"""
Build external benchmark from curated examples.

Generates a set of hand-curated temporal reasoning examples from
public sources (Wikipedia-style edit history, news headlines)
formatted for evaluation with the existing TemporalEvaluator.

Usage:
    python scripts/build_external_benchmark.py
    python scripts/build_external_benchmark.py --output data/external/benchmark.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_llm.data.external_benchmark import (
    format_external_example,
    build_external_benchmark,
)


# Hand-curated external examples from public knowledge
CURATED_EXAMPLES = [
    # --- HIGH VOLATILITY ---
    {
        "id": "ext_earthquake_1",
        "source": "news",
        "context": "A 6.2 magnitude earthquake was reported near Tokyo. No tsunami warning issued.",
        "query": "Is there still earthquake activity near Tokyo?",
        "elapsed_seconds": 300,  # 5 minutes
        "fact_type": "weather_now",
        "volatility": "high",
        "correct_action": "refresh",
    },
    {
        "id": "ext_election_count",
        "source": "news",
        "context": "With 62% of votes counted, Candidate A leads with 54% of the vote.",
        "query": "Who's winning the election?",
        "elapsed_seconds": 7200,  # 2 hours
        "fact_type": "live_score",
        "volatility": "high",
        "correct_action": "refresh",
    },
    {
        "id": "ext_flight_delay",
        "source": "user_scenario",
        "context": "Flight BA282 to London is delayed. New departure time is 8:45 PM.",
        "query": "Is my flight still delayed?",
        "elapsed_seconds": 10800,  # 3 hours
        "fact_type": "flight_status",
        "volatility": "high",
        "correct_action": "refresh",
    },
    {
        "id": "ext_stock_close",
        "source": "news",
        "context": "Tesla shares jumped 8% after the earnings report, closing at $248.50.",
        "query": "How's Tesla stock doing?",
        "elapsed_seconds": 60,  # 1 minute
        "fact_type": "stock_price",
        "volatility": "high",
        "correct_action": "answer_directly",
    },
    {
        "id": "ext_weather_storm",
        "source": "news",
        "context": "A severe thunderstorm warning was issued for the metro area until 6 PM.",
        "query": "Is the storm still going?",
        "elapsed_seconds": 18000,  # 5 hours
        "fact_type": "weather_now",
        "volatility": "high",
        "correct_action": "refresh",
    },
    {
        "id": "ext_crypto_flash",
        "source": "news",
        "context": "Bitcoin flash-crashed to $52,000, down 12% in 30 minutes.",
        "query": "Has Bitcoin recovered?",
        "elapsed_seconds": 900,  # 15 minutes
        "fact_type": "stock_price",
        "volatility": "high",
        "correct_action": "refresh",
    },
    {
        "id": "ext_sports_live",
        "source": "user_scenario",
        "context": "The World Cup final is scoreless at halftime. Brazil vs Germany.",
        "query": "What's the score?",
        "elapsed_seconds": 30,  # 30 seconds
        "fact_type": "live_score",
        "volatility": "high",
        "correct_action": "answer_directly",
    },
    {
        "id": "ext_traffic_accident",
        "source": "user_scenario",
        "context": "Major accident on I-95 northbound. All lanes blocked. Expect 2-hour delays.",
        "query": "Is the highway still blocked?",
        "elapsed_seconds": 5400,  # 1.5 hours
        "fact_type": "traffic",
        "volatility": "high",
        "correct_action": "refresh",
    },

    # --- MEDIUM VOLATILITY ---
    {
        "id": "ext_ceo_change",
        "source": "wikipedia",
        "context": "The CEO of OpenAI is Sam Altman, who returned after a brief departure in Nov 2023.",
        "query": "Who leads OpenAI?",
        "elapsed_seconds": 15552000,  # 6 months
        "fact_type": "company_info",
        "volatility": "medium",
        "correct_action": "refresh",
    },
    {
        "id": "ext_python_version",
        "source": "wikipedia",
        "context": "Python 3.12.1 was released on December 8, 2023.",
        "query": "What's the latest Python version?",
        "elapsed_seconds": 7776000,  # 3 months
        "fact_type": "project_status",
        "volatility": "medium",
        "correct_action": "refresh",
    },
    {
        "id": "ext_restaurant_hours",
        "source": "user_scenario",
        "context": "The Italian place on 5th Street is open until 10 PM on weekdays.",
        "query": "Are they still open?",
        "elapsed_seconds": 604800,  # 1 week
        "fact_type": "restaurant_availability",
        "volatility": "medium",
        "correct_action": "refresh",
    },
    {
        "id": "ext_weather_forecast",
        "source": "news",
        "context": "The 5-day forecast shows rain on Thursday and clearing by the weekend.",
        "query": "Is it still going to rain Thursday?",
        "elapsed_seconds": 172800,  # 2 days
        "fact_type": "weather_forecast",
        "volatility": "medium",
        "correct_action": "refresh",
    },
    {
        "id": "ext_package_status",
        "source": "user_scenario",
        "context": "Your package shipped from the warehouse in Memphis yesterday.",
        "query": "Where's my package now?",
        "elapsed_seconds": 86400,  # 1 day
        "fact_type": "package_tracking",
        "volatility": "medium",
        "correct_action": "refresh",
    },
    {
        "id": "ext_meeting_fresh",
        "source": "user_scenario",
        "context": "The team standup is at 9:30 AM in the main conference room.",
        "query": "When is the standup?",
        "elapsed_seconds": 3600,  # 1 hour
        "fact_type": "meeting_schedule",
        "volatility": "medium",
        "correct_action": "answer_directly",
    },

    # --- LOW VOLATILITY ---
    {
        "id": "ext_wiki_population",
        "source": "wikipedia",
        "context": "The population of Tokyo is approximately 13.96 million as of the 2020 census.",
        "query": "What's Tokyo's population?",
        "elapsed_seconds": 7776000,  # 3 months
        "fact_type": "company_info",
        "volatility": "low",
        "correct_action": "answer_directly",
    },
    {
        "id": "ext_user_allergy",
        "source": "user_scenario",
        "context": "The user mentioned they have a severe nut allergy.",
        "query": "Do I have any food allergies?",
        "elapsed_seconds": 2592000,  # 1 month
        "fact_type": "personal_preference",
        "volatility": "low",
        "correct_action": "answer_directly",
    },
    {
        "id": "ext_phone_number",
        "source": "user_scenario",
        "context": "Your dentist Dr. Park is at 555-0199.",
        "query": "What's my dentist's number?",
        "elapsed_seconds": 604800,  # 1 week
        "fact_type": "contact_info",
        "volatility": "low",
        "correct_action": "answer_directly",
    },
    {
        "id": "ext_company_hq",
        "source": "wikipedia",
        "context": "Nvidia is headquartered in Santa Clara, California.",
        "query": "Where is Nvidia based?",
        "elapsed_seconds": 15552000,  # 6 months
        "fact_type": "company_info",
        "volatility": "low",
        "correct_action": "answer_directly",
    },

    # --- STABLE ---
    {
        "id": "ext_moon_landing",
        "source": "wikipedia",
        "context": "The first Moon landing was on July 20, 1969, during the Apollo 11 mission.",
        "query": "When was the Moon landing?",
        "elapsed_seconds": 31536000,  # 1 year
        "fact_type": "historical_fact",
        "volatility": "stable",
        "correct_action": "answer_directly",
    },
    {
        "id": "ext_boiling_point",
        "source": "wikipedia",
        "context": "Water boils at 100\u00b0C (212\u00b0F) at standard atmospheric pressure.",
        "query": "At what temperature does water boil?",
        "elapsed_seconds": 15552000,  # 6 months
        "fact_type": "math_constant",
        "volatility": "stable",
        "correct_action": "answer_directly",
    },
    {
        "id": "ext_capital_japan",
        "source": "wikipedia",
        "context": "The capital of Japan is Tokyo.",
        "query": "What's the capital of Japan?",
        "elapsed_seconds": 31536000,  # 1 year
        "fact_type": "geography",
        "volatility": "stable",
        "correct_action": "answer_directly",
    },

    # --- COUNTERFACTUAL PAIRS (same context, different elapsed times) ---
    {
        "id": "ext_server_status_fresh",
        "source": "user_scenario",
        "context": "The production server is healthy. CPU at 45%, memory at 62%. All services green.",
        "query": "Is the server still healthy?",
        "elapsed_seconds": 60,  # 1 minute
        "fact_type": "live_score",
        "volatility": "high",
        "correct_action": "answer_directly",
    },
    {
        "id": "ext_server_status_stale",
        "source": "user_scenario",
        "context": "The production server is healthy. CPU at 45%, memory at 62%. All services green.",
        "query": "Is the server still healthy?",
        "elapsed_seconds": 21600,  # 6 hours
        "fact_type": "live_score",
        "volatility": "high",
        "correct_action": "refresh",
    },
    {
        "id": "ext_medication_fresh",
        "source": "user_scenario",
        "context": "Your doctor prescribed metformin 500mg twice daily for blood sugar management.",
        "query": "What's my metformin dosage?",
        "elapsed_seconds": 604800,  # 1 week
        "fact_type": "personal_preference",
        "volatility": "low",
        "correct_action": "answer_directly",
    },
    {
        "id": "ext_medication_stale",
        "source": "user_scenario",
        "context": "Your doctor prescribed metformin 500mg twice daily for blood sugar management.",
        "query": "What's my metformin dosage?",
        "elapsed_seconds": 31536000,  # 1 year
        "fact_type": "personal_preference",
        "volatility": "low",
        "correct_action": "refresh",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Build external benchmark")
    parser.add_argument("--output", type=str,
                        default="data/external/news_temporal_benchmark.jsonl",
                        help="Output JSONL file")
    args = parser.parse_args()

    print(f"Building external benchmark with {len(CURATED_EXAMPLES)} examples...")
    n = build_external_benchmark(CURATED_EXAMPLES, args.output)
    print(f"Wrote {n} examples to {args.output}")

    # Print distribution
    from collections import Counter
    vols = Counter(ex["volatility"] for ex in CURATED_EXAMPLES)
    actions = Counter(ex["correct_action"] for ex in CURATED_EXAMPLES)

    print(f"\nBy volatility:")
    for v, c in sorted(vols.items()):
        print(f"  {v}: {c}")
    print(f"\nBy action:")
    for a, c in sorted(actions.items()):
        print(f"  {a}: {c}")


if __name__ == "__main__":
    main()
