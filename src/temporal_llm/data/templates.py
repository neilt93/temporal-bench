"""
Conversation templates for each fact type.

Each template is a conversation skeleton that can be instantiated with
specific facts, timestamps, and elapsed times to produce training examples.
"""

from dataclasses import dataclass, field
from typing import Optional
import random


@dataclass
class ConversationTurn:
    role: str           # "user" or "assistant"
    content: str
    is_observation: bool = False  # True if this turn establishes a fact


@dataclass
class ConversationTemplate:
    fact_type: str
    description: str
    # The conversation history (observation phase)
    history: list[ConversationTurn]
    # The follow-up query that requires a temporal decision
    query: str
    # Variants of the query for diversity
    query_variants: list[str] = field(default_factory=list)
    # Optional memory entries (content templates)
    memory_templates: list[str] = field(default_factory=list)

    def sample_query(self, rng=None) -> str:
        rng = rng or random
        all_queries = [self.query] + self.query_variants
        return rng.choice(all_queries)


# ---------------------------------------------------------------------------
# Templates organized by fact type
# ---------------------------------------------------------------------------

TEMPLATES: list[ConversationTemplate] = [

    # === HIGH VOLATILITY ===

    # Flight status
    ConversationTemplate(
        fact_type="flight_status",
        description="User asks about previously mentioned flight details",
        history=[
            ConversationTurn("user", "Can you check the status of flight {flight_id}?"),
            ConversationTurn("assistant",
                "Flight {flight_id} is {flight_status}. {flight_detail}.",
                is_observation=True),
        ],
        query="What time is my flight again?",
        query_variants=[
            "Remind me about my flight details?",
            "Is my flight still on time?",
            "What gate was my flight at?",
            "Any updates on flight {flight_id}?",
        ],
        memory_templates=[
            "Flight {flight_id}: {flight_status}. {flight_detail}.",
        ],
    ),

    # Stock prices
    ConversationTemplate(
        fact_type="stock_price",
        description="User asks about previously mentioned stock price",
        history=[
            ConversationTurn("user", "What's {ticker} trading at?"),
            ConversationTurn("assistant",
                "{ticker} is currently at ${price}, {change} today.",
                is_observation=True),
        ],
        query="What was {ticker} at again?",
        query_variants=[
            "How's {ticker} doing?",
            "Is {ticker} still around that price?",
            "What's the latest on {ticker}?",
        ],
        memory_templates=[
            "{ticker}: ${price} ({change})",
        ],
    ),

    # Traffic
    ConversationTemplate(
        fact_type="traffic",
        description="User asks about previously mentioned traffic conditions",
        history=[
            ConversationTurn("user", "How's the traffic on {road}?"),
            ConversationTurn("assistant",
                "{road} is {traffic_status}. Estimated travel time is {travel_time}.",
                is_observation=True),
        ],
        query="Is the traffic still bad?",
        query_variants=[
            "How long will my commute take?",
            "Has the traffic cleared up?",
            "Should I take a different route?",
        ],
        memory_templates=[
            "{road}: {traffic_status}, ~{travel_time}",
        ],
    ),

    # Live sports scores
    ConversationTemplate(
        fact_type="live_score",
        description="User asks about previously mentioned game score",
        history=[
            ConversationTurn("user", "What's the score of the {team_a} vs {team_b} game?"),
            ConversationTurn("assistant",
                "{team_a} {score_a} - {team_b} {score_b}, {game_time}.",
                is_observation=True),
        ],
        query="What's the score now?",
        query_variants=[
            "Who's winning?",
            "Has the score changed?",
            "How's the game going?",
        ],
        memory_templates=[
            "{team_a} {score_a} - {team_b} {score_b} ({game_time})",
        ],
    ),

    # Current weather
    ConversationTemplate(
        fact_type="weather_now",
        description="User asks about previously mentioned current weather",
        history=[
            ConversationTurn("user", "What's the weather like right now?"),
            ConversationTurn("assistant",
                "It's currently {temp} and {conditions}. {weather_detail}.",
                is_observation=True),
        ],
        query="Is it still {conditions} outside?",
        query_variants=[
            "What's it like outside now?",
            "Has the weather changed?",
            "Do I still need an umbrella?",
        ],
        memory_templates=[
            "Current weather: {temp}, {conditions}. {weather_detail}.",
        ],
    ),

    # === MEDIUM VOLATILITY ===

    # Meeting schedule
    ConversationTemplate(
        fact_type="meeting_schedule",
        description="User asks about previously mentioned meeting",
        history=[
            ConversationTurn("user", "When is the {meeting_name}?"),
            ConversationTurn("assistant",
                "The {meeting_name} is scheduled for {meeting_time} in {meeting_location}.",
                is_observation=True),
        ],
        query="When was that meeting again?",
        query_variants=[
            "Remind me about the {meeting_name}?",
            "Is the {meeting_name} still at the same time?",
            "Where is the meeting?",
            "Did the meeting time change?",
        ],
        memory_templates=[
            "{meeting_name}: {meeting_time}, {meeting_location}",
        ],
    ),

    # Project status
    ConversationTemplate(
        fact_type="project_status",
        description="User asks about previously mentioned project status",
        history=[
            ConversationTurn("user", "What's the status of {project_name}?"),
            ConversationTurn("assistant",
                "{project_name} is {project_status}. {project_detail}.",
                is_observation=True),
        ],
        query="How's {project_name} going?",
        query_variants=[
            "Any updates on {project_name}?",
            "Is {project_name} still on track?",
            "Where are we with {project_name}?",
        ],
        memory_templates=[
            "{project_name}: {project_status}. {project_detail}.",
        ],
    ),

    # Weather forecast
    ConversationTemplate(
        fact_type="weather_forecast",
        description="User asks about previously mentioned forecast",
        history=[
            ConversationTurn("user", "What's the weather forecast for {forecast_day}?"),
            ConversationTurn("assistant",
                "The forecast for {forecast_day} is {forecast}. {forecast_detail}.",
                is_observation=True),
        ],
        query="What was the forecast for {forecast_day}?",
        query_variants=[
            "Is it still supposed to {forecast_action} on {forecast_day}?",
            "Has the forecast changed?",
            "Should I plan for rain?",
        ],
        memory_templates=[
            "Forecast {forecast_day}: {forecast}. {forecast_detail}.",
        ],
    ),

    # Package tracking
    ConversationTemplate(
        fact_type="package_tracking",
        description="User asks about previously mentioned package",
        history=[
            ConversationTurn("user", "Where's my package?"),
            ConversationTurn("assistant",
                "Your package is {package_status}. {package_detail}.",
                is_observation=True),
        ],
        query="Has my package arrived yet?",
        query_variants=[
            "Any updates on my delivery?",
            "Where is my package now?",
            "When will my package get here?",
        ],
        memory_templates=[
            "Package: {package_status}. {package_detail}.",
        ],
    ),

    # Restaurant availability
    ConversationTemplate(
        fact_type="restaurant_availability",
        description="User asks about previously mentioned restaurant",
        history=[
            ConversationTurn("user", "Can we get a table at {restaurant}?"),
            ConversationTurn("assistant",
                "{restaurant} {restaurant_status}. {restaurant_detail}.",
                is_observation=True),
        ],
        query="Is there still a table available?",
        query_variants=[
            "How long is the wait now?",
            "Should we try {restaurant}?",
            "Can we still get in tonight?",
        ],
        memory_templates=[
            "{restaurant}: {restaurant_status}. {restaurant_detail}.",
        ],
    ),

    # === LOW VOLATILITY ===

    # Company info
    ConversationTemplate(
        fact_type="company_info",
        description="User asks about previously mentioned company details",
        history=[
            ConversationTurn("user", "Where is {company} headquartered?"),
            ConversationTurn("assistant",
                "{company} is headquartered in {company_hq}. {company_detail}.",
                is_observation=True),
        ],
        query="Where is {company} based again?",
        query_variants=[
            "Remind me where {company}'s HQ is?",
            "Is {company} still in {company_hq}?",
        ],
        memory_templates=[
            "{company}: HQ in {company_hq}. {company_detail}.",
        ],
    ),

    # Personal preferences
    ConversationTemplate(
        fact_type="personal_preference",
        description="User asks about previously mentioned preference",
        history=[
            ConversationTurn("user", "I {preference_statement}."),
            ConversationTurn("assistant",
                "Got it, I'll remember that you {preference_statement}.",
                is_observation=True),
        ],
        query="What was my preference for {preference_topic}?",
        query_variants=[
            "Do you remember what I said about {preference_topic}?",
            "What did I tell you about my {preference_topic}?",
        ],
        memory_templates=[
            "User preference: {preference_statement}",
        ],
    ),

    # Contact info
    ConversationTemplate(
        fact_type="contact_info",
        description="User asks about previously mentioned contact details",
        history=[
            ConversationTurn("user", "What's {contact_name}'s {contact_type}?"),
            ConversationTurn("assistant",
                "{contact_name}'s {contact_type} is {contact_value}.",
                is_observation=True),
        ],
        query="What was {contact_name}'s {contact_type} again?",
        query_variants=[
            "Remind me of {contact_name}'s info?",
            "Do you have {contact_name}'s {contact_type}?",
        ],
        memory_templates=[
            "{contact_name}: {contact_type} = {contact_value}",
        ],
    ),

    # === STABLE ===

    # Historical facts
    ConversationTemplate(
        fact_type="historical_fact",
        description="User asks about previously mentioned historical fact",
        history=[
            ConversationTurn("user", "When did {historical_event} happen?"),
            ConversationTurn("assistant",
                "{historical_event} occurred in {historical_year}. {historical_detail}.",
                is_observation=True),
        ],
        query="When was {historical_event} again?",
        query_variants=[
            "Remind me when {historical_event} happened?",
            "What year was {historical_event}?",
        ],
        memory_templates=[
            "{historical_event}: {historical_year}. {historical_detail}.",
        ],
    ),

    # Math/science constants
    ConversationTemplate(
        fact_type="math_constant",
        description="User asks about previously mentioned constant",
        history=[
            ConversationTurn("user", "What's the value of {constant_name}?"),
            ConversationTurn("assistant",
                "{constant_name} is approximately {constant_value}.",
                is_observation=True),
        ],
        query="What was {constant_name} again?",
        query_variants=[
            "Remind me the value of {constant_name}?",
        ],
        memory_templates=[
            "{constant_name} ≈ {constant_value}",
        ],
    ),

    # Geography
    ConversationTemplate(
        fact_type="geography",
        description="User asks about previously mentioned geographic fact",
        history=[
            ConversationTurn("user", "What's the capital of {country}?"),
            ConversationTurn("assistant",
                "The capital of {country} is {capital}.",
                is_observation=True),
        ],
        query="What was the capital of {country}?",
        query_variants=[
            "Remind me, what's {country}'s capital?",
            "Is {capital} still the capital of {country}?",
        ],
        memory_templates=[
            "Capital of {country}: {capital}",
        ],
    ),

    # === TEMPLATES TARGETING RARE ACTIONS ===

    # Contradictory info → ask_clarify
    ConversationTemplate(
        fact_type="meeting_schedule",
        description="Contradictory meeting info requiring clarification",
        history=[
            ConversationTurn("user", "When is the {meeting_name}?"),
            ConversationTurn("assistant",
                "The {meeting_name} is scheduled for {meeting_time} in {meeting_location}.",
                is_observation=True),
            ConversationTurn("user", "Actually I heard it might have been moved to a different time."),
        ],
        query="So when is the {meeting_name} actually happening?",
        query_variants=[
            "Can you confirm the {meeting_name} time?",
            "Which time is correct for the {meeting_name}?",
        ],
        memory_templates=[
            "{meeting_name}: {meeting_time}, {meeting_location} (possibly changed)",
        ],
    ),

    # Contradictory project info → ask_clarify
    ConversationTemplate(
        fact_type="project_status",
        description="Conflicting project status reports requiring clarification",
        history=[
            ConversationTurn("user", "What's the status of {project_name}?"),
            ConversationTurn("assistant",
                "{project_name} is {project_status}. {project_detail}.",
                is_observation=True),
            ConversationTurn("user", "But the team lead said something different about the timeline."),
        ],
        query="What's the real status of {project_name}?",
        query_variants=[
            "Can you check on {project_name} again?",
            "Which update about {project_name} is correct?",
        ],
        memory_templates=[
            "{project_name}: {project_status} (conflicting reports). {project_detail}.",
        ],
    ),

    # Memory retrieval — stale main context but fresh memory
    ConversationTemplate(
        fact_type="personal_preference",
        description="Old context but recent memory about preferences",
        history=[
            ConversationTurn("user", "I {preference_statement}."),
            ConversationTurn("assistant",
                "Got it, noted that you {preference_statement}.",
                is_observation=True),
        ],
        query="Do you remember my {preference_topic} preference? I think I may have updated it.",
        query_variants=[
            "What do you have on file for my {preference_topic}?",
            "Check your memory for my {preference_topic} preference.",
        ],
        memory_templates=[
            "User preference (updated): {preference_statement}",
        ],
    ),

    # Memory retrieval — contact info with memory
    ConversationTemplate(
        fact_type="contact_info",
        description="Contact info may have changed, check memory",
        history=[
            ConversationTurn("user", "What's {contact_name}'s {contact_type}?"),
            ConversationTurn("assistant",
                "{contact_name}'s {contact_type} is {contact_value}.",
                is_observation=True),
        ],
        query="I think {contact_name}'s {contact_type} changed recently. What do you have?",
        query_variants=[
            "Check if we have a newer {contact_type} for {contact_name}.",
            "Do you have an updated {contact_type} for {contact_name}?",
        ],
        memory_templates=[
            "{contact_name}: {contact_type} = {contact_value} (may be updated)",
        ],
    ),

    # Contradictory restaurant info → ask_clarify
    ConversationTemplate(
        fact_type="restaurant_availability",
        description="Contradictory restaurant availability",
        history=[
            ConversationTurn("user", "Can we get a table at {restaurant}?"),
            ConversationTurn("assistant",
                "{restaurant} {restaurant_status}. {restaurant_detail}.",
                is_observation=True),
            ConversationTurn("user", "Wait, my friend just called them and got different info."),
        ],
        query="So is {restaurant} available or not?",
        query_variants=[
            "Can you double-check {restaurant} availability?",
            "Which info about {restaurant} is right?",
        ],
        memory_templates=[
            "{restaurant}: {restaurant_status} (conflicting info). {restaurant_detail}.",
        ],
    ),
]

TEMPLATE_MAP = {}
for t in TEMPLATES:
    TEMPLATE_MAP.setdefault(t.fact_type, []).append(t)


# ---------------------------------------------------------------------------
# Slot fillers — concrete values for template variables
# ---------------------------------------------------------------------------

SLOT_FILLERS = {
    # Flight
    "flight_id": ["AA123", "UA456", "DL789", "SW1234", "BA282", "JL7", "EK215", "LH401"],
    "flight_status": [
        "scheduled to depart at 6:10 PM",
        "delayed by 45 minutes",
        "on time for a 3:30 PM departure",
        "boarding now at gate C14",
        "scheduled to depart at 8:00 AM",
    ],
    "flight_detail": [
        "Gate B12",
        "Gate C7, terminal 2",
        "New departure time is 7:45 PM",
        "Boarding begins in 20 minutes",
        "Gate changed to A3",
    ],

    # Stock
    "ticker": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"],
    "price": ["187.50", "142.30", "378.91", "248.50", "155.00", "495.22", "353.10"],
    "change": ["up 1.2%", "down 0.8%", "up 3.5%", "down 2.1%", "flat", "up 5.7%", "down 1.4%"],

    # Traffic
    "road": ["I-405", "Route 101", "Highway 1", "I-95", "the 10 freeway", "I-280"],
    "traffic_status": [
        "backed up due to an accident",
        "moving slowly",
        "clear with light traffic",
        "heavy congestion",
        "moderate traffic",
    ],
    "travel_time": ["45 minutes", "20 minutes", "35 minutes", "1 hour", "25 minutes"],

    # Sports
    "team_a": ["Lakers", "Warriors", "Manchester United", "Yankees", "Patriots"],
    "team_b": ["Celtics", "Suns", "Liverpool", "Red Sox", "Bills"],
    "score_a": ["87", "102", "2", "5", "21"],
    "score_b": ["82", "98", "1", "3", "17"],
    "game_time": [
        "third quarter",
        "halftime",
        "67th minute",
        "top of the 7th",
        "4th quarter, 5 minutes left",
    ],

    # Weather
    "temp": ["72°F", "58°F", "85°F", "45°F", "68°F", "33°F"],
    "conditions": ["sunny", "rainy", "cloudy", "partly cloudy", "snowing", "windy"],
    "weather_detail": [
        "Humidity at 45%",
        "Winds from the west at 10 mph",
        "UV index is high",
        "Visibility is poor",
        "Wind chill makes it feel like 28°F",
    ],

    # Meetings
    "meeting_name": [
        "standup", "design review", "sprint planning", "1-on-1 with Sarah",
        "all-hands", "quarterly review", "client call",
    ],
    "meeting_time": [
        "10 AM", "2 PM", "tomorrow at 3 PM", "Friday at 11 AM",
        "Monday morning at 9", "this afternoon at 4",
    ],
    "meeting_location": [
        "room 3B", "the large conference room", "Zoom", "room 201",
        "the CEO's office", "Google Meet",
    ],

    # Projects
    "project_name": [
        "the API migration", "Project Phoenix", "the redesign",
        "the database upgrade", "the mobile app", "the auth refactor",
    ],
    "project_status": [
        "80% complete", "on track", "blocked", "behind schedule",
        "in QA", "ready for review",
    ],
    "project_detail": [
        "Expected to finish by end of week",
        "Waiting on the backend team",
        "Two critical bugs remaining",
        "Just passed code review",
        "Deployed to staging",
    ],

    # Forecast
    "forecast_day": ["Thursday", "this weekend", "tomorrow", "Wednesday", "next Monday"],
    "forecast": [
        "rain in the afternoon",
        "clear and warm",
        "thunderstorms likely",
        "cold with a chance of snow",
        "mild and overcast",
    ],
    "forecast_detail": [
        "High of 75°F",
        "Bring an umbrella",
        "Temperatures dropping to the 40s",
        "Perfect day for outdoor activities",
        "Wind advisory in effect",
    ],
    "forecast_action": ["rain", "snow", "be warm", "be cold", "storm"],

    # Package
    "package_status": [
        "out for delivery",
        "at the local sorting facility",
        "in transit, currently in Memphis",
        "delivered to your front door",
        "held at customs",
    ],
    "package_detail": [
        "Expected delivery by 5 PM today",
        "Should arrive by Thursday",
        "Tracking shows it left at 6 AM",
        "Signature required on delivery",
        "No estimated delivery time available",
    ],

    # Restaurant
    "restaurant": [
        "Olive Garden", "the sushi place on 5th", "Chez Pierre",
        "the Italian place", "Noma", "the new Thai restaurant",
    ],
    "restaurant_status": [
        "has a 30-minute wait",
        "has tables available",
        "is fully booked tonight",
        "just had a cancellation for 7:30",
        "is closing early today",
    ],
    "restaurant_detail": [
        "I can put your name on the list",
        "Would you like me to make a reservation?",
        "They have outdoor seating available",
        "The kitchen closes at 10 PM",
        "They're only doing takeout right now",
    ],

    # Company
    "company": ["Apple", "Google", "Microsoft", "Amazon", "Tesla", "Netflix"],
    "company_hq": ["Cupertino, California", "Mountain View", "Redmond, Washington",
                    "Seattle", "Austin, Texas", "Los Gatos, California"],
    "company_detail": [
        "They have about 160,000 employees",
        "Founded in the 1970s",
        "They moved their HQ in 2021",
        "It's their main campus",
    ],

    # Preferences
    "preference_statement": [
        "prefer window seats on flights",
        "like my coffee black",
        "am allergic to shellfish",
        "prefer dark mode in all apps",
        "don't eat gluten",
    ],
    "preference_topic": ["seating", "coffee", "food allergies", "display settings", "diet"],

    # Contacts
    "contact_name": ["Sarah", "Dr. Smith", "the plumber", "my dentist", "Alex"],
    "contact_type": ["phone number", "email", "address"],
    "contact_value": [
        "555-0123", "sarah@example.com", "100 Main St",
        "dr.smith@clinic.com", "555-9876",
    ],

    # Historical
    "historical_event": [
        "World War II ended", "the moon landing", "the Berlin Wall fell",
        "the Declaration of Independence was signed",
    ],
    "historical_year": ["1945", "1969", "1989", "1776"],
    "historical_detail": [
        "It marked the end of a global conflict",
        "Neil Armstrong was the first to step on the surface",
        "It was a pivotal moment in the Cold War",
        "It was adopted by the Continental Congress",
    ],

    # Constants
    "constant_name": ["pi", "the speed of light", "Avogadro's number", "Euler's number"],
    "constant_value": ["3.14159", "3 × 10⁸ m/s", "6.022 × 10²³", "2.71828"],

    # Geography
    "country": ["France", "Japan", "Brazil", "Egypt", "Australia", "Canada"],
    "capital": ["Paris", "Tokyo", "Brasília", "Cairo", "Canberra", "Ottawa"],
}

STRUCTURED_SLOT_FILLERS = {
    "contact_info": [
        {
            "contact_name": "Sarah",
            "contact_type": "email",
            "contact_value": "sarah@example.com",
        },
        {
            "contact_name": "Dr. Smith",
            "contact_type": "email",
            "contact_value": "dr.smith@clinic.com",
        },
        {
            "contact_name": "the plumber",
            "contact_type": "phone number",
            "contact_value": "555-0123",
        },
        {
            "contact_name": "my dentist",
            "contact_type": "phone number",
            "contact_value": "555-9876",
        },
        {
            "contact_name": "Alex",
            "contact_type": "address",
            "contact_value": "100 Main St",
        },
    ],
}


def fill_template(template_str: str, slot_values: dict[str, str]) -> str:
    """Fill a template string with slot values, leaving unfilled slots as-is."""
    result = template_str
    for key, value in slot_values.items():
        result = result.replace("{" + key + "}", value)
    return result


# Correlated slot groups: these must be sampled together by index
# Each entry maps a "primary" slot to its correlated slots.
# All lists must have the same length.
CORRELATED_SLOTS = {
    "country": ["capital"],
    "historical_event": ["historical_year", "historical_detail"],
    "preference_statement": ["preference_topic"],
    "constant_name": ["constant_value"],
    "team_a": ["team_b", "score_a", "score_b", "game_time"],
    "ticker": ["price", "change"],
}


def sample_slots(fact_type: str, rng=None) -> dict[str, str]:
    """Sample a consistent set of slot values for a given fact type.

    Correlated slots (e.g., country/capital) are sampled together by index
    to maintain factual consistency.
    """
    import re
    rng = rng or random

    templates = TEMPLATE_MAP.get(fact_type, [])
    if not templates:
        return {}

    needed_slots = set()
    for t in templates:
        for turn in t.history:
            needed_slots.update(re.findall(r'\{(\w+)\}', turn.content))
        needed_slots.update(re.findall(r'\{(\w+)\}', t.query))
        for v in t.query_variants:
            needed_slots.update(re.findall(r'\{(\w+)\}', v))
        for m in t.memory_templates:
            needed_slots.update(re.findall(r'\{(\w+)\}', m))

    values = {}
    # Track which slots have been filled via correlation
    filled = set()

    # Fact types with structured records need field-level consistency.
    if fact_type in STRUCTURED_SLOT_FILLERS:
        record = rng.choice(STRUCTURED_SLOT_FILLERS[fact_type])
        for slot, value in record.items():
            if slot in needed_slots:
                values[slot] = value
                filled.add(slot)

    # First pass: fill correlated slot groups together
    for primary, dependents in CORRELATED_SLOTS.items():
        if primary not in needed_slots or primary not in SLOT_FILLERS:
            continue
        # Check if any of the group is needed
        group = [primary] + dependents
        if not any(s in needed_slots for s in group):
            continue
        # Sample a shared index
        primary_list = SLOT_FILLERS[primary]
        idx = rng.randrange(len(primary_list))
        values[primary] = primary_list[idx]
        filled.add(primary)
        for dep in dependents:
            if dep in SLOT_FILLERS and dep in needed_slots:
                dep_list = SLOT_FILLERS[dep]
                # Use same index if lists are aligned, otherwise wrap
                values[dep] = dep_list[idx % len(dep_list)]
                filled.add(dep)

    # Second pass: fill remaining slots independently
    for slot in sorted(needed_slots):
        if slot in filled:
            continue
        if slot in SLOT_FILLERS:
            values[slot] = rng.choice(SLOT_FILLERS[slot])
        else:
            values[slot] = f"[{slot}]"

    return values
