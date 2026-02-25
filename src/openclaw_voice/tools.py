"""
Chip's local tool implementations.

Small, fast tools that Chip can execute directly without escalating to Bel.
Each tool returns a string result that gets fed back to the LLM for a
natural-language response.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import httpx

log = logging.getLogger("openclaw_voice.tools")

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather and short forecast for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Ladysmith, Canada' or 'Vancouver BC'",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return top results. Use for factual questions, current events, or anything you're not sure about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current date and time. Use when asked about the time, date, or day of week.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone name, e.g. 'America/Vancouver'. Defaults to Pacific.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_bel",
            "description": (
                "Escalate a request to Bel (the main AI agent) for complex tasks. "
                "Use for: calendar, email, personal data, multi-step tasks, "
                "anything requiring memory or tools you don't have. "
                "Bel can check calendars, send messages, manage files, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "Clear description of what the user needs, with full context.",
                    },
                },
                "required": ["request"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def get_weather(location: str) -> str:
    """Fetch weather from wttr.in (no API key needed)."""
    try:
        t0 = time.monotonic()
        with httpx.Client(timeout=15.0) as client:
            # Format 3 gives a compact one-line summary
            resp = client.get(
                f"https://wttr.in/{location}",
                params={"format": "j1"},
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        current = data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "?")
        feels_like = current.get("FeelsLikeC", "?")
        desc = current.get("weatherDesc", [{}])[0].get("value", "unknown")
        humidity = current.get("humidity", "?")
        wind_kmph = current.get("windspeedKmph", "?")

        # Get today's forecast for high/low
        forecast = data.get("weather", [{}])[0]
        max_c = forecast.get("maxtempC", "?")
        min_c = forecast.get("mintempC", "?")

        elapsed = int((time.monotonic() - t0) * 1000)
        log.info("Weather fetched for %s in %dms", location, elapsed)

        return (
            f"Weather in {location}: {desc}, {temp_c}°C (feels like {feels_like}°C). "
            f"High {max_c}°C, Low {min_c}°C. Humidity {humidity}%, Wind {wind_kmph} km/h."
        )
    except Exception as exc:
        log.warning("Weather fetch failed for %s: %s", location, exc)
        return f"Could not fetch weather for {location}: {exc}"


def web_search(query: str) -> str:
    """Search via Brave Search API."""
    import os

    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        return "Web search unavailable — no BRAVE_SEARCH_API_KEY configured."

    try:
        t0 = time.monotonic()
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 3},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("web", {}).get("results", [])
        if not results:
            return f"No results found for: {query}"

        summaries = []
        for r in results[:3]:
            title = r.get("title", "")
            snippet = r.get("description", "")
            summaries.append(f"{title}: {snippet}")

        elapsed = int((time.monotonic() - t0) * 1000)
        log.info("Web search for '%s' returned %d results in %dms", query, len(results), elapsed)

        return " | ".join(summaries)

    except Exception as exc:
        log.warning("Web search failed for '%s': %s", query, exc)
        return f"Search failed: {exc}"


def get_time(timezone_name: str = "America/Vancouver") -> str:
    """Get current date/time in the given timezone."""
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(timezone_name)
        now = datetime.now(tz)
        return now.strftime("It's %A, %B %d, %Y at %I:%M %p (%Z).")
    except Exception:
        now = datetime.now()
        return now.strftime("It's %A, %B %d, %Y at %I:%M %p (local).")


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

# Map of tool name → callable
TOOL_HANDLERS: dict[str, callable] = {
    "get_weather": lambda args: get_weather(args.get("location", "")),
    "web_search": lambda args: web_search(args.get("query", "")),
    "get_time": lambda args: get_time(args.get("timezone", "America/Vancouver")),
    # escalate_to_bel is handled specially by the caller — not here
}


def execute_tool(name: str, arguments: dict) -> str | None:
    """Execute a tool by name. Returns result string or None if unknown.

    ``escalate_to_bel`` is NOT handled here — the caller must intercept it.
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return None
    try:
        return handler(arguments)
    except Exception as exc:
        log.exception("Tool %s failed: %s", name, exc)
        return f"Tool error: {exc}"
