#!/usr/bin/env python3
"""
Update channel summaries for active Discord channels.

Fetches recent messages from channels active in the last N minutes,
summarizes them using a local LLM, and writes to summary files that
voice bots (and other agents) can load for context.

Usage:
    python update-channel-summaries.py [--config /path/to/config.toml]

Designed to run as a cron job (e.g. every 5 minutes).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("channel-summaries")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUMMARY_DIR = Path.home() / ".openclaw" / "workspace" / "memory"
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8000/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
MAX_MESSAGES = 20  # messages to fetch per channel
GUILD_ID = os.environ.get("GUILD_ID", "")


def fetch_channel_messages(channel_id: str, limit: int = MAX_MESSAGES) -> list[dict]:
    """Fetch recent messages from a Discord channel via bot API."""
    if not DISCORD_BOT_TOKEN:
        log.error("No DISCORD_BOT_TOKEN set")
        return []
    headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages?limit={limit}"
    try:
        resp = httpx.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        messages = resp.json()
        # Reverse to chronological order
        messages.reverse()
        return messages
    except Exception as exc:
        log.error("Failed to fetch messages for channel %s: %s", channel_id, exc)
        return []


def fetch_guild_channels(guild_id: str) -> list[dict]:
    """Fetch all text channels in a guild."""
    if not DISCORD_BOT_TOKEN:
        return []
    headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
    url = f"https://discord.com/api/v10/guilds/{guild_id}/channels"
    try:
        resp = httpx.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return [c for c in resp.json() if c.get("type") in (0, 5)]  # text + announcement
    except Exception as exc:
        log.error("Failed to fetch guild channels: %s", exc)
        return []


def format_messages_for_summary(messages: list[dict]) -> str:
    """Format Discord messages into a readable transcript."""
    lines = []
    for msg in messages:
        author = msg.get("author", {}).get("global_name") or msg.get("author", {}).get("username", "unknown")
        content = msg.get("content", "").strip()
        ts = msg.get("timestamp", "")
        if not content:
            # Skip empty messages (embeds only, etc.)
            continue
        # Truncate long messages
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"[{ts[:16]}] {author}: {content}")
    return "\n".join(lines)


def summarize_with_llm(channel_name: str, transcript: str) -> str:
    """Use local LLM to summarize a channel transcript."""
    prompt = (
        f"Summarize the recent conversation in the Discord channel #{channel_name}. "
        f"Focus on: what topics were discussed, any decisions made, action items, "
        f"and current status of ongoing work. Be concise — 3-5 bullet points max. "
        f"Write as if briefing someone who just joined the channel.\n/no_think\n\n"
        f"Transcript:\n{transcript}"
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.3,
    }
    try:
        resp = httpx.post(LLM_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        # Strip think tags
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text
    except Exception as exc:
        log.error("LLM summarization failed: %s", exc)
        return ""


def was_recently_active(messages: list[dict], minutes: int = 30) -> bool:
    """Check if any message was posted within the last N minutes."""
    if not messages:
        return False
    try:
        last_ts = messages[-1].get("timestamp", "")
        # Discord timestamps are ISO 8601
        last_dt = datetime.fromisoformat(last_ts.replace("+00:00", "+00:00"))
        age_seconds = (datetime.now(timezone.utc) - last_dt).total_seconds()
        return age_seconds < (minutes * 60)
    except Exception:
        return True  # if we can't parse, assume active


def update_channel_summary(channel_id: str, channel_name: str) -> bool:
    """Fetch messages, summarize, and write to file. Returns True if updated."""
    messages = fetch_channel_messages(channel_id)
    if not messages:
        return False

    if not was_recently_active(messages, minutes=30):
        log.debug("Channel #%s not recently active, skipping", channel_name)
        return False

    transcript = format_messages_for_summary(messages)
    if not transcript.strip():
        return False

    summary = summarize_with_llm(channel_name, transcript)
    if not summary:
        return False

    # Write summary file
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / f"discord-{channel_id}.md"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    content = (
        f"# Channel Summary: #{channel_name}\n"
        f"**Channel ID:** {channel_id}\n"
        f"**Last updated:** {now}\n\n"
        f"## Recent Activity\n{summary}\n"
    )

    summary_path.write_text(content)
    log.info("Updated summary for #%s (%d messages → %d chars)", channel_name, len(messages), len(summary))
    return True


def main():
    parser = argparse.ArgumentParser(description="Update Discord channel summaries")
    parser.add_argument("--guild-id", default=GUILD_ID, help="Discord guild ID")
    parser.add_argument("--channels", nargs="*", help="Specific channel IDs to update")
    parser.add_argument("--all", action="store_true", help="Update all text channels in guild")
    args = parser.parse_args()

    if args.channels:
        for ch_id in args.channels:
            update_channel_summary(ch_id, f"channel-{ch_id}")
    elif args.guild_id:
        channels = fetch_guild_channels(args.guild_id)
        updated = 0
        for ch in channels:
            ch_id = ch["id"]
            ch_name = ch.get("name", ch_id)
            if update_channel_summary(ch_id, ch_name):
                updated += 1
        log.info("Updated %d/%d channels", updated, len(channels))
    else:
        log.error("Provide --guild-id or --channels")
        sys.exit(1)


if __name__ == "__main__":
    main()
