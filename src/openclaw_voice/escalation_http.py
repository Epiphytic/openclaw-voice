"""
HTTP-based escalation client for the Discord voice bot.

Replaces gateway_client.py (~344 lines of WebSocket protocol + Ed25519 auth)
with a simple local HTTP POST to the OpenClaw plugin's escalation endpoint.

Usage::

    response = await escalate(
        message="User asked about the calendar",
        guild_id=12345,
        channel_id=67890,
        user_id=11111,
        gateway_port=18789,
    )

The plugin registers ``/rpc/discord-voice.escalate`` on the OpenClaw gateway
HTTP server. This function posts to that endpoint and returns the agent's
response text.

Falls back to ``gateway_client.send_to_bel`` if ``gateway_port`` is None
(standalone mode without the plugin running).
"""

from __future__ import annotations

import logging

log = logging.getLogger("openclaw_voice.escalation_http")

_DEFAULT_TIMEOUT_S = 120.0


async def escalate(
    message: str,
    guild_id: int,
    channel_id: int,
    user_id: int,
    gateway_port: int | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> str | None:
    """Send an escalation request to the main agent via the OpenClaw plugin.

    Args:
        message:      Full escalation message (context + user request).
        guild_id:     Discord guild (server) ID.
        channel_id:   Discord channel ID.
        user_id:      Discord user ID.
        gateway_port: OpenClaw gateway HTTP port. If None, falls back to the
                      legacy WebSocket gateway client.
        timeout_s:    Request timeout in seconds.

    Returns:
        Agent response text, or None on failure.
    """
    if gateway_port is None:
        # Standalone mode — use the legacy WebSocket client
        log.debug("gateway_port not set, falling back to gateway_client")
        try:
            from openclaw_voice.gateway_client import send_to_bel  # noqa: PLC0415
            return await send_to_bel(message, timeout_s=timeout_s)
        except ImportError:
            log.error("gateway_client not available and gateway_port not configured")
            return None

    import aiohttp  # type: ignore[import]

    url = f"http://127.0.0.1:{gateway_port}/rpc/discord-voice.escalate"
    payload = {
        "message": message,
        "guildId": str(guild_id),
        "channelId": str(channel_id),
        "userId": str(user_id),
    }

    log.info(
        "Escalating to main agent via plugin HTTP",
        extra={"url": url, "guild_id": guild_id, "channel_id": channel_id},
    )

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    log.error(
                        "Escalation HTTP error %d: %s",
                        resp.status,
                        body[:200],
                    )
                    return None
                data = await resp.json()
                text = data.get("text") or ""
                log.info(
                    "Escalation response received",
                    extra={"length": len(text)},
                )
                return text or None
    except aiohttp.ClientConnectorError as exc:
        log.error("Cannot connect to OpenClaw plugin at %s: %s", url, exc)
        return None
    except TimeoutError:
        log.error("Escalation request timed out after %.0fs", timeout_s)
        return None
    except Exception as exc:
        log.exception("Escalation request failed: %s", exc)
        return None
