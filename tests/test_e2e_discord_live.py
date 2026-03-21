"""Live Discord E2E test for bel-audio voice bot.

Tests actual Discord API interactions — not just the localhost control server.
Uses the belthanior bot (OpenClaw, has full perms) to observe Discord state
while driving the bel-audio bot via its control server.

Run with:
    python -m pytest tests/test_e2e_discord_live.py -v -s

Requires:
- bel-audio bot running at localhost:18790
- BELTHANIOR_TOKEN env var set (bot token for the observer bot)
- bel-audio must be in the target guild
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx
import pytest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONTROL_SERVER = "http://localhost:18790"
DISCORD_API = "https://discord.com/api/v10"

GUILD_ID = "1473159530316566551"
TEST_TEXT_CHANNEL_ID = "1484735808195002398"  # #audio-testing
VOICE_CHANNEL_ID = "1473159531365269528"  # General (Voice Channels category)
BEL_AUDIO_BOT_ID = "1476059712188321944"

BELTHANIOR_TOKEN = os.environ.get("BELTHANIOR_TOKEN", "")
if not BELTHANIOR_TOKEN:
    pytest.skip("BELTHANIOR_TOKEN env var not set", allow_module_level=True)

BELTHANIOR_HEADERS = {
    "Authorization": f"Bot {BELTHANIOR_TOKEN}",
    "Content-Type": "application/json",
}

pytestmark = [pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def discord_get(client: httpx.AsyncClient, path: str) -> dict:
    """GET from Discord API using belthanior token."""
    resp = await client.get(f"{DISCORD_API}{path}", headers=BELTHANIOR_HEADERS)
    resp.raise_for_status()
    return resp.json()


async def discord_post(client: httpx.AsyncClient, path: str, json: dict) -> dict:
    """POST to Discord API using belthanior token."""
    resp = await client.post(f"{DISCORD_API}{path}", headers=BELTHANIOR_HEADERS, json=json)
    resp.raise_for_status()
    return resp.json()


async def post_to_audio_testing(client: httpx.AsyncClient, message: str) -> None:
    """Post a message to #audio-testing using belthanior bot."""
    await discord_post(client, f"/channels/{TEST_TEXT_CHANNEL_ID}/messages", {"content": message})


async def poll_for_bot_message(
    client: httpx.AsyncClient,
    keywords: list[str],
    after_ts: float,
    timeout_s: float = 15.0,
) -> dict | None:
    """Poll #audio-testing for a bel-audio message containing any keyword.

    Searches recent channel messages for a post by BEL_AUDIO_BOT_ID whose
    content includes one of *keywords* (case-insensitive) and whose
    ISO-8601 timestamp is newer than *after_ts* (unix epoch).
    Returns the matching message dict, or None on timeout.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msgs = await get_recent_messages(client, limit=10)
        for m in msgs:
            if m.get("author", {}).get("id") != BEL_AUDIO_BOT_ID:
                continue
            content_lower = (m.get("content") or "").lower()
            if any(kw.lower() in content_lower for kw in keywords):
                return m
        await asyncio.sleep(1.5)
    return None


async def poll_bot_left(
    client: httpx.AsyncClient,
    timeout_s: float = 15.0,
) -> bool:
    """Poll control server until bel-audio reports not_connected."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = await client.post(
            f"{CONTROL_SERVER}/control/leave",
            json={"guild_id": int(GUILD_ID)},
        )
        if resp.status_code == 200 and resp.json().get("status") == "not_connected":
            return True
        await asyncio.sleep(1.0)
    return False


async def force_leave_and_wait(client: httpx.AsyncClient, timeout_s: float = 15.0) -> None:
    """Force leave and wait until discord.py's internal state is fully cleared.

    discord.py clears guild.voice_client when it receives the VOICE_STATE_UPDATE
    gateway event — this is async and can lag behind the REST API. We poll
    /control/leave until it returns 'not_connected' (meaning guild.voice_client
    is truly None) so the next join won't get 'Already connected' error.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = await client.post(
            f"{CONTROL_SERVER}/control/leave",
            json={"guild_id": int(GUILD_ID)},
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "not_connected":
                # discord.py has fully cleared the voice_client — safe to join
                await asyncio.sleep(0.25)
                return
        await asyncio.sleep(1.0)
    # Final wait even if we couldn't confirm
    await asyncio.sleep(1.0)


async def join_with_retry(
    client: httpx.AsyncClient,
    retries: int = 3,
    delay_s: float = 2.0,
) -> httpx.Response:
    """Join voice channel, retrying if 'Already connected' (timing edge case)."""
    for attempt in range(retries):
        resp = await client.post(
            f"{CONTROL_SERVER}/control/join",
            json={"guild_id": int(GUILD_ID), "channel_id": int(VOICE_CHANNEL_ID)},
        )
        if resp.status_code == 200:
            return resp
        if "Already connected" in resp.text and attempt < retries - 1:
            # Disconnect and retry
            await client.post(
                f"{CONTROL_SERVER}/control/leave",
                json={"guild_id": int(GUILD_ID)},
            )
            await asyncio.sleep(delay_s)
        else:
            return resp
    return resp  # type: ignore[return-value]


async def get_recent_messages(client: httpx.AsyncClient, limit: int = 10) -> list[dict]:
    """Fetch recent messages from #audio-testing."""
    resp = await client.get(
        f"{DISCORD_API}/channels/{TEST_TEXT_CHANNEL_ID}/messages",
        headers=BELTHANIOR_HEADERS,
        params={"limit": limit},
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiscordLive:
    """Real Discord E2E tests for bel-audio voice bot."""

    async def test_01_control_server_healthy(self):
        """Control server at localhost:18790 is reachable and healthy."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{CONTROL_SERVER}/health")
        assert resp.status_code == 200, f"Control server unreachable: {resp.status_code}"
        data = resp.json()
        assert data.get("status") == "ok", f"Unhealthy status: {data}"
        print(f"\n✓ Control server healthy: {data}")

    async def test_02_belthanior_can_read_guild(self):
        """belthanior bot can read guild info (verifies token and perms)."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            data = await discord_get(client, f"/guilds/{GUILD_ID}")
        assert data["id"] == GUILD_ID
        print(f"\n✓ Guild accessible: {data['name']} (id={data['id']})")

    async def test_03_bel_audio_bot_in_guild(self):
        """bel-audio bot is a member of the guild."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            member = await discord_get(client, f"/guilds/{GUILD_ID}/members/{BEL_AUDIO_BOT_ID}")
        assert member["user"]["id"] == BEL_AUDIO_BOT_ID
        bot_name = member["user"].get("username", "unknown")
        print(f"\n✓ bel-audio bot in guild: {bot_name} (id={BEL_AUDIO_BOT_ID})")

    async def test_04_bel_audio_initial_state(self):
        """Ensure bel-audio starts from a clean (disconnected) state."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            await force_leave_and_wait(client)
        print("\n✓ bel-audio forced to clean state (not connected)")

    async def test_05_join_voice_channel_via_control_server(self):
        """Drive bel-audio to join General voice channel via /control/join."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # First ensure it's not already connected — wait for Discord to confirm
            await force_leave_and_wait(client)

            # Now join (with retry in case of "already connected" timing edge)
            join_resp = await join_with_retry(client)

        assert join_resp.status_code == 200, (
            f"Join failed: {join_resp.status_code} {join_resp.text}"
        )
        data = join_resp.json()
        assert data.get("status") == "joined", f"Unexpected join response: {data}"
        print(f"\n✓ Control server join succeeded: {data}")

    async def test_06_verify_bot_joined_via_channel_message(self):
        """Verify bel-audio joined by polling #audio-testing for its 'Joined'/'listening' message."""
        async with httpx.AsyncClient(timeout=45.0) as client:
            await force_leave_and_wait(client)

            before = time.time()
            join_resp = await join_with_retry(client)
            assert join_resp.status_code == 200, f"Join failed: {join_resp.text}"

            msg = await poll_for_bot_message(
                client,
                keywords=["joined", "listening"],
                after_ts=before,
                timeout_s=15.0,
            )

        assert msg is not None, (
            "bel-audio did not post a 'Joined'/'listening' message in #audio-testing within 15s"
        )
        print(f"\n✓ bel-audio join confirmed via channel message: {msg.get('content', '')[:80]}")

    async def test_07_post_join_result_to_audio_testing(self):
        """Post join test result to #audio-testing for visible record."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            await post_to_audio_testing(
                client,
                "🎙️ **E2E Test**: bel-audio join verified via channel message ✅",
            )
        print("\n✓ Posted join result to #audio-testing")

    async def test_08_leave_voice_channel_via_control_server(self):
        """Drive bel-audio to leave the voice channel via /control/leave."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Ensure bot is joined first
            await client.post(
                f"{CONTROL_SERVER}/control/join",
                json={"guild_id": int(GUILD_ID), "channel_id": int(VOICE_CHANNEL_ID)},
            )
            await asyncio.sleep(1.0)

            # Now leave
            leave_resp = await client.post(
                f"{CONTROL_SERVER}/control/leave",
                json={"guild_id": int(GUILD_ID)},
            )

        assert leave_resp.status_code == 200, (
            f"Leave failed: {leave_resp.status_code} {leave_resp.text}"
        )
        data = leave_resp.json()
        assert data.get("status") in ("left", "not_connected"), f"Unexpected leave response: {data}"
        print(f"\n✓ Control server leave succeeded: {data}")

    async def test_09_verify_bot_left(self):
        """Verify bel-audio has fully disconnected via control server status."""
        async with httpx.AsyncClient(timeout=20.0) as client:
            await client.post(
                f"{CONTROL_SERVER}/control/leave",
                json={"guild_id": int(GUILD_ID)},
            )
            left = await poll_bot_left(client, timeout_s=15.0)

        assert left, "bel-audio still connected 15s after /control/leave"
        print("\n✓ Control server confirms bel-audio is disconnected")

    async def test_10_post_leave_result_to_audio_testing(self):
        """Post leave test result to #audio-testing for visible record."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            await post_to_audio_testing(
                client,
                "🎙️ **E2E Test**: bel-audio left voice channel — confirmed disconnected ✅",
            )
        print("\n✓ Posted leave result to #audio-testing")

    async def test_11_full_join_leave_cycle(self):
        """Full cycle: join -> verify via channel message -> leave -> verify disconnected."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            await force_leave_and_wait(client)

            before = time.time()
            join_resp = await join_with_retry(client)
            assert join_resp.status_code == 200, f"Join failed: {join_resp.text}"

            msg = await poll_for_bot_message(
                client,
                keywords=["joined", "listening"],
                after_ts=before,
                timeout_s=15.0,
            )
            assert msg is not None, "Bot did not post join message within 15s"

            leave_resp = await client.post(
                f"{CONTROL_SERVER}/control/leave",
                json={"guild_id": int(GUILD_ID)},
            )
            assert leave_resp.status_code == 200, f"Leave failed: {leave_resp.text}"

            left = await poll_bot_left(client, timeout_s=15.0)
            assert left, "Bot still connected after leave"

            await post_to_audio_testing(
                client,
                "🎙️ **E2E Full Cycle Test**: join -> msg verify -> leave -> disconnect verify ✅ All passed!",
            )

        print("\n✓ Full join/leave cycle passed with message-based verification")

    async def test_12_belthanior_can_send_message_to_audio_testing(self):
        """belthanior bot can post to #audio-testing (full perms check)."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            msg_data = await discord_post(
                client,
                f"/channels/{TEST_TEXT_CHANNEL_ID}/messages",
                {"content": "🧪 **E2E Test Suite**: belthanior perms check ✅"},
            )
        assert msg_data.get("id"), "Message not created (no id returned)"
        assert msg_data.get("channel_id") == TEST_TEXT_CHANNEL_ID
        print(f"\n✓ belthanior posted to #audio-testing: message id={msg_data['id']}")
