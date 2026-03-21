"""End-to-end tests for the bel-audio Discord voice bot.

These tests validate the live bel-audio bot by hitting the control server
at localhost:18790 and the Discord gateway API. They require the bot to be
running and connected.

All tests are marked ``@pytest.mark.integration`` and can be skipped with::

    pytest -m "not integration"

Run just these tests::

    python -m pytest tests/test_e2e_discord.py -v
"""

from __future__ import annotations

import base64
import math
import struct

import httpx
import pytest

CONTROL_SERVER = "http://localhost:18790"
DISCORD_API = "https://discord.com/api/v10"

BOT_USER_ID = "1476059712188321944"
GUILD_ID = "1473159530316566551"
TEST_CHANNEL_ID = "1484735808195002398"

TOKEN_FILE = "/home/models/discord-voice-token.txt"


def _read_bot_token() -> str:
    """Read the bel-audio bot token from disk."""
    with open(TOKEN_FILE) as f:
        return f.read().strip()


def _make_pcm(duration_ms: int = 500, freq_hz: int = 440) -> bytes:
    """Generate a sine wave PCM buffer (16kHz mono int16)."""
    sample_rate = 16_000
    n_samples = sample_rate * duration_ms // 1000
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        val = int(16000 * math.sin(2 * math.pi * freq_hz * t))
        samples.append(max(-32768, min(32767, val)))
    return struct.pack(f"<{n_samples}h", *samples)


def _make_silent_pcm(duration_ms: int = 500) -> bytes:
    """Generate silent PCM bytes (16kHz mono int16)."""
    n_samples = 16_000 * duration_ms // 1000
    return b"\x00" * (n_samples * 2)


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# 1. Bot health check - control server and backend services
# ---------------------------------------------------------------------------


class TestBotHealth:
    """Verify the control server is up and backend services are reachable."""

    async def test_control_server_reachable(self):
        """GET /health returns 200 and reports bot status."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{CONTROL_SERVER}/health")
        assert resp.status_code == 200, f"Control server unreachable: {resp.status_code}"
        data = resp.json()
        assert data["status"] == "ok"

    async def test_service_health_endpoint(self):
        """GET /test/health returns status for LLM, STT, and TTS backends."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{CONTROL_SERVER}/test/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "services" in data

        for svc_name in ("llm", "stt", "tts"):
            svc = data["services"].get(svc_name)
            assert svc is not None, f"Missing service: {svc_name}"
            assert "status" in svc
            assert "url" in svc

    async def test_all_services_healthy(self):
        """All backend services (LLM, STT, TTS) should report ok."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{CONTROL_SERVER}/test/health")
        data = resp.json()
        unhealthy = {
            name: info for name, info in data["services"].items() if info["status"] != "ok"
        }
        assert not unhealthy, f"Unhealthy services: {unhealthy}"
        assert data["all_healthy"] is True


# ---------------------------------------------------------------------------
# 2. Pipeline test - LLM + TTS via control server
# ---------------------------------------------------------------------------


class TestPipeline:
    """Test the LLM + TTS pipeline through the control server endpoint."""

    async def test_pipeline_text_response(self):
        """POST /test/pipeline with text returns an LLM response or escalation."""
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{CONTROL_SERVER}/test/pipeline",
                json={"text": "Say the word hello and nothing else."},
            )
        assert resp.status_code == 200, f"Pipeline failed: {resp.text}"
        data = resp.json()
        assert data["status"] == "ok"
        has_response = len(data["llm_response"]) > 0
        has_escalation = data.get("escalation") is not None
        assert has_response or has_escalation, (
            "Pipeline returned neither a response nor an escalation"
        )
        if has_response:
            assert data["tts_bytes"] > 0, "LLM responded but TTS returned no audio"

    async def test_pipeline_skip_tts(self):
        """POST /test/pipeline with skip_tts=true returns LLM text without TTS audio."""
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{CONTROL_SERVER}/test/pipeline",
                json={"text": "What is 2+2?", "skip_tts": True},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        has_response = len(data["llm_response"]) > 0
        has_escalation = data.get("escalation") is not None
        assert has_response or has_escalation, (
            "Pipeline returned neither a response nor an escalation"
        )
        assert data["tts_bytes"] == 0, "skip_tts=true but TTS bytes returned"

    async def test_pipeline_empty_text_rejected(self):
        """POST /test/pipeline with empty text returns 400."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{CONTROL_SERVER}/test/pipeline",
                json={"text": ""},
            )
        assert resp.status_code == 400

    async def test_pipeline_latency_reasonable(self):
        """Pipeline response time should be under 2 minutes."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{CONTROL_SERVER}/test/pipeline",
                json={"text": "Say OK.", "skip_tts": True},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["latency_ms"] < 120_000, f"Pipeline too slow: {data['latency_ms']}ms"


# ---------------------------------------------------------------------------
# 3. Discord gateway - confirm bot is connected
# ---------------------------------------------------------------------------


class TestDiscordGateway:
    """Verify the bel-audio bot is connected to Discord."""

    async def test_bot_is_authenticated(self):
        """GET /gateway/bot confirms the bot token is valid and connected."""
        token = _read_bot_token()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{DISCORD_API}/gateway/bot",
                headers={"Authorization": f"Bot {token}"},
            )
        assert resp.status_code == 200, f"Gateway auth failed: {resp.status_code} {resp.text}"
        data = resp.json()
        assert "url" in data, "Missing gateway URL in response"
        assert "session_start_limit" in data

    async def test_bot_user_info(self):
        """GET /users/@me returns the bel-audio bot user with expected ID."""
        token = _read_bot_token()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{DISCORD_API}/users/@me",
                headers={"Authorization": f"Bot {token}"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == BOT_USER_ID
        assert data["bot"] is True

    async def test_bot_in_guild(self):
        """GET /guilds/{id} confirms the bot is in the target guild."""
        token = _read_bot_token()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{DISCORD_API}/guilds/{GUILD_ID}",
                headers={"Authorization": f"Bot {token}"},
            )
        assert resp.status_code == 200, f"Bot not in guild: {resp.status_code}"
        data = resp.json()
        assert data["id"] == GUILD_ID


# ---------------------------------------------------------------------------
# 4. STT endpoint - speech-to-text via control server
# ---------------------------------------------------------------------------


class TestSTT:
    """Test the STT endpoint with synthetic audio."""

    async def test_stt_with_sine_wave(self):
        """POST /test/stt with a sine wave PCM returns a transcript (possibly empty)."""
        pcm = _make_pcm(duration_ms=1000, freq_hz=440)
        audio_b64 = base64.b64encode(pcm).decode()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{CONTROL_SERVER}/test/stt",
                json={"audio_b64": audio_b64},
            )
        assert resp.status_code == 200, f"STT failed: {resp.text}"
        data = resp.json()
        assert data["status"] == "ok"
        assert "transcript" in data
        assert data["audio_bytes"] == len(pcm)

    async def test_stt_with_silence(self):
        """POST /test/stt with silence returns an empty or near-empty transcript."""
        pcm = _make_silent_pcm(duration_ms=1000)
        audio_b64 = base64.b64encode(pcm).decode()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{CONTROL_SERVER}/test/stt",
                json={"audio_b64": audio_b64},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    async def test_stt_empty_audio_rejected(self):
        """POST /test/stt with no audio field returns 400."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{CONTROL_SERVER}/test/stt",
                json={},
            )
        assert resp.status_code == 400

    async def test_stt_invalid_base64_rejected(self):
        """POST /test/stt with invalid base64 returns 400."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{CONTROL_SERVER}/test/stt",
                json={"audio_b64": "!!!not-valid-base64!!!"},
            )
        assert resp.status_code == 400

    async def test_stt_latency_reasonable(self):
        """STT response time for 1s of audio should be under 10 seconds."""
        pcm = _make_pcm(duration_ms=1000, freq_hz=300)
        audio_b64 = base64.b64encode(pcm).decode()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{CONTROL_SERVER}/test/stt",
                json={"audio_b64": audio_b64},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["latency_ms"] < 10_000, f"STT too slow: {data['latency_ms']}ms"
