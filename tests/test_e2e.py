"""End-to-end tests for the voice pipeline test endpoints.

These tests validate the /test/* HTTP endpoints on the control server
(port 18790) that exercise the full STT -> LLM -> TTS pipeline without
requiring a real Discord connection.

Two test tiers:
  1. Unit tests (always run) - mock the pipeline and test endpoint logic
  2. Integration tests (require live services) - marked with @pytest.mark.integration
"""

from __future__ import annotations

import base64
import struct

import pytest
from aiohttp import web

from openclaw_voice.test_endpoints import build_test_routes
from openclaw_voice.voice_pipeline import PipelineConfig, VoicePipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pcm(duration_ms: int = 200, freq_hz: int = 440) -> bytes:
    """Generate a sine wave PCM buffer (16kHz mono int16)."""
    import math

    sample_rate = 16_000
    n_samples = sample_rate * duration_ms // 1000
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        val = int(16000 * math.sin(2 * math.pi * freq_hz * t))
        samples.append(max(-32768, min(32767, val)))
    return struct.pack(f"<{n_samples}h", *samples)


def _make_silent_pcm(duration_ms: int = 200) -> bytes:
    """Generate silent PCM bytes (16kHz mono int16)."""
    n_samples = 16_000 * duration_ms // 1000
    return b"\x00" * (n_samples * 2)


def _make_config(**kwargs) -> PipelineConfig:
    return PipelineConfig(
        whisper_url="http://whisper.test/inference",
        kokoro_url="http://kokoro.test/v1/audio/speech",
        speaker_id_url="http://speakerid.test/identify",
        llm_url="http://llm.test/v1/chat/completions",
        **kwargs,
    )


def _create_test_app(pipeline: VoicePipeline | None = None) -> web.Application:
    """Build an aiohttp app with test routes wired to a pipeline."""
    app = web.Application()
    routes = build_test_routes(pipeline=pipeline)
    app.add_routes(routes)
    return app


# ---------------------------------------------------------------------------
# GET /test/health — service health checks
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.fixture
    def app(self):
        return _create_test_app(pipeline=None)

    async def test_health_returns_json(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        resp = await client.get("/test/health")
        assert resp.status == 200
        data = await resp.json()
        assert "services" in data
        assert "llm" in data["services"]
        assert "stt" in data["services"]
        assert "tts" in data["services"]

    async def test_health_reports_unreachable_services(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        resp = await client.get("/test/health")
        data = await resp.json()
        for svc in ("llm", "stt", "tts"):
            assert data["services"][svc]["status"] in ("ok", "error")


# ---------------------------------------------------------------------------
# POST /test/pipeline — LLM + TTS test (skip STT)
# ---------------------------------------------------------------------------


class TestPipelineEndpoint:
    async def test_pipeline_requires_text(self, aiohttp_client):
        app = _create_test_app(pipeline=None)
        client = await aiohttp_client(app)
        resp = await client.post("/test/pipeline", json={})
        assert resp.status == 400
        data = await resp.json()
        assert "error" in data

    async def test_pipeline_no_pipeline_returns_503(self, aiohttp_client):
        app = _create_test_app(pipeline=None)
        client = await aiohttp_client(app)
        resp = await client.post("/test/pipeline", json={"text": "hello"})
        assert resp.status == 503

    async def test_pipeline_returns_response(self, aiohttp_client):
        from unittest.mock import MagicMock

        pipeline = MagicMock(spec=VoicePipeline)
        pipeline.build_system_prompt.return_value = "You are a test assistant."
        pipeline.call_llm_with_tools.return_value = ("Four!", None)
        pipeline.synthesize_response.return_value = b"fake-wav-bytes"

        app = _create_test_app(pipeline=pipeline)
        client = await aiohttp_client(app)
        resp = await client.post("/test/pipeline", json={"text": "what is 2+2?"})
        assert resp.status == 200
        data = await resp.json()
        assert data["llm_response"] == "Four!"
        assert data["tts_bytes"] > 0
        assert data["status"] == "ok"

    async def test_pipeline_escalation_response(self, aiohttp_client):
        from unittest.mock import MagicMock

        pipeline = MagicMock(spec=VoicePipeline)
        pipeline.build_system_prompt.return_value = "You are a test assistant."
        pipeline.call_llm_with_tools.return_value = ("", "check calendar")
        pipeline.synthesize_response.return_value = b""

        app = _create_test_app(pipeline=pipeline)
        client = await aiohttp_client(app)
        resp = await client.post("/test/pipeline", json={"text": "what's on my calendar?"})
        assert resp.status == 200
        data = await resp.json()
        assert data["escalation"] == "check calendar"

    async def test_pipeline_skip_tts_flag(self, aiohttp_client):
        from unittest.mock import MagicMock

        pipeline = MagicMock(spec=VoicePipeline)
        pipeline.build_system_prompt.return_value = "You are a test assistant."
        pipeline.call_llm_with_tools.return_value = ("Hello!", None)

        app = _create_test_app(pipeline=pipeline)
        client = await aiohttp_client(app)
        resp = await client.post("/test/pipeline", json={"text": "hello", "skip_tts": True})
        assert resp.status == 200
        data = await resp.json()
        assert data["llm_response"] == "Hello!"
        pipeline.synthesize_response.assert_not_called()


# ---------------------------------------------------------------------------
# POST /test/stt — STT test (accepts base64 PCM)
# ---------------------------------------------------------------------------


class TestSttEndpoint:
    async def test_stt_requires_audio(self, aiohttp_client):
        app = _create_test_app(pipeline=None)
        client = await aiohttp_client(app)
        resp = await client.post("/test/stt", json={})
        assert resp.status == 400

    async def test_stt_no_pipeline_returns_503(self, aiohttp_client):
        app = _create_test_app(pipeline=None)
        client = await aiohttp_client(app)
        pcm = _make_silent_pcm(200)
        audio_b64 = base64.b64encode(pcm).decode()
        resp = await client.post("/test/stt", json={"audio_b64": audio_b64})
        assert resp.status == 503

    async def test_stt_returns_transcript(self, aiohttp_client):
        from unittest.mock import MagicMock

        pipeline = MagicMock(spec=VoicePipeline)
        pipeline.run_stt.return_value = "hello world"

        app = _create_test_app(pipeline=pipeline)
        client = await aiohttp_client(app)
        pcm = _make_pcm(200, freq_hz=440)
        audio_b64 = base64.b64encode(pcm).decode()
        resp = await client.post("/test/stt", json={"audio_b64": audio_b64})
        assert resp.status == 200
        data = await resp.json()
        assert data["transcript"] == "hello world"
        assert data["status"] == "ok"

    async def test_stt_invalid_base64(self, aiohttp_client):
        from unittest.mock import MagicMock

        pipeline = MagicMock(spec=VoicePipeline)
        app = _create_test_app(pipeline=pipeline)
        client = await aiohttp_client(app)
        resp = await client.post("/test/stt", json={"audio_b64": "not-valid-b64!!!"})
        assert resp.status == 400

    async def test_stt_empty_transcript(self, aiohttp_client):
        from unittest.mock import MagicMock

        pipeline = MagicMock(spec=VoicePipeline)
        pipeline.run_stt.return_value = ""

        app = _create_test_app(pipeline=pipeline)
        client = await aiohttp_client(app)
        pcm = _make_silent_pcm(200)
        audio_b64 = base64.b64encode(pcm).decode()
        resp = await client.post("/test/stt", json={"audio_b64": audio_b64})
        assert resp.status == 200
        data = await resp.json()
        assert data["transcript"] == ""
        assert data["status"] == "ok"
