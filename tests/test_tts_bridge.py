"""Tests for the Wyoming TTS bridge."""

from __future__ import annotations

import pytest

from openclaw_voice.tts_bridge import (
    NATIVE_VOICES,
    VOICE_ALIASES,
    TTSConfig,
    _all_voice_names,
    _resolve_voice,
)


class TestTTSConfig:
    def test_defaults(self):
        cfg = TTSConfig()
        assert cfg.port == 10200
        assert cfg.kokoro_url == "http://localhost:8002/v1/audio/speech"
        assert cfg.default_voice == "af_heart"
        assert cfg.speed == 1.0

    def test_custom_values(self):
        cfg = TTSConfig(
            port=9999,
            kokoro_url="http://myserver:8002/v1/audio/speech",
            default_voice="am_adam",
            speed=1.2,
        )
        assert cfg.port == 9999
        assert cfg.default_voice == "am_adam"
        assert cfg.speed == 1.2


class TestVoiceResolution:
    def test_openai_alias_resolves(self):
        assert _resolve_voice("alloy") == "af_heart"
        assert _resolve_voice("echo") == "am_adam"
        assert _resolve_voice("nova") == "af_nova"

    def test_native_voice_passthrough(self):
        assert _resolve_voice("af_heart") == "af_heart"
        assert _resolve_voice("am_michael") == "am_michael"

    def test_unknown_voice_passthrough(self):
        assert _resolve_voice("custom_voice") == "custom_voice"

    def test_all_aliases_mapped(self):
        for alias, native in VOICE_ALIASES.items():
            assert _resolve_voice(alias) == native
            assert native in NATIVE_VOICES or native.startswith(("af_", "am_", "bf_", "bm_"))


class TestAllVoiceNames:
    def test_includes_native_voices(self):
        cfg = TTSConfig()
        names = _all_voice_names(cfg)
        for v in NATIVE_VOICES:
            assert v in names

    def test_includes_aliases(self):
        cfg = TTSConfig()
        names = _all_voice_names(cfg)
        for alias in VOICE_ALIASES:
            assert alias in names

    def test_includes_extra_voices(self):
        cfg = TTSConfig(extra_voices=["custom_1", "custom_2"])
        names = _all_voice_names(cfg)
        assert "custom_1" in names
        assert "custom_2" in names

    def test_no_duplicates(self):
        cfg = TTSConfig()
        names = _all_voice_names(cfg)
        assert len(names) == len(set(names))


class TestKokoroEventHandlerDescribe:
    """Integration-style test for the Describe handler."""

    @pytest.mark.asyncio
    async def test_describe_returns_info(self):
        from unittest.mock import AsyncMock, MagicMock

        from wyoming.info import Describe

        from openclaw_voice.tts_bridge import KokoroEventHandler

        config = TTSConfig()
        handler = KokoroEventHandler.__new__(KokoroEventHandler)
        handler.config = config
        # Mock write_event
        written_events = []
        handler.write_event = AsyncMock(side_effect=lambda e: written_events.append(e))

        describe_event = Describe().event()
        await handler._handle_describe()

        assert len(written_events) == 1
        from wyoming.info import Info

        info = Info.from_event(written_events[0])
        assert info.tts is not None
        assert len(info.tts) == 1
        assert info.tts[0].name == "kokoro"
        # Should have voices listed
        assert len(info.tts[0].voices) > 0


class TestKokoroEventHandlerSynthesize:
    @pytest.mark.asyncio
    async def test_synthesize_empty_text_sends_stop(self):
        from unittest.mock import AsyncMock

        from openclaw_voice.tts_bridge import KokoroEventHandler
        from wyoming.tts import Synthesize

        config = TTSConfig(kokoro_url="http://mock-kokoro/v1/audio/speech")
        handler = KokoroEventHandler.__new__(KokoroEventHandler)
        handler.config = config

        written_events = []
        handler.write_event = AsyncMock(side_effect=lambda e: written_events.append(e))

        synth = Synthesize(text="   ")
        await handler._handle_synthesize(synth)

        from wyoming.audio import AudioStop

        assert any(AudioStop.is_type(e.type) for e in written_events)

    @pytest.mark.asyncio
    async def test_synthesize_streams_audio(self):
        """Mock Kokoro to return WAV bytes and verify AudioChunk events are sent."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from openclaw_voice.tts_bridge import KokoroEventHandler, AUDIO_CHUNK_SIZE
        from wyoming.audio import AudioChunk, AudioStop, AudioStart
        from wyoming.tts import Synthesize

        config = TTSConfig(kokoro_url="http://mock-kokoro/v1/audio/speech")
        handler = KokoroEventHandler.__new__(KokoroEventHandler)
        handler.config = config

        written_events = []
        handler.write_event = AsyncMock(side_effect=lambda e: written_events.append(e))

        # Fake audio data — 2 full chunks + partial
        fake_audio = b"\x01\x02" * (AUDIO_CHUNK_SIZE + 100)

        # Build an async iterator over the fake audio
        async def _aiter_bytes(chunk_size=4096):
            offset = 0
            while offset < len(fake_audio):
                yield fake_audio[offset : offset + chunk_size]
                offset += chunk_size

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_bytes = _aiter_bytes
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream.return_value = mock_resp

        with patch("openclaw_voice.tts_bridge.httpx.AsyncClient", return_value=mock_client):
            synth = Synthesize(text="Hello, world!")
            await handler._handle_synthesize(synth)

        types = [e.type for e in written_events]
        assert AudioStart.type in types
        assert AudioChunk.type in types
        assert AudioStop.type in types
        # AudioStop should be last
        assert written_events[-1].type == AudioStop.type
