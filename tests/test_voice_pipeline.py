"""Tests for VoicePipeline (voice_pipeline.py)."""

from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import pytest

from openclaw_voice.voice_pipeline import (
    PipelineConfig,
    VoicePipeline,
    _pcm_to_wav,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pcm(duration_ms: int = 200) -> bytes:
    """Generate silent PCM bytes (16kHz mono int16)."""
    n_samples = 16_000 * duration_ms // 1000
    return b"\x00" * (n_samples * 2)


def _make_wav(duration_ms: int = 200) -> bytes:
    """Return a valid WAV file with silent audio."""
    return _pcm_to_wav(_make_pcm(duration_ms))


def _make_config(**kwargs) -> PipelineConfig:
    return PipelineConfig(
        whisper_url="http://whisper.test/inference",
        kokoro_url="http://kokoro.test/v1/audio/speech",
        speaker_id_url="http://speakerid.test/identify",
        llm_url="http://llm.test/v1/chat/completions",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# _pcm_to_wav utility
# ---------------------------------------------------------------------------

class TestPcmToWav:
    def test_returns_valid_wav(self):
        pcm = _make_pcm(100)
        wav = _pcm_to_wav(pcm)
        buf = io.BytesIO(wav)
        with wave.open(buf) as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16_000
            assert wf.getnframes() == len(pcm) // 2

    def test_empty_pcm(self):
        wav = _pcm_to_wav(b"")
        assert wav[:4] == b"RIFF"  # still a valid WAV container

    def test_custom_parameters(self):
        pcm = b"\x00" * 1000
        wav = _pcm_to_wav(pcm, sample_rate=8000, sample_width=2, channels=1)
        buf = io.BytesIO(wav)
        with wave.open(buf) as wf:
            assert wf.getframerate() == 8000


# ---------------------------------------------------------------------------
# PipelineConfig defaults
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.whisper_url == "http://localhost:8001/inference"
        assert cfg.kokoro_url == "http://localhost:8002/v1/audio/speech"
        assert cfg.llm_url == "http://localhost:8000/v1/chat/completions"
        assert cfg.llm_model == "Qwen/Qwen2.5-32B-Instruct"
        assert cfg.tts_voice == "af_heart"
        assert cfg.max_history_turns == 20
        assert not cfg.enable_speaker_id

    def test_custom_values(self):
        cfg = PipelineConfig(
            llm_model="my-model",
            tts_voice="af_nova",
            max_history_turns=5,
        )
        assert cfg.llm_model == "my-model"
        assert cfg.tts_voice == "af_nova"
        assert cfg.max_history_turns == 5


# ---------------------------------------------------------------------------
# VoicePipeline init
# ---------------------------------------------------------------------------

class TestVoicePipelineInit:
    def test_default_config(self):
        pipeline = VoicePipeline()
        assert pipeline.config.llm_model == "Qwen/Qwen2.5-32B-Instruct"
        assert pipeline.history == []

    def test_custom_channel_id(self):
        pipeline = VoicePipeline(channel_id="guild-42-channel-99")
        assert pipeline._channel_id == "guild-42-channel-99"

    def test_history_empty_on_init(self):
        pipeline = VoicePipeline(_make_config())
        assert pipeline.history == []


# ---------------------------------------------------------------------------
# Full pipeline: mock everything
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def _make_pipeline(self, **kwargs) -> VoicePipeline:
        return VoicePipeline(config=_make_config(**kwargs), channel_id="test-channel")

    @pytest.fixture
    def pipeline(self):
        return self._make_pipeline()

    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_full_pipeline_produces_response(
        self,
        mock_whisper_cls,
        mock_kokoro_cls,
    ):
        """Happy path: whisper returns transcript, LLM returns text, kokoro returns audio."""
        # Mock WhisperFacade
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = "Hello there"
        mock_whisper_cls.return_value = mock_whisper

        # Mock KokoroFacade — stream_audio is an async generator
        async def _fake_stream(*args, **kwargs):
            yield _make_wav(100)
        mock_kokoro = MagicMock()
        mock_kokoro.stream_audio = _fake_stream
        mock_kokoro_cls.return_value = mock_kokoro

        pipeline = self._make_pipeline()

        # Mock LLM call
        with patch("httpx.Client") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Hi there!"}}]
            }
            mock_resp.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.__enter__ = lambda s: mock_client
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            transcript, text, audio = pipeline.process_utterance(_make_pcm(), user_id="123")

        assert text == "Hi there!"
        assert isinstance(audio, bytes)
        assert len(audio) > 0

    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_empty_transcript_returns_empty(self, mock_whisper_cls, mock_kokoro_cls):
        """Empty transcript → return ('', b'') without calling LLM."""
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = ""  # Empty transcript
        mock_whisper_cls.return_value = mock_whisper

        mock_kokoro = MagicMock()
        mock_kokoro_cls.return_value = mock_kokoro

        pipeline = self._make_pipeline()

        with patch("httpx.Client") as mock_client_cls:
            transcript, text, audio = pipeline.process_utterance(_make_pcm(), user_id="456")

        assert text == ""
        assert audio == b""
        # LLM should NOT have been called
        mock_client_cls.assert_not_called()

    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_whisper_failure_returns_empty(self, mock_whisper_cls, mock_kokoro_cls):
        """WhisperFacade returns empty on failure → pipeline returns empty."""
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = ""
        mock_whisper_cls.return_value = mock_whisper

        mock_kokoro_cls.return_value = MagicMock()

        pipeline = self._make_pipeline()
        transcript, text, audio = pipeline.process_utterance(_make_pcm(), user_id="789")
        assert text == ""
        assert audio == b""

    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_llm_service_down_returns_empty(self, mock_whisper_cls, mock_kokoro_cls):
        """LLM connection error → pipeline returns empty, no crash."""
        import httpx

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = "Say something"
        mock_whisper_cls.return_value = mock_whisper

        mock_kokoro_cls.return_value = MagicMock()

        pipeline = self._make_pipeline()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = lambda s: mock_client
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value = mock_client

            transcript, text, audio = pipeline.process_utterance(_make_pcm(), user_id="999")

        assert text == ""
        assert audio == b""

    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_unexpected_exception_returns_empty(self, mock_whisper_cls, mock_kokoro_cls):
        """Any unexpected exception in the pipeline returns ('', b''), no crash."""
        mock_whisper = MagicMock()
        mock_whisper.transcribe.side_effect = RuntimeError("Unexpected crash!")
        mock_whisper_cls.return_value = mock_whisper

        mock_kokoro_cls.return_value = MagicMock()

        pipeline = self._make_pipeline()
        transcript, text, audio = pipeline.process_utterance(_make_pcm(), user_id="crash-user")
        assert text == ""
        assert audio == b""


# ---------------------------------------------------------------------------
# Conversation history management
# ---------------------------------------------------------------------------

class TestConversationHistory:
    def _make_pipeline(self, max_turns: int = 5) -> VoicePipeline:
        return VoicePipeline(
            config=_make_config(max_history_turns=max_turns),
            channel_id="hist-test",
        )

    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_history_appends_user_and_assistant(self, mock_whisper_cls, mock_kokoro_cls):
        """After a successful exchange, history should have user+assistant entries."""
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = "Hello"
        mock_whisper_cls.return_value = mock_whisper

        async def _fake_stream(*args, **kwargs):
            yield _make_wav(50)
        mock_kokoro = MagicMock()
        mock_kokoro.stream_audio = _fake_stream
        mock_kokoro_cls.return_value = mock_kokoro

        pipeline = self._make_pipeline(max_turns=5)

        with patch("httpx.Client") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Hey!"}}]
            }
            mock_resp.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.__enter__ = lambda s: mock_client
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            pipeline.process_utterance(_make_pcm(), user_id="u1")

        history = pipeline.history
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hey!"

    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_history_bounded_by_max_turns(self, mock_whisper_cls, mock_kokoro_cls):
        """History should not exceed max_history_turns * 2 entries."""
        max_turns = 3
        pipeline = self._make_pipeline(max_turns=max_turns)

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = "Hi"
        mock_whisper_cls.return_value = mock_whisper

        async def _fake_stream(*args, **kwargs):
            yield _make_wav(50)
        mock_kokoro = MagicMock()
        mock_kokoro.stream_audio = _fake_stream
        mock_kokoro_cls.return_value = mock_kokoro

        with patch("httpx.Client") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Hello!"}}]
            }
            mock_resp.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.__enter__ = lambda s: mock_client
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            # Run more exchanges than max_turns
            for _ in range(max_turns + 3):
                pipeline.process_utterance(_make_pcm(), user_id="u1")

        assert len(pipeline.history) <= max_turns * 2

    def test_clear_history(self):
        pipeline = self._make_pipeline()
        # Manually inject history
        pipeline._history.append({"role": "user", "content": "test"})
        pipeline._history.append({"role": "assistant", "content": "reply"})

        pipeline.clear_history()
        assert pipeline.history == []

    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_failed_llm_does_not_add_to_history(self, mock_whisper_cls, mock_kokoro_cls):
        """When LLM fails, the user message should be rolled back from history."""
        import httpx

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = "Will this be saved?"
        mock_whisper_cls.return_value = mock_whisper
        mock_kokoro_cls.return_value = MagicMock()

        pipeline = self._make_pipeline()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = lambda s: mock_client
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.ConnectError("down")
            mock_client_cls.return_value = mock_client

            pipeline.process_utterance(_make_pcm(), user_id="u1")

        # History should be empty — user message rolled back
        assert pipeline.history == []


# ---------------------------------------------------------------------------
# Speaker ID (optional)
# ---------------------------------------------------------------------------

class TestSpeakerIdIntegration:
    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_speaker_id_prefixes_message(self, mock_whisper_cls, mock_kokoro_cls):
        """When speaker ID is enabled and returns a name, message should be prefixed."""
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = "I have a question"
        mock_whisper_cls.return_value = mock_whisper

        async def _fake_stream(*args, **kwargs):
            yield _make_wav(50)
        mock_kokoro = MagicMock()
        mock_kokoro.stream_audio = _fake_stream
        mock_kokoro_cls.return_value = mock_kokoro

        pipeline = VoicePipeline(
            config=_make_config(enable_speaker_id=True),
            channel_id="test",
        )

        with patch("httpx.Client") as mock_client_cls:
            mock_resp_llm = MagicMock()
            mock_resp_llm.json.return_value = {
                "choices": [{"message": {"content": "Answer"}}]
            }
            mock_resp_llm.raise_for_status = MagicMock()

            mock_resp_sid = MagicMock()
            mock_resp_sid.status_code = 200
            mock_resp_sid.json.return_value = {"speaker": "Liam"}
            mock_resp_sid.raise_for_status = MagicMock()

            call_count = [0]

            def _post(url, **kwargs):
                call_count[0] += 1
                if "identify" in url:
                    return mock_resp_sid
                return mock_resp_llm

            mock_client = MagicMock()
            mock_client.__enter__ = lambda s: mock_client
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = _post
            mock_client_cls.return_value = mock_client

            pipeline.process_utterance(_make_pcm(), user_id="u1")

        history = pipeline.history
        assert history[0]["content"] == "[Liam]: I have a question"
