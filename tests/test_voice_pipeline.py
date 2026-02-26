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
        assert cfg.llm_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
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
# VoicePipeline init (v2: stateless)
# ---------------------------------------------------------------------------


class TestVoicePipelineInit:
    def test_default_config(self):
        pipeline = VoicePipeline()
        assert pipeline.config.llm_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"

    def test_custom_channel_id(self):
        pipeline = VoicePipeline(channel_id="guild-42-channel-99")
        assert pipeline._channel_id == "guild-42-channel-99"

    def test_stateless_no_internal_history(self):
        """v2 pipeline is stateless — history is owned by ConversationLog, not VoicePipeline."""
        pipeline = VoicePipeline(_make_config())
        assert not hasattr(pipeline, "history")
        assert not hasattr(pipeline, "_history")


# ---------------------------------------------------------------------------
# run_stt: mock WhisperFacade
# ---------------------------------------------------------------------------


class TestRunStt:
    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_returns_transcript(self, mock_whisper_cls):
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = "Hello there"
        mock_whisper_cls.return_value = mock_whisper

        pipeline = VoicePipeline(config=_make_config(), channel_id="test")
        result = pipeline.run_stt(_make_pcm(), user_id="u1")
        assert result == "Hello there"

    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_empty_transcript_returns_empty_string(self, mock_whisper_cls):
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = ""
        mock_whisper_cls.return_value = mock_whisper

        pipeline = VoicePipeline(config=_make_config(), channel_id="test")
        result = pipeline.run_stt(_make_pcm(), user_id="u1")
        assert result == ""

    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_whisper_exception_returns_empty_string(self, mock_whisper_cls):
        mock_whisper = MagicMock()
        mock_whisper.transcribe.side_effect = RuntimeError("Whisper crashed")
        mock_whisper_cls.return_value = mock_whisper

        pipeline = VoicePipeline(config=_make_config(), channel_id="test")
        result = pipeline.run_stt(_make_pcm(), user_id="u1")
        assert result == ""

    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_hallucination_filtered(self, mock_whisper_cls):
        """Known Whisper hallucinations should be filtered to empty string."""
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = "thank you for watching"
        mock_whisper_cls.return_value = mock_whisper

        pipeline = VoicePipeline(config=_make_config(), channel_id="test")
        result = pipeline.run_stt(_make_pcm(), user_id="u1")
        assert result == ""

    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    def test_hallucination_prefix_stripped(self, mock_whisper_cls):
        """Repeated hallucination lines at start should be stripped."""
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = (
            "Thank you.\n Thank you.\n Thank you.\n Okay, read me the last PR."
        )
        mock_whisper_cls.return_value = mock_whisper

        pipeline = VoicePipeline(config=_make_config(), channel_id="test")
        result = pipeline.run_stt(_make_pcm(), user_id="u1")
        assert "thank you" not in result.lower()
        assert "read me the last PR" in result


# ---------------------------------------------------------------------------
# strip_hallucination_prefix unit tests
# ---------------------------------------------------------------------------


class TestStripHallucinationPrefix:
    def test_all_hallucinations_returns_empty(self):
        text = "Thank you.\n Thank you.\n Thanks."
        assert VoicePipeline.strip_hallucination_prefix(text) == ""

    def test_leading_hallucinations_stripped(self):
        text = "Thank you.\n Thank you.\n Okay, let's go."
        result = VoicePipeline.strip_hallucination_prefix(text)
        assert result == "Okay, let's go."

    def test_no_hallucinations_unchanged(self):
        text = "Read me the last PR please."
        result = VoicePipeline.strip_hallucination_prefix(text)
        assert result == "Read me the last PR please."

    def test_many_repeated_thank_yous(self):
        lines = ["Thank you."] * 50 + ["What is the weather?"]
        text = "\n ".join(lines)
        result = VoicePipeline.strip_hallucination_prefix(text)
        assert result == "What is the weather?"

    def test_empty_string(self):
        assert VoicePipeline.strip_hallucination_prefix("") == ""

    def test_mixed_hallucination_types(self):
        text = "Thanks.\n Bye.\n Hmm.\n So tell me about the project."
        result = VoicePipeline.strip_hallucination_prefix(text)
        assert result == "So tell me about the project."

    def test_preserves_mid_text_hallucination_words(self):
        """Hallucination words in the middle of real speech should NOT be stripped."""
        text = "Thank you.\n I want to say thanks for the help.\n What's next?"
        result = VoicePipeline.strip_hallucination_prefix(text)
        # "I want to say thanks for the help." is real speech (not in hallucination set)
        assert "thanks for the help" in result


# ---------------------------------------------------------------------------
# call_llm_with_tools: mock httpx.Client
# ---------------------------------------------------------------------------


class TestCallLlmWithTools:
    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    def test_returns_text_response(self, mock_kokoro_cls, mock_whisper_cls):
        pipeline = VoicePipeline(config=_make_config(), channel_id="test")

        with patch("httpx.Client") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
            mock_resp.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.__enter__ = lambda s: mock_client
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            messages = [
                {"role": "system", "content": "You are a voice assistant."},
                {"role": "user", "content": "Hello"},
            ]
            text, escalation = pipeline.call_llm_with_tools(messages)

        assert text == "Hi!"
        assert escalation is None

    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    def test_llm_down_returns_empty(self, mock_kokoro_cls, mock_whisper_cls):
        import httpx

        pipeline = VoicePipeline(config=_make_config(), channel_id="test")

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = lambda s: mock_client
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value = mock_client

            messages = [
                {"role": "system", "content": "You are a voice assistant."},
                {"role": "user", "content": "Hello"},
            ]
            text, escalation = pipeline.call_llm_with_tools(messages)

        assert text == ""
        assert escalation is None

    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    def test_escalate_tool_call(self, mock_kokoro_cls, mock_whisper_cls):
        """LLM calling escalate() returns ('', escalation_request)."""
        import json

        pipeline = VoicePipeline(config=_make_config(), channel_id="test")

        with patch("httpx.Client") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "escalate",
                                        "arguments": json.dumps({"request": "Need Bel's help"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
            mock_resp.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.__enter__ = lambda s: mock_client
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            messages = [
                {"role": "system", "content": "You are a voice assistant."},
                {"role": "user", "content": "Who is Bel?"},
            ]
            text, escalation = pipeline.call_llm_with_tools(messages)

        assert text == ""
        assert escalation == "Need Bel's help"


# ---------------------------------------------------------------------------
# synthesize_response: mock KokoroFacade
# ---------------------------------------------------------------------------


class TestSynthesizeResponse:
    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    def test_returns_wav_bytes(self, mock_kokoro_cls, mock_whisper_cls):
        async def _fake_stream(*args, **kwargs):
            yield _make_wav(100)

        mock_kokoro = MagicMock()
        mock_kokoro.stream_audio = _fake_stream
        mock_kokoro_cls.return_value = mock_kokoro

        pipeline = VoicePipeline(config=_make_config(), channel_id="test")
        result = pipeline.synthesize_response("Hello there!", user_id="u1")
        assert isinstance(result, bytes)
        assert len(result) > 0

    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    def test_empty_text_returns_empty(self, mock_kokoro_cls, mock_whisper_cls):
        pipeline = VoicePipeline(config=_make_config(), channel_id="test")
        result = pipeline.synthesize_response("", user_id="u1")
        assert result == b""

    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    def test_kokoro_failure_returns_empty(self, mock_kokoro_cls, mock_whisper_cls):
        async def _bad_stream(*args, **kwargs):
            raise RuntimeError("TTS server down")
            yield  # make it a generator

        mock_kokoro = MagicMock()
        mock_kokoro.stream_audio = _bad_stream
        mock_kokoro_cls.return_value = mock_kokoro

        pipeline = VoicePipeline(config=_make_config(), channel_id="test")
        result = pipeline.synthesize_response("Say something", user_id="u1")
        assert result == b""


# ---------------------------------------------------------------------------
# Speaker ID integration (run_stt with speaker_id enabled)
# ---------------------------------------------------------------------------


class TestSpeakerIdIntegration:
    @patch("openclaw_voice.voice_pipeline.WhisperFacade")
    @patch("openclaw_voice.voice_pipeline.KokoroFacade")
    def test_run_stt_returns_raw_transcript(self, mock_kokoro_cls, mock_whisper_cls):
        """run_stt returns raw transcript; speaker ID prefixing is the caller's responsibility."""
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = "I have a question"
        mock_whisper_cls.return_value = mock_whisper
        mock_kokoro_cls.return_value = MagicMock()

        pipeline = VoicePipeline(
            config=_make_config(enable_speaker_id=True),
            channel_id="test",
        )
        transcript = pipeline.run_stt(_make_pcm(), user_id="u1")
        # VoicePipeline.run_stt returns the transcript without speaker prefix;
        # the discord_bot layer is responsible for speaker identification and prefixing.
        assert transcript == "I have a question"
