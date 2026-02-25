"""Tests for the Wyoming STT bridge."""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from openclaw_voice.stt_bridge import STTConfig, WhisperEventHandler


def _make_config(**kwargs) -> STTConfig:
    return STTConfig(
        whisper_url="http://mock-whisper/inference",
        speaker_id_url="http://mock-speaker/identify",
        **kwargs,
    )


def _make_wav_bytes(duration_samples: int = 16000) -> bytes:
    """Create a minimal valid WAV file (1s of silence at 16kHz mono 16-bit)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * duration_samples)
    return buf.getvalue()


class TestSTTConfig:
    def test_defaults(self):
        cfg = STTConfig()
        assert cfg.port == 10300
        assert cfg.whisper_url == "http://localhost:8001/inference"
        assert cfg.enable_speaker_id is False
        assert cfg.transcript_log is None

    def test_custom_values(self):
        cfg = STTConfig(
            port=9999,
            whisper_url="http://myserver:8001/inference",
            enable_speaker_id=True,
            transcript_log=Path("/tmp/test.jsonl"),
        )
        assert cfg.port == 9999
        assert cfg.enable_speaker_id is True
        assert cfg.transcript_log == Path("/tmp/test.jsonl")


class TestBuildWav:
    """Unit tests for the WAV packaging helper."""

    def test_build_wav_valid(self):
        config = _make_config()
        handler = WhisperEventHandler.__new__(WhisperEventHandler)
        handler.config = config
        handler.channels = 1
        handler.sample_width = 2
        handler.sample_rate = 16000

        pcm = b"\x00\x01" * 100
        wav = handler._build_wav(pcm)

        buf = io.BytesIO(wav)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000


class TestTranscribeSyncMocked:
    """Test _transcribe_sync with a mocked HTTP response."""

    def test_successful_transcription(self):
        config = _make_config()
        handler = WhisperEventHandler.__new__(WhisperEventHandler)
        handler.config = config

        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "  hello world  "}
        mock_response.raise_for_status = MagicMock()

        with patch("openclaw_voice.stt_bridge.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = handler._transcribe_sync(b"fake-wav")

        assert result == "hello world"

    def test_transcription_http_error_returns_empty(self):
        config = _make_config()
        handler = WhisperEventHandler.__new__(WhisperEventHandler)
        handler.config = config

        with patch("openclaw_voice.stt_bridge.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = Exception("connection refused")
            mock_client_cls.return_value = mock_client

            result = handler._transcribe_sync(b"fake-wav")

        assert result == ""


class TestIdentifySpeakerMocked:
    def test_successful_identification(self):
        config = _make_config(enable_speaker_id=True)
        handler = WhisperEventHandler.__new__(WhisperEventHandler)
        handler.config = config

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "speaker": "Liam",
            "confidence": 0.92,
            "access_level": "full",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("openclaw_voice.stt_bridge.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = handler._identify_speaker_sync(b"fake-wav")

        assert result["speaker"] == "Liam"
        assert result["confidence"] == 0.92

    def test_identification_failure_returns_defaults(self):
        config = _make_config(enable_speaker_id=True)
        handler = WhisperEventHandler.__new__(WhisperEventHandler)
        handler.config = config

        with patch("openclaw_voice.stt_bridge.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = Exception("timeout")
            mock_client_cls.return_value = mock_client

            result = handler._identify_speaker_sync(b"fake-wav")

        assert result["speaker"] is None
        assert result["access_level"] == "basic"


class TestTranscriptLog:
    def test_write_log(self, tmp_path):
        log_file = tmp_path / "transcripts.jsonl"
        config = _make_config(transcript_log=log_file)
        handler = WhisperEventHandler.__new__(WhisperEventHandler)
        handler.config = config

        handler._write_transcript_log("hello", "Liam", "full", 0.95)

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["text"] == "hello"
        assert entry["speaker"] == "Liam"
        assert entry["confidence"] == 0.95
        assert entry["posted"] is False

    def test_no_log_when_path_is_none(self):
        config = _make_config(transcript_log=None)
        handler = WhisperEventHandler.__new__(WhisperEventHandler)
        handler.config = config
        # Should not raise
        handler._write_transcript_log("hello", "unknown", "basic", 0.0)
