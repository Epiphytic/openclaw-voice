"""
Wyoming STT Bridge — whisper.cpp backend.

Bridges the Wyoming protocol (Home Assistant) to a whisper.cpp HTTP server.
Optionally queries a speaker identification server in parallel with transcription.

Wyoming protocol flow:
  1. Client connects and sends Transcribe (language hint)
  2. Client streams AudioChunk events (raw PCM)
  3. Client sends AudioStop
  4. Bridge transcribes audio via whisper.cpp
  5. Bridge optionally identifies speaker via speaker-id server
  6. Bridge sends Transcript event back

Usage:
    from openclaw_voice.stt_bridge import run_stt_bridge, STTConfig

    config = STTConfig(
        host="0.0.0.0",
        port=10300,
        whisper_url="http://localhost:8001/inference",
        speaker_id_url="http://localhost:8003/identify",
        enable_speaker_id=True,
        transcript_log=Path("/var/log/transcripts.jsonl"),
    )
    asyncio.run(run_stt_bridge(config))
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import wave
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

log = logging.getLogger("openclaw_voice.stt")


@dataclass
class STTConfig:
    """Configuration for the Wyoming STT bridge."""

    # Server binding
    host: str = "0.0.0.0"
    port: int = 10300

    # Upstream services
    whisper_url: str = "http://localhost:8001/inference"
    speaker_id_url: str = "http://localhost:8003/identify"

    # Feature flags
    enable_speaker_id: bool = False

    # Speaker ID thresholds
    speaker_id_threshold: float = 0.75

    # Transcript logging (set to None to disable)
    transcript_log: Path | None = None

    # Whisper model metadata (for Describe response)
    model_name: str = "large-v3-turbo"
    model_description: str = "Large v3 Turbo (GPU accelerated)"
    languages: list[str] = field(default_factory=lambda: ["en"])

    # HTTP timeouts (seconds)
    whisper_timeout: float = 30.0
    speaker_id_timeout: float = 10.0


class WhisperEventHandler(AsyncEventHandler):
    """Handles one Wyoming ASR session (one client connection)."""

    def __init__(
        self,
        *args: Any,
        config: STTConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.audio_bytes = bytearray()
        self.sample_rate: int = 16000
        self.sample_width: int = 2
        self.channels: int = 1

    async def handle_event(self, event: Event) -> bool:
        """Dispatch incoming Wyoming events."""

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.audio_bytes.extend(chunk.audio)
            self.sample_rate = chunk.rate
            self.sample_width = chunk.width
            self.channels = chunk.channels
            return True

        if AudioStop.is_type(event.type):
            return await self._handle_audio_stop()

        if Transcribe.is_type(event.type):
            # Language hint — accept and wait for audio
            return True

        if Describe.is_type(event.type):
            info = Info(
                asr=[
                    AsrProgram(
                        name="whisper.cpp",
                        description="whisper.cpp (openclaw-voice bridge)",
                        attribution=Attribution(
                            name="OpenAI",
                            url="https://github.com/openai/whisper",
                        ),
                        installed=True,
                        models=[
                            AsrModel(
                                name=self.config.model_name,
                                description=self.config.model_description,
                                attribution=Attribution(
                                    name="OpenAI",
                                    url="https://github.com/openai/whisper",
                                ),
                                installed=True,
                                languages=self.config.languages,
                            )
                        ],
                    )
                ],
            )
            self.write_event(info.event())
            return True

        return True

    async def _handle_audio_stop(self) -> bool:
        """Transcribe buffered audio and optionally identify speaker."""
        if not self.audio_bytes:
            log.warning("AudioStop received with no audio buffered")
            await self.write_event(Transcript(text="").event())
            return False

        log.info("Received %d bytes of audio", len(self.audio_bytes))

        wav_data = self._build_wav(bytes(self.audio_bytes))

        # Run transcription and (optional) speaker ID in parallel
        loop = asyncio.get_event_loop()
        tasks: list[asyncio.Task] = [
            loop.run_in_executor(None, self._transcribe_sync, wav_data)
        ]
        if self.config.enable_speaker_id:
            tasks.append(
                loop.run_in_executor(None, self._identify_speaker_sync, wav_data)
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        text: str = results[0] if not isinstance(results[0], Exception) else ""
        if isinstance(results[0], Exception):
            log.error("Transcription raised exception: %s", results[0])

        speaker_info: dict | None = None
        if self.config.enable_speaker_id and len(results) > 1:
            if isinstance(results[1], Exception):
                log.warning("Speaker ID raised exception: %s", results[1])
            else:
                speaker_info = results[1]

        speaker_name = "unknown"
        access_level = "basic"
        confidence = 0.0

        if speaker_info:
            speaker_name = speaker_info.get("speaker") or "unknown"
            access_level = speaker_info.get("access_level", "basic")
            confidence = float(speaker_info.get("confidence", 0))
            log.info(
                "Speaker: %s (confidence=%.2f, access=%s)",
                speaker_name,
                confidence,
                access_level,
            )
            text_with_meta = (
                f"[speaker:{speaker_name}|access:{access_level}|conf:{confidence:.2f}] {text}"
            )
        else:
            text_with_meta = text

        self._write_transcript_log(text, speaker_name, access_level, confidence)
        log.info("Transcript: %s", text)

        await self.write_event(Transcript(text=text_with_meta).event())

        # Reset for next utterance
        self.audio_bytes = bytearray()
        return False

    def _build_wav(self, pcm: bytes) -> bytes:
        """Wrap raw PCM bytes in a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()

    def _transcribe_sync(self, wav_data: bytes) -> str:
        """POST audio to whisper.cpp and return transcript text (synchronous)."""
        try:
            with httpx.Client(timeout=self.config.whisper_timeout) as client:
                response = client.post(
                    self.config.whisper_url,
                    files={"file": ("audio.wav", wav_data, "audio/wav")},
                    data={"response_format": "json"},
                )
                response.raise_for_status()
                return response.json().get("text", "").strip()
        except Exception as exc:
            log.error("Transcription failed: %s", exc)
            return ""

    def _identify_speaker_sync(self, wav_data: bytes) -> dict:
        """POST audio to speaker-id server and return result (synchronous)."""
        try:
            with httpx.Client(timeout=self.config.speaker_id_timeout) as client:
                response = client.post(
                    self.config.speaker_id_url,
                    files={"file": ("audio.wav", wav_data, "audio/wav")},
                    data={"threshold": str(self.config.speaker_id_threshold)},
                )
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            log.warning("Speaker ID failed: %s", exc)
            return {"speaker": None, "access_level": "basic", "confidence": 0}

    def _write_transcript_log(
        self,
        text: str,
        speaker: str,
        access_level: str,
        confidence: float,
    ) -> None:
        """Append transcript to JSONL log file (if configured)."""
        if not self.config.transcript_log:
            return
        try:
            entry = {
                "ts": time.time(),
                "speaker": speaker,
                "access_level": access_level,
                "confidence": round(confidence, 3),
                "text": text,
                "posted": False,
            }
            self.config.transcript_log.parent.mkdir(parents=True, exist_ok=True)
            with self.config.transcript_log.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            log.warning("Failed to write transcript log: %s", exc)


async def run_stt_bridge(config: STTConfig) -> None:
    """Start the Wyoming STT bridge server."""
    log.info("Starting Wyoming STT bridge on %s:%d", config.host, config.port)
    log.info("  Whisper URL:    %s", config.whisper_url)
    log.info("  Speaker ID:     %s", "enabled" if config.enable_speaker_id else "disabled")
    if config.enable_speaker_id:
        log.info("  Speaker ID URL: %s", config.speaker_id_url)
    if config.transcript_log:
        log.info("  Transcript log: %s", config.transcript_log)

    server = AsyncServer.from_uri(f"tcp://{config.host}:{config.port}")
    handler_factory = partial(WhisperEventHandler, config=config)
    await server.run(handler_factory)
