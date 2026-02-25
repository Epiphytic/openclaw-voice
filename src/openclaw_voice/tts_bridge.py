"""
Wyoming TTS Bridge — Kokoro backend (OpenAI-compatible).

Bridges Wyoming TTS events (from Home Assistant) to a Kokoro TTS server
exposing the OpenAI-compatible ``/v1/audio/speech`` endpoint.

Wyoming TTS protocol flow:
  1. Client sends Describe  → Bridge returns Info with available voices
  2. Client sends Synthesize → Bridge POSTs to Kokoro, streams audio back
     as AudioChunk events, followed by AudioStop

Kokoro voice names (also accepts OpenAI-compatible aliases):
  af_heart, af_nova, af_sky, am_adam, am_michael, bf_emma
  Aliases: alloy→af_heart, echo→am_adam, fable→bf_emma,
           onyx→am_michael, nova→af_nova, shimmer→af_sky

Usage:
    from openclaw_voice.tts_bridge import run_tts_bridge, TTSConfig

    config = TTSConfig(
        host="0.0.0.0",
        port=10200,
        kokoro_url="http://localhost:8002/v1/audio/speech",
    )
    asyncio.run(run_tts_bridge(config))
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import httpx

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

log = logging.getLogger("openclaw_voice.tts")

# Kokoro sample rate for generated audio
KOKORO_SAMPLE_RATE = 24000
KOKORO_SAMPLE_WIDTH = 2  # 16-bit PCM
KOKORO_CHANNELS = 1

# Chunk size when streaming audio (bytes) — ~20ms at 24kHz 16-bit mono
AUDIO_CHUNK_SIZE = KOKORO_SAMPLE_RATE * KOKORO_SAMPLE_WIDTH * KOKORO_CHANNELS // 50

# OpenAI-name → Kokoro voice mapping (mirrors the Kokoro server)
VOICE_ALIASES: dict[str, str] = {
    "alloy": "af_heart",
    "echo": "am_adam",
    "fable": "bf_emma",
    "onyx": "am_michael",
    "nova": "af_nova",
    "shimmer": "af_sky",
}

# All native Kokoro voices available from the server
NATIVE_VOICES: list[str] = [
    "af_heart",
    "af_nova",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
]


@dataclass
class TTSConfig:
    """Configuration for the Wyoming TTS bridge."""

    # Server binding
    host: str = "0.0.0.0"
    port: int = 10200

    # Kokoro TTS server
    kokoro_url: str = "http://localhost:8002/v1/audio/speech"

    # Default voice (Kokoro native name or OpenAI alias)
    default_voice: str = "af_heart"

    # Audio format to request from Kokoro
    response_format: str = "wav"

    # Speech speed (0.5–2.0)
    speed: float = 1.0

    # HTTP timeout (seconds) — TTS can be slow for long texts
    http_timeout: float = 60.0

    # Additional voices to advertise (beyond defaults)
    extra_voices: list[str] = field(default_factory=list)


def _resolve_voice(name: str) -> str:
    """Resolve an OpenAI alias or pass-through a native Kokoro voice name."""
    return VOICE_ALIASES.get(name, name)


def _all_voice_names(config: TTSConfig) -> list[str]:
    """Return the full list of voices to advertise in Describe responses."""
    names = list(NATIVE_VOICES) + list(VOICE_ALIASES.keys()) + config.extra_voices
    return list(dict.fromkeys(names))  # deduplicate while preserving order


class KokoroEventHandler(AsyncEventHandler):
    """Handles one Wyoming TTS session (one client connection)."""

    def __init__(self, *args: Any, config: TTSConfig, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

    async def handle_event(self, event: Event) -> bool:
        """Dispatch incoming Wyoming TTS events."""

        if Describe.is_type(event.type):
            await self._handle_describe()
            return True

        if Synthesize.is_type(event.type):
            synth = Synthesize.from_event(event)
            await self._handle_synthesize(synth)
            return False  # close connection after synthesis

        return True

    async def _handle_describe(self) -> None:
        """Return Info with available TTS voices."""
        voices = [
            TtsVoice(
                name=name,
                description=f"Kokoro voice: {name}",
                attribution=Attribution(
                    name="Kokoro TTS",
                    url="https://github.com/hexgrad/kokoro",
                ),
                installed=True,
                languages=["en"],
            )
            for name in _all_voice_names(self.config)
        ]
        info = Info(
            tts=[
                TtsProgram(
                    name="kokoro",
                    description="Kokoro TTS (openclaw-voice bridge)",
                    attribution=Attribution(
                        name="Kokoro TTS",
                        url="https://github.com/hexgrad/kokoro",
                    ),
                    installed=True,
                    voices=voices,
                )
            ]
        )
        await self.write_event(info.event())

    async def _handle_synthesize(self, synth: Synthesize) -> None:
        """Synthesize text and stream audio back as Wyoming events."""
        text = synth.text
        if not text or not text.strip():
            log.warning("Empty synthesis request")
            await self.write_event(AudioStop().event())
            return

        # Determine voice
        requested_voice = (
            synth.voice.name if synth.voice and synth.voice.name else self.config.default_voice
        )
        kokoro_voice = _resolve_voice(requested_voice)

        log.info(
            "Synthesizing %d chars with voice=%s (kokoro=%s)",
            len(text),
            requested_voice,
            kokoro_voice,
        )

        payload = {
            "model": "kokoro",
            "input": text.strip(),
            "voice": kokoro_voice,
            "response_format": self.config.response_format,
            "speed": self.config.speed,
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.http_timeout) as client:
                async with client.stream("POST", self.config.kokoro_url, json=payload) as resp:
                    resp.raise_for_status()

                    # Send AudioStart before chunks
                    await self.write_event(
                        AudioStart(
                            rate=KOKORO_SAMPLE_RATE,
                            width=KOKORO_SAMPLE_WIDTH,
                            channels=KOKORO_CHANNELS,
                        ).event()
                    )

                    # Stream audio chunks
                    total_bytes = 0
                    buffer = bytearray()
                    async for chunk in resp.aiter_bytes(chunk_size=4096):
                        buffer.extend(chunk)
                        # Emit complete AUDIO_CHUNK_SIZE blocks
                        while len(buffer) >= AUDIO_CHUNK_SIZE:
                            block = bytes(buffer[:AUDIO_CHUNK_SIZE])
                            buffer = buffer[AUDIO_CHUNK_SIZE:]
                            total_bytes += len(block)
                            await self.write_event(
                                AudioChunk(
                                    audio=block,
                                    rate=KOKORO_SAMPLE_RATE,
                                    width=KOKORO_SAMPLE_WIDTH,
                                    channels=KOKORO_CHANNELS,
                                ).event()
                            )

                    # Flush any remaining bytes
                    if buffer:
                        total_bytes += len(buffer)
                        await self.write_event(
                            AudioChunk(
                                audio=bytes(buffer),
                                rate=KOKORO_SAMPLE_RATE,
                                width=KOKORO_SAMPLE_WIDTH,
                                channels=KOKORO_CHANNELS,
                            ).event()
                        )

                    log.info("Streamed %d bytes of audio for synthesis", total_bytes)

        except httpx.HTTPStatusError as exc:
            log.error(
                "Kokoro returned HTTP %d: %s", exc.response.status_code, exc.response.text[:200]
            )
        except httpx.RequestError as exc:
            log.error("Failed to connect to Kokoro at %s: %s", self.config.kokoro_url, exc)
        except Exception as exc:
            log.exception("Unexpected error during synthesis: %s", exc)
        finally:
            await self.write_event(AudioStop().event())


async def run_tts_bridge(config: TTSConfig) -> None:
    """Start the Wyoming TTS bridge server."""
    log.info("Starting Wyoming TTS bridge on %s:%d", config.host, config.port)
    log.info("  Kokoro URL:    %s", config.kokoro_url)
    log.info("  Default voice: %s", config.default_voice)
    log.info("  Speed:         %.1f", config.speed)

    server = AsyncServer.from_uri(f"tcp://{config.host}:{config.port}")
    handler_factory = partial(KokoroEventHandler, config=config)
    await server.run(handler_factory)
