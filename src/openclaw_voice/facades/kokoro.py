"""
Facade for Kokoro TTS HTTP endpoint (OpenAI-compatible).

Wraps all direct HTTP calls to the Kokoro TTS server so that the TTS bridge
depends only on this facade — not on ``httpx`` or the Kokoro API shape.
Swapping to a different TTS backend (e.g. Piper, ElevenLabs) only requires
updating this module.

Kokoro TTS endpoint: POST /v1/audio/speech
  JSON body:
    model           — always "kokoro"
    input           — text to synthesise
    voice           — Kokoro voice name or OpenAI alias
    response_format — "wav" | "mp3" | "opus" (default: "wav")
    speed           — speech speed multiplier (0.5–2.0, default: 1.0)

Response: raw audio bytes in the requested format.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx

log = logging.getLogger("openclaw_voice.facades.kokoro")


class KokoroFacade:
    """Facade wrapping Kokoro TTS HTTP synthesis calls.

    Args:
        url:     Full URL of the Kokoro /v1/audio/speech endpoint.
        timeout: HTTP timeout in seconds (TTS can be slow for long texts).
    """

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        self._url = url
        self._timeout = timeout

    def build_payload(
        self,
        text: str,
        voice: str,
        *,
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> dict:
        """Build the JSON payload for a Kokoro synthesis request.

        Args:
            text:            Text to synthesise.
            voice:           Kokoro native voice name.
            response_format: Audio format ("wav", "mp3", "opus").
            speed:           Speech speed (0.5–2.0).

        Returns:
            Dict suitable for JSON-encoding as the request body.
        """
        return {
            "model": "kokoro",
            "input": text.strip(),
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }

    async def stream_audio(
        self,
        text: str,
        voice: str,
        *,
        response_format: str = "wav",
        speed: float = 1.0,
        chunk_size: int = 4096,
    ) -> AsyncIterator[bytes]:
        """Stream audio bytes from Kokoro for the given text and voice.

        Yields raw audio byte chunks as they arrive from the server.
        The caller is responsible for framing (e.g. WAV header handling).

        Args:
            text:            Text to synthesise.
            voice:           Kokoro native voice name.
            response_format: Audio format ("wav", "mp3", "opus").
            speed:           Speech speed multiplier (0.5–2.0).
            chunk_size:      Read chunk size in bytes.

        Yields:
            Raw audio byte chunks.

        Raises:
            httpx.HTTPStatusError: On non-2xx HTTP response.
            httpx.RequestError:    On connection failure.
        """
        payload = self.build_payload(
            text, voice, response_format=response_format, speed=speed
        )

        log.debug(
            "Starting Kokoro synthesis stream",
            extra={
                "url": self._url,
                "voice": voice,
                "text_len": len(text),
                "format": response_format,
                "speed": speed,
            },
        )

        async with (
            httpx.AsyncClient(timeout=self._timeout) as client,
            client.stream("POST", self._url, json=payload) as resp,
        ):
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes(chunk_size=chunk_size):
                yield chunk
