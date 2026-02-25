"""
Facade for whisper.cpp HTTP inference endpoint.

Wraps all direct HTTP calls to the whisper.cpp server so that the STT bridge
depends only on this facade — not on ``httpx`` or the whisper.cpp API shape.
Swapping to a different backend (e.g. faster-whisper, OpenAI API) only
requires updating this module.

Typical whisper.cpp server endpoint: POST /inference
  Form fields:
    file            — WAV audio file
    response_format — "json" (default) or "text"
    language        — optional ISO-639-1 language hint (e.g. "en")

Response JSON: {"text": "transcribed text"}
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger("openclaw_voice.facades.whisper")


class WhisperFacade:
    """Facade wrapping whisper.cpp HTTP inference calls.

    Args:
        url:     Full URL of the whisper.cpp /inference endpoint.
        timeout: HTTP timeout in seconds.
    """

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self._url = url
        self._timeout = timeout

    def transcribe(
        self,
        wav_data: bytes,
        *,
        language: str | None = None,
        response_format: str = "json",
    ) -> str:
        """POST WAV audio to whisper.cpp and return the transcript text.

        Args:
            wav_data:        Raw WAV file bytes.
            language:        Optional ISO-639-1 language hint.
            response_format: "json" (default) or "text".

        Returns:
            Stripped transcript text, or empty string on failure.
        """
        form_data: dict[str, str] = {"response_format": response_format}
        if language:
            form_data["language"] = language

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    self._url,
                    files={"file": ("audio.wav", wav_data, "audio/wav")},
                    data=form_data,
                )
                response.raise_for_status()

                if response_format == "json":
                    return response.json().get("text", "").strip()
                return response.text.strip()

        except httpx.HTTPStatusError as exc:
            log.error(
                "whisper.cpp returned HTTP %d: %s",
                exc.response.status_code,
                exc.response.text[:200],
                extra={"url": self._url, "status_code": exc.response.status_code},
            )
        except httpx.RequestError as exc:
            log.error(
                "Failed to connect to whisper.cpp at %s: %s",
                self._url,
                exc,
                extra={"url": self._url},
            )
        except Exception as exc:
            log.exception(
                "Unexpected error during whisper.cpp transcription: %s",
                exc,
                extra={"url": self._url},
            )

        return ""
