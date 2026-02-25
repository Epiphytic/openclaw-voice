"""
Voice Pipeline — orchestrates STT → LLM → TTS for Discord voice.

Accepts raw PCM utterance bytes from a user, runs them through:
  1. WhisperFacade (STT) — transcript text
  2. Speaker ID endpoint (optional, parallel) — speaker name
  3. LLM (OpenAI-compatible endpoint) — response text, using conversation history
  4. KokoroFacade (TTS) — response audio bytes

Maintains per-channel conversation history (last N turns).
All external calls use facades or httpx directly (never the openai package).

Usage::

    config = PipelineConfig(
        whisper_url="http://localhost:8001/inference",
        kokoro_url="http://localhost:8002/v1/audio/speech",
        llm_url="http://localhost:8000/v1/chat/completions",
    )
    pipeline = VoicePipeline(config, channel_id="guild-123-channel-456")
    response_text, response_audio = pipeline.process_utterance(pcm_bytes, user_id="123456")
"""

from __future__ import annotations

import io
import logging
import time
import wave
from collections import deque
from dataclasses import dataclass

import httpx

from openclaw_voice.facades.kokoro import KokoroFacade
from openclaw_voice.facades.whisper import WhisperFacade

log = logging.getLogger("openclaw_voice.voice_pipeline")

DEFAULT_SYSTEM_PROMPT = (
    "You are a voice assistant in a Discord voice channel. "
    "Keep responses SHORT — 1-2 sentences max. They will be spoken aloud. "
    "Never use lists, bullet points, or markdown. Be natural and conversational. "
    "If asked a complex question, give a brief answer and offer to elaborate."
)


@dataclass
class PipelineConfig:
    """Configuration for VoicePipeline.

    All URL fields default to localhost service addresses matching the
    standard openclaw-voice deployment.
    """

    whisper_url: str = "http://localhost:8001/inference"
    kokoro_url: str = "http://localhost:8002/v1/audio/speech"
    speaker_id_url: str = "http://localhost:8003/identify"
    llm_url: str = "http://localhost:8000/v1/chat/completions"

    # LLM settings
    llm_model: str = "Qwen/Qwen2.5-32B-Instruct"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    llm_timeout: float = 60.0
    llm_temperature: float = 0.7
    llm_max_tokens: int = 150

    # TTS settings
    tts_voice: str = "af_heart"

    # Context window
    max_history_turns: int = 20  # each turn = one user + one assistant message pair

    # Optional features
    enable_speaker_id: bool = False
    speaker_id_timeout: float = 5.0
    whisper_timeout: float = 30.0
    kokoro_timeout: float = 60.0


class VoicePipeline:
    """Orchestrates the full voice → text → LLM → voice pipeline.

    Maintains per-channel conversation history so context is preserved
    across multiple exchanges in the same voice session.

    Args:
        config:     Pipeline configuration dataclass.
        channel_id: Unique identifier for this voice channel (used to
                    namespace history). Defaults to "default".
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        channel_id: str = "default",
    ) -> None:
        self._config = config or PipelineConfig()
        self._channel_id = channel_id

        self._whisper = WhisperFacade(
            url=self._config.whisper_url,
            timeout=self._config.whisper_timeout,
        )
        self._kokoro = KokoroFacade(
            url=self._config.kokoro_url,
            timeout=self._config.kokoro_timeout,
        )

        # history: list of {"role": ..., "content": ...} dicts
        # max size = max_history_turns * 2 (user + assistant per turn)
        self._history: deque[dict[str, str]] = deque(maxlen=self._config.max_history_turns * 2)

        log.info(
            "VoicePipeline initialised",
            extra={
                "channel_id": channel_id,
                "llm_model": self._config.llm_model,
                "tts_voice": self._config.tts_voice,
                "max_history_turns": self._config.max_history_turns,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_utterance(
        self,
        audio_bytes: bytes,
        user_id: str,
    ) -> tuple[str, str, bytes]:
        """Process a complete utterance through the full pipeline.

        Args:
            audio_bytes: Raw 16kHz mono int16 PCM bytes (no WAV header).
            user_id:     Discord user ID string — used as speaker hint.

        Returns:
            Tuple of (transcript, response_text, response_audio_wav_bytes).
            ``transcript`` is the STT result; empty string on STT failure.
            ``response_text`` is the LLM reply; empty on failure.
            ``response_audio_wav_bytes`` is the TTS WAV; empty on failure.
            All three are empty on any unhandled error.
        """
        try:
            return self._run_pipeline(audio_bytes, user_id)
        except Exception as exc:
            log.exception(
                "Unhandled error in voice pipeline",
                extra={"user_id": user_id, "error": str(exc)},
            )
            return "", "", b""

    def clear_history(self) -> None:
        """Clear conversation history for this channel."""
        self._history.clear()
        log.info("Conversation history cleared", extra={"channel_id": self._channel_id})

    @property
    def history(self) -> list[dict[str, str]]:
        """Read-only snapshot of the current conversation history."""
        return list(self._history)

    @property
    def config(self) -> PipelineConfig:
        """Pipeline configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        audio_bytes: bytes,
        user_id: str,
    ) -> tuple[str, str, bytes]:
        """Internal pipeline execution. Exceptions propagate to caller.

        Returns:
            Tuple of (transcript, response_text, response_audio_wav_bytes).
        """
        t_start = time.monotonic()

        # Step 1: Convert PCM → WAV for whisper.cpp
        wav_bytes = _pcm_to_wav(audio_bytes)

        # Step 2: Transcribe via whisper.cpp
        t_stt_start = time.monotonic()
        transcript = self._whisper.transcribe(wav_bytes)
        stt_ms = int((time.monotonic() - t_stt_start) * 1000)

        if not transcript:
            log.warning(
                "Empty transcript from whisper, skipping",
                extra={"user_id": user_id, "stt_ms": stt_ms},
            )
            return "", "", b""

        log.info(
            "Transcribed utterance",
            extra={"user_id": user_id, "transcript": transcript, "stt_ms": stt_ms},
        )

        # Step 3: Speaker identification (optional, best-effort)
        speaker_name: str | None = None
        if self._config.enable_speaker_id:
            speaker_name = self._identify_speaker(audio_bytes, user_id)

        # Step 4: Build user message with speaker context
        user_content = self._build_user_message(transcript, speaker_name, user_id)

        # Step 5: Append to history and call LLM
        self._history.append({"role": "user", "content": user_content})
        t_llm_start = time.monotonic()
        response_text = self._call_llm()
        llm_ms = int((time.monotonic() - t_llm_start) * 1000)

        if not response_text:
            log.warning(
                "Empty LLM response",
                extra={"user_id": user_id, "llm_ms": llm_ms},
            )
            # Roll back the user message since we have no response
            self._history.pop()
            return transcript, "", b""

        # Step 6: Append assistant response to history
        self._history.append({"role": "assistant", "content": response_text})
        log.info(
            "LLM response generated",
            extra={
                "user_id": user_id,
                "response_preview": response_text[:100],
                "llm_ms": llm_ms,
            },
        )

        # Step 7: Synthesise audio via Kokoro
        t_tts_start = time.monotonic()
        response_audio = self._synthesize(response_text)
        tts_ms = int((time.monotonic() - t_tts_start) * 1000)

        total_ms = int((time.monotonic() - t_start) * 1000)
        log.info(
            "Pipeline complete",
            extra={
                "user_id": user_id,
                "stt_ms": stt_ms,
                "llm_ms": llm_ms,
                "tts_ms": tts_ms,
                "total_ms": total_ms,
            },
        )

        return transcript, response_text, response_audio

    def _identify_speaker(self, audio_bytes: bytes, fallback_user_id: str) -> str | None:
        """Call the speaker ID endpoint. Returns speaker name or None."""
        try:
            wav_bytes = _pcm_to_wav(audio_bytes)
            with httpx.Client(timeout=self._config.speaker_id_timeout) as client:
                resp = client.post(
                    self._config.speaker_id_url,
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                )
                resp.raise_for_status()
                data = resp.json()
                name = data.get("speaker") or data.get("name")
                if name:
                    log.debug(
                        "Speaker identified",
                        extra={"speaker": name, "user_id": fallback_user_id},
                    )
                    return str(name)
        except Exception as exc:
            log.debug(
                "Speaker ID failed (non-fatal): %s",
                exc,
                extra={"user_id": fallback_user_id},
            )
        return None

    def _build_user_message(
        self,
        transcript: str,
        speaker_name: str | None,
        user_id: str,
    ) -> str:
        """Build the user message content, optionally prefixed with speaker name."""
        if speaker_name:
            return f"[{speaker_name}]: {transcript}"
        return transcript

    def _call_llm(self) -> str:
        """Call the LLM with current conversation history. Returns response text."""
        messages = [{"role": "system", "content": self._config.system_prompt}]
        messages.extend(self._history)

        payload = {
            "model": self._config.llm_model,
            "messages": messages,
            "temperature": self._config.llm_temperature,
            "max_tokens": self._config.llm_max_tokens,
        }

        try:
            with httpx.Client(timeout=self._config.llm_timeout) as client:
                resp = client.post(
                    self._config.llm_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    log.error("LLM returned no choices: %s", data)
                    return ""
                return choices[0].get("message", {}).get("content", "").strip()

        except httpx.HTTPStatusError as exc:
            log.error(
                "LLM HTTP error %d: %s",
                exc.response.status_code,
                exc.response.text[:300],
                extra={"llm_url": self._config.llm_url},
            )
        except httpx.RequestError as exc:
            log.error(
                "LLM connection error: %s",
                exc,
                extra={"llm_url": self._config.llm_url},
            )
        except Exception as exc:
            log.exception("Unexpected LLM error: %s", exc)

        return ""

    def _synthesize(self, text: str) -> bytes:
        """Synthesise text to WAV audio via Kokoro. Returns WAV bytes or b""."""
        import asyncio

        chunks: list[bytes] = []
        try:

            async def _collect() -> None:
                async for chunk in self._kokoro.stream_audio(
                    text,
                    self._config.tts_voice,
                    response_format="wav",
                ):
                    chunks.append(chunk)

            # Run async Kokoro in a new event loop (we're in a sync context here)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in an event loop (e.g. discord.py async context)
                    # schedule as a coroutine — caller must await appropriately
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(asyncio.run, _collect())
                        future.result(timeout=self._config.kokoro_timeout)
                else:
                    loop.run_until_complete(_collect())
            except RuntimeError:
                asyncio.run(_collect())

        except Exception as exc:
            log.error(
                "TTS synthesis failed: %s",
                exc,
                extra={"voice": self._config.tts_voice},
            )
            return b""

        audio = b"".join(chunks)
        log.debug("TTS synthesis complete", extra={"audio_bytes": len(audio)})
        return audio


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def _pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 16_000,
    sample_width: int = 2,
    channels: int = 1,
) -> bytes:
    """Wrap raw PCM bytes in a WAV container.

    Args:
        pcm_bytes:    Raw int16 PCM samples.
        sample_rate:  Samples per second (default 16 kHz).
        sample_width: Bytes per sample (default 2 = int16).
        channels:     Number of channels (default 1 = mono).

    Returns:
        Complete WAV file bytes including header.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()
