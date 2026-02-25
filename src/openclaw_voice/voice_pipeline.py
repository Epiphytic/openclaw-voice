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
from openclaw_voice.tools import TOOL_DEFINITIONS, execute_tool

log = logging.getLogger("openclaw_voice.voice_pipeline")

DEFAULT_SYSTEM_PROMPT = (
    "You are a voice assistant in a Discord voice channel. "
    "Keep responses SHORT — 1-2 sentences max. They will be spoken aloud via TTS. "
    "Never use lists, bullet points, markdown, or special characters. "
    "Be natural, warm, and conversational — like a helpful friend. "
    "\n\n"
    "TOOLS — you MUST use them, not just talk about them:\n"
    "- get_weather: weather/forecast questions\n"
    "- get_time: current time/date\n"
    "- web_search: factual questions, current events\n"
    "- escalate_to_bel: ANYTHING about the main agent, calendar, email, "
    "personal data, project status, channel activity, or tasks you cannot do yourself. "
    "If someone asks about what the main agent is doing, working on, or has said — "
    "ALWAYS call escalate_to_bel. Never say 'I don't know' or 'I'll check' without calling the tool."
    "\n/no_think"
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
    llm_api_key: str | None = None
    llm_timeout: float = 60.0
    llm_temperature: float = 0.7
    llm_max_tokens: int = 300

    # TTS settings
    tts_voice: str = "af_heart"

    # Context window
    max_history_turns: int = 20  # each turn = one user + one assistant message pair

    # Optional features
    enable_speaker_id: bool = False
    speaker_id_timeout: float = 5.0
    whisper_timeout: float = 30.0
    kokoro_timeout: float = 60.0

    # Whisper vocabulary hint (helps with unusual names/terms)
    whisper_prompt: str = ""

    # Identity & context (loaded from config file, not hardcoded)
    bot_name: str = "Assistant"
    main_agent_name: str = "main agent"
    default_location: str = ""  # e.g. "Cassidy, BC, Canada"
    default_timezone: str = "UTC"
    extra_context: str = ""  # freeform context appended to system prompt

    def build_system_prompt(self) -> str:
        """Build the full system prompt from template + config context."""
        parts = [self.system_prompt]

        identity_parts = []
        if self.bot_name and self.bot_name != "Assistant":
            identity_parts.append(f"Your name is {self.bot_name}.")
        if self.main_agent_name and self.main_agent_name != "main agent":
            identity_parts.append(
                f"You work alongside {self.main_agent_name} (the main AI agent). "
                f"When a request needs {self.main_agent_name}, you MUST call the escalate_to_bel tool — "
                f"do NOT just say you'll check, actually invoke the tool."
            )
        if identity_parts:
            parts.append("\n\n" + " ".join(identity_parts))

        context_parts = []
        if self.default_location:
            context_parts.append(
                f"Default location for weather: {self.default_location}."
            )
        if self.default_timezone and self.default_timezone != "UTC":
            context_parts.append(f"Default timezone: {self.default_timezone}.")
        if self.extra_context:
            context_parts.append(self.extra_context)
        if context_parts:
            parts.append("\n\n" + " ".join(context_parts))

        return "".join(parts)


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

        # Build the full system prompt from config (identity + context)
        self._system_prompt = self._config.build_system_prompt()

        # history: list of {"role": ..., "content": ...} dicts
        # max size = max_history_turns * 2 (user + assistant per turn)
        self._history: deque[dict[str, str]] = deque(maxlen=self._config.max_history_turns * 2)

        # Set after process_utterance if the LLM requested escalation to Bel
        self._last_escalation: str | None = None

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
    ) -> tuple[str, str]:
        """Process a complete utterance through STT → LLM (with tools).

        Args:
            audio_bytes: Raw 16kHz mono int16 PCM bytes (no WAV header).
            user_id:     Discord user ID string — used as speaker hint.

        Returns:
            Tuple of (transcript, response_text).
            ``transcript`` is the STT result; empty string on STT failure.
            ``response_text`` is the LLM reply; empty on failure.
            Check ``last_escalation`` after calling to see if Bel was invoked.
        """
        self._last_escalation = None
        try:
            return self._run_pipeline(audio_bytes, user_id)
        except Exception as exc:
            log.exception(
                "Unhandled error in voice pipeline",
                extra={"user_id": user_id, "error": str(exc)},
            )
            return "", ""

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
        transcript = self._whisper.transcribe(
            wav_bytes,
            initial_prompt=self._config.whisper_prompt or None,
        )
        stt_ms = int((time.monotonic() - t_stt_start) * 1000)

        if not transcript:
            log.warning(
                "Empty transcript from whisper, skipping",
                extra={"user_id": user_id, "stt_ms": stt_ms},
            )
            return "", ""

        # Filter Whisper hallucinations — known phantom phrases generated
        # from near-silent audio or background noise.
        _HALLUCINATIONS = {
            "thank you", "thanks", "thank you.", "thanks.",
            "thanks for watching", "thanks for watching.",
            "thank you for watching", "thank you for watching.",
            "bye", "bye.", "goodbye", "goodbye.",
            "you", "you.", "the end", "the end.",
            "so", "so.", "hmm", "hmm.",
        }
        # Check each line — Whisper sometimes repeats hallucinations
        # across a long audio segment ("Thank you.\n Thank you.\n Thank you.")
        lines = [ln.strip().lower() for ln in transcript.strip().splitlines() if ln.strip()]
        if lines and all(ln in _HALLUCINATIONS for ln in lines):
            log.info(
                "Filtered Whisper hallucination",
                extra={"user_id": user_id, "transcript": transcript, "stt_ms": stt_ms},
            )
            return "", ""

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

        # Step 5: Append to history and call LLM (with tool loop)
        self._history.append({"role": "user", "content": user_content})
        t_llm_start = time.monotonic()
        response_text, escalation = self._call_llm_with_tools()
        llm_ms = int((time.monotonic() - t_llm_start) * 1000)

        # Strip Qwen3 think tags (generated even in /no_think mode)
        if response_text and "<think>" in response_text:
            import re
            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

        if not response_text and not escalation:
            log.warning(
                "Empty LLM response",
                extra={"user_id": user_id, "llm_ms": llm_ms},
            )
            self._history.pop()
            return transcript, ""

        # Store escalation request for the caller to retrieve
        self._last_escalation = escalation

        # Step 6: Append assistant response to history
        if response_text:
            self._history.append({"role": "assistant", "content": response_text})
        log.info(
            "LLM response generated",
            extra={
                "user_id": user_id,
                "response_preview": response_text[:100],
                "llm_ms": llm_ms,
            },
        )

        return transcript, response_text

    def synthesize_response(self, response_text: str, user_id: str = "") -> bytes:
        """Run TTS on a response string. Returns WAV bytes.

        Separated from _run_pipeline so callers can insert a cancellation
        checkpoint between LLM and TTS.
        """
        t_tts_start = time.monotonic()
        response_audio = self._synthesize(response_text)
        tts_ms = int((time.monotonic() - t_tts_start) * 1000)
        log.info(
            "TTS complete",
            extra={"user_id": user_id, "tts_ms": tts_ms},
        )
        return response_audio

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

    @property
    def last_escalation(self) -> str | None:
        """The escalation request from the last process_utterance call, if any."""
        return self._last_escalation

    def _call_llm_with_tools(self, max_rounds: int = 3) -> tuple[str, str | None]:
        """Call the LLM with tool support, looping on tool calls.

        Returns:
            (response_text, escalation_request)
            escalation_request is non-None if the LLM called escalate_to_bel.
        """
        import json as _json

        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._history)

        escalation: str | None = None

        for round_num in range(max_rounds):
            # Call LLM with tools
            result = self._call_openai_compat(messages, tools=TOOL_DEFINITIONS)

            # result is now either a string (direct text) or a dict with tool_calls
            if isinstance(result, str):
                log.debug("LLM returned text (no tool call): %.100s", result)

                # Fallback: if the LLM talks about checking/looking into things
                # without actually calling a tool, force an escalation.
                text_lower = result.lower()
                fallback_triggers = [
                    "let me check",
                    "let me find",
                    "let me look",
                    "let me search",
                    "i'll check",
                    "i'll look",
                    "i'll find",
                    "i'll search",
                    "i'm checking",
                    "i'm looking",
                    "i'm searching",
                    "don't have access",
                    "don't have that information",
                    "i'm not sure what",
                    "i don't know what",
                    "checking the channel",
                    "checking with",
                ]
                if any(phrase in text_lower for phrase in fallback_triggers):
                    log.info(
                        "Fallback escalation triggered — LLM hedged without tool call: %.60s",
                        result,
                    )
                    # Reconstruct the user's original request from history
                    user_msg = ""
                    for m in reversed(self._history):
                        if m["role"] == "user":
                            user_msg = m["content"]
                            break
                    return result, user_msg or "User asked a question that needs escalation"

                return result, escalation

            log.info("LLM returned tool_calls: %s", [tc["function"]["name"] for tc in result.get("tool_calls", [])])

            # Handle tool calls
            tool_calls = result.get("tool_calls", [])
            assistant_msg = result.get("message", {})
            messages.append(assistant_msg)

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = _json.loads(tc["function"]["arguments"])
                except (ValueError, KeyError):
                    fn_args = {}

                log.info("Tool call: %s(%s)", fn_name, fn_args)

                if fn_name == "escalate_to_bel":
                    escalation = fn_args.get("request", "")
                    # Give the LLM the escalation acknowledgment so it
                    # can produce an interim spoken response like
                    # "Let me ask Bel about that."
                    tool_result = (
                        "Escalation sent to Bel. Tell the user you're checking "
                        "with Bel and will have an answer shortly."
                    )
                else:
                    tool_result = execute_tool(fn_name, fn_args) or f"Unknown tool: {fn_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(tool_result),
                })

            # Loop back to get the final text response incorporating tool results

        # Fell through max rounds — shouldn't happen
        log.warning("Tool loop exhausted after %d rounds", max_rounds)
        return "", escalation

    def _call_llm(self) -> str:
        """Call the LLM with current conversation history. Returns response text.

        If ``llm_url`` starts with ``anthropic://`` the Anthropic Messages API
        is used (model taken from the path, e.g. ``anthropic://claude-3-5-haiku-20241022``).
        The API key is read from ``llm_api_key`` config or the standard
        ``ANTHROPIC_API_KEY`` env var.

        Otherwise the OpenAI-compatible ``/v1/chat/completions`` path is used.
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._history)

        if self._config.llm_url.startswith("anthropic://"):
            return self._call_anthropic(messages)

        return self._call_openai_compat(messages)

    def _call_anthropic(self, messages: list[dict]) -> str:
        """Call Anthropic Messages API via the official SDK."""
        try:
            import anthropic
        except ImportError:
            log.error("anthropic package not installed — pip install anthropic")
            return ""

        # Extract system prompt (first message) and user/assistant messages
        system_text = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                chat_messages.append(m)

        model = self._config.llm_url.removeprefix("anthropic://")
        api_key = getattr(self._config, "llm_api_key", None) or None

        try:
            client = anthropic.Anthropic(api_key=api_key, timeout=self._config.llm_timeout)
            resp = client.messages.create(
                model=model,
                max_tokens=self._config.llm_max_tokens,
                system=system_text,
                messages=chat_messages,
                temperature=self._config.llm_temperature,
            )
            text = resp.content[0].text if resp.content else ""
            return text.strip()
        except Exception as exc:
            log.exception("Anthropic LLM error: %s", exc)
            return ""

    def _call_openai_compat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> str | dict:
        """Call OpenAI-compatible /v1/chat/completions endpoint.

        If ``tools`` is provided and the LLM returns tool_calls, returns a
        dict with ``{"message": ..., "tool_calls": [...]}``.
        Otherwise returns the response text as a string.
        """
        payload: dict = {
            "model": self._config.llm_model,
            "messages": messages,
            "temperature": self._config.llm_temperature,
            "max_tokens": self._config.llm_max_tokens,
        }
        if tools:
            payload["tools"] = tools

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

                message = choices[0].get("message", {})
                tool_calls = message.get("tool_calls")

                if tool_calls:
                    return {"message": message, "tool_calls": tool_calls}

                return (message.get("content") or "").strip()

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
