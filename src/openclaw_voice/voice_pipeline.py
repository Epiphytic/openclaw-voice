"""
Voice Pipeline — STT, LLM, and TTS helpers for Discord voice.

In v2 architecture the ConversationLog (see conversation_log.py) is the single
source of truth for conversation history.  VoicePipeline no longer maintains
its own ``_history`` deque; instead callers pass the message list directly
to ``call_llm_with_tools()``.

Public API:
  - ``run_stt(audio_bytes)``          — WAV + Whisper → transcript (str)
  - ``call_llm_with_tools(messages)`` — LLM tool-loop → (text, escalation)
  - ``synthesize_response(text)``     — TTS → WAV bytes
  - ``build_system_prompt()``         — build system prompt from config
  - ``is_hallucination(transcript)``  — hallucination filter predicate

Helpers preserved from v1 (used by discord_bot.py):
  - ``_call_openai_compat()``         — raw OpenAI-compat HTTP call
  - ``_rephrase_for_speech()``        — LLM rephrase for natural TTS

Config integration unchanged — PipelineConfig and build_system_prompt()
still drive all identity, voice, and endpoint settings.

Usage::

    config = PipelineConfig(...)
    pipeline = VoicePipeline(config=config)

    # STT
    transcript = pipeline.run_stt(pcm_bytes)

    # LLM with conversation history from ConversationLog
    messages = [{"role": "system", "content": system_prompt}]
    messages += log.to_messages()
    text, escalation = pipeline.call_llm_with_tools(messages)

    # TTS
    wav_bytes = pipeline.synthesize_response(text)
"""

from __future__ import annotations

import io
import logging
import time
import wave
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

# Whisper hallucinations — known phantom phrases generated from near-silent audio
# or background noise. Matched against each lowercased line of the transcript.
_HALLUCINATIONS: frozenset[str] = frozenset(
    {
        "thank you",
        "thanks",
        "thank you.",
        "thanks.",
        "thanks for watching",
        "thanks for watching.",
        "thank you for watching",
        "thank you for watching.",
        "bye",
        "bye.",
        "goodbye",
        "goodbye.",
        "you",
        "you.",
        "the end",
        "the end.",
        "so",
        "so.",
        "hmm",
        "hmm.",
    }
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
    llm_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    llm_api_key: str | None = None
    llm_timeout: float = 60.0
    llm_temperature: float = 0.7
    llm_max_tokens: int = 300

    # TTS settings
    tts_voice: str = "af_heart"

    # Context window — kept for reference; log truncation handled by caller
    max_history_turns: int = 20

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
    default_location: str = ""  # e.g. "Seattle, WA, USA"
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
    """Stateless STT/LLM/TTS helper.

    In v2 architecture VoicePipeline no longer maintains internal conversation
    history.  All history is owned by ``ConversationLog``; callers pass the
    message list to ``call_llm_with_tools()``.

    This class is intentionally kept synchronous so it can be called from
    asyncio via ``loop.run_in_executor()``, matching the original v1 pattern.

    Args:
        config:     Pipeline configuration dataclass.
        channel_id: Identifier for this voice channel (used in log messages).
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

        log.info(
            "VoicePipeline initialised",
            extra={
                "channel_id": channel_id,
                "llm_model": self._config.llm_model,
                "tts_voice": self._config.tts_voice,
            },
        )

    # ------------------------------------------------------------------
    # Public API (v2)
    # ------------------------------------------------------------------

    def run_stt(self, audio_bytes: bytes, user_id: str = "") -> str:
        """Run Whisper STT on raw PCM bytes.

        Wraps PCM in a WAV container, sends to whisper.cpp, and returns the
        transcript. Returns empty string on failure or hallucination.

        Args:
            audio_bytes: Raw 16kHz mono int16 PCM bytes (no WAV header).
            user_id:     Optional Discord user ID for logging.

        Returns:
            Transcript string, or "" if empty/hallucination/error.
        """
        try:
            t_start = time.monotonic()
            wav_bytes = _pcm_to_wav(audio_bytes)
            transcript = self._whisper.transcribe(
                wav_bytes,
                initial_prompt=self._config.whisper_prompt or None,
            )
            stt_ms = int((time.monotonic() - t_start) * 1000)

            if not transcript:
                log.warning(
                    "Empty transcript from Whisper",
                    extra={"user_id": user_id, "stt_ms": stt_ms},
                )
                return ""

            if self.is_hallucination(transcript):
                log.info(
                    "Filtered Whisper hallucination",
                    extra={"user_id": user_id, "transcript": transcript, "stt_ms": stt_ms},
                )
                return ""

            log.info(
                "Transcribed utterance",
                extra={"user_id": user_id, "transcript": transcript, "stt_ms": stt_ms},
            )
            return transcript

        except Exception as exc:
            log.exception(
                "STT error",
                extra={"user_id": user_id, "error": str(exc)},
            )
            return ""

    @staticmethod
    def is_hallucination(transcript: str) -> bool:
        """Return True if the transcript is a known Whisper hallucination.

        Checks each (non-empty) line of the transcript against the hallucination
        set. Returns True only if ALL lines match — a single real word rescues
        the entire transcript.

        Args:
            transcript: Raw Whisper output string.

        Returns:
            True if the transcript should be discarded as a hallucination.
        """
        lines = [ln.strip().lower() for ln in transcript.strip().splitlines() if ln.strip()]
        if not lines:
            return True  # empty = discard
        return all(ln in _HALLUCINATIONS for ln in lines)

    def call_llm_with_tools(
        self,
        messages: list[dict],
        max_rounds: int = 3,
    ) -> tuple[str, str | None]:
        """Call the LLM with tool support, looping on tool calls.

        The caller is responsible for building the message list (system prompt +
        conversation history from ConversationLog.to_messages() + current user
        message if not yet in the log).

        Args:
            messages:   Full message list: [system, ...history..., user_msg].
            max_rounds: Maximum tool-call rounds before giving up.

        Returns:
            Tuple of ``(response_text, escalation_request)``.
            ``escalation_request`` is non-None if the LLM called escalate_to_bel.
        """
        import json as _json

        escalation: str | None = None
        working_messages = list(messages)  # local copy — don't mutate caller's list

        for round_num in range(max_rounds):
            result = self._call_openai_compat(working_messages, tools=TOOL_DEFINITIONS)

            if isinstance(result, str):
                log.debug("LLM returned text (no tool call): %.100s", result)

                # Fallback: if the LLM hedges without calling a tool, force escalation
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
                    # Extract the most recent user message for the escalation request
                    user_msg = ""
                    for m in reversed(working_messages):
                        if m["role"] == "user":
                            user_msg = m["content"]
                            break
                    return result, user_msg or "User asked a question that needs escalation"

                return result, escalation

            log.info(
                "LLM returned tool_calls: %s",
                [tc["function"]["name"] for tc in result.get("tool_calls", [])],
            )

            tool_calls = result.get("tool_calls", [])
            assistant_msg = result.get("message", {})
            working_messages.append(assistant_msg)

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = _json.loads(tc["function"]["arguments"])
                except (ValueError, KeyError):
                    fn_args = {}

                log.info("Tool call: %s(%s)", fn_name, fn_args)

                if fn_name == "escalate_to_bel":
                    escalation = fn_args.get("request", "")
                    tool_result = (
                        "Escalation sent to Bel. Tell the user you're checking "
                        "with Bel and will have an answer shortly."
                    )
                else:
                    tool_result = execute_tool(fn_name, fn_args) or f"Unknown tool: {fn_name}"

                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(tool_result),
                    }
                )

            # Loop back to get the final text response incorporating tool results

        log.warning("Tool loop exhausted after %d rounds", max_rounds)
        return "", escalation

    def synthesize_response(self, response_text: str, user_id: str = "") -> bytes:
        """Run TTS on a response string. Returns WAV bytes (empty on failure).

        Args:
            response_text: Text to synthesise.
            user_id:       Optional user ID for logging.

        Returns:
            WAV audio bytes, or ``b""`` on failure.
        """
        t_start = time.monotonic()
        audio = self._synthesize(response_text)
        tts_ms = int((time.monotonic() - t_start) * 1000)
        log.info(
            "TTS complete",
            extra={"user_id": user_id, "tts_ms": tts_ms, "audio_bytes": len(audio)},
        )
        return audio

    def build_system_prompt(self) -> str:
        """Return the pre-built system prompt (config-driven)."""
        return self._system_prompt

    @property
    def config(self) -> PipelineConfig:
        """Pipeline configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Backward-compat helpers used by discord_bot.py
    # ------------------------------------------------------------------

    def rephrase_for_speech(self, bel_response: str) -> str:
        """Use the LLM to rephrase the main agent's response for natural speech.

        Kept as a pipeline method since it needs LLM access and config.

        Args:
            bel_response: Raw text from the main agent.

        Returns:
            Natural speech-ready rephrasing (1-2 sentences).
        """
        agent_name = self._config.main_agent_name
        bot_name = self._config.bot_name
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {bot_name}. {agent_name} (the main AI agent) sent you a response to relay "
                    f"to the user via voice. Rephrase it naturally for speech — short, "
                    f"warm, conversational. 1-2 sentences max. Don't mention {agent_name} — "
                    f"just deliver the information naturally. /no_think"
                ),
            },
            {
                "role": "user",
                "content": f"{agent_name}'s response to relay: {bel_response}",
            },
        ]
        result = self._call_openai_compat(messages)
        if isinstance(result, str) and result:
            import re
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            return result
        return bel_response  # fallback: use raw response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_openai_compat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> str | dict:
        """Call OpenAI-compatible /v1/chat/completions endpoint.

        If ``tools`` is provided and the LLM returns tool_calls, returns a
        dict ``{"message": ..., "tool_calls": [...]}``.
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

    def _synthesize(self, text: str) -> bytes:
        """Synthesise text to WAV audio via Kokoro. Returns WAV bytes or b""."""
        import asyncio
        import concurrent.futures

        chunks: list[bytes] = []

        async def _collect() -> None:
            async for chunk in self._kokoro.stream_audio(
                text,
                self._config.tts_voice,
                response_format="wav",
            ):
                chunks.append(chunk)

        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
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
