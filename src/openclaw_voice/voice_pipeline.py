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
import re
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

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
    "IMPORTANT: You appear as a SINGLE assistant to the user. "
    "Never mention internal tools, backends, or other agents by name. "
    "Never say things like 'let me check with X' or 'I'll have X do that'. "
    "Just say 'one moment' or 'let me check' and call the tool silently."
    "\n\n"
    "TOOLS — you MUST use them, not just talk about them:\n"
    "- get_weather: weather/forecast questions\n"
    "- get_time: current time/date\n"
    "- web_search: factual questions, current events\n"
    "- escalate: ANYTHING about calendar, email, personal data, "
    "project status, channel activity, code changes, or tasks you cannot do yourself. "
    "If someone asks about ongoing work, what's been done, or what's happening — "
    "ALWAYS call escalate. Never say 'I don't know' or 'I'll check' without calling the tool."
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

    # Post-STT word corrections file (TOML with [corrections] section)
    corrections_file: str = ""
    # Loaded correction map (populated by __post_init__)
    _corrections: dict[str, str] = field(default_factory=dict, repr=False)

    # Text channel context injection into LLM prompts
    channel_context_messages: int = 10

    # Text-to-voice bridge: read aloud messages posted to the linked text channel
    tts_read_channel: bool = True

    def __post_init__(self) -> None:
        """Load corrections from TOML file if configured."""
        if self.corrections_file:
            self._corrections = _load_corrections(self.corrections_file)

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
                "For complex requests, use the escalate tool silently. "
                "Never mention the tool or any backend agent name to the user. "
                "Just say something brief like 'one moment' or 'hang on' and call the tool."
            )
        if identity_parts:
            parts.append("\n\n" + " ".join(identity_parts))

        context_parts = []
        if self.default_location:
            context_parts.append(f"Default location for weather: {self.default_location}.")
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

            # Apply post-STT word corrections
            if self._config._corrections:
                corrected = _apply_corrections(transcript, self._config._corrections)
                if corrected != transcript:
                    log.info(
                        "Applied word corrections",
                        extra={"user_id": user_id, "before": transcript, "after": corrected},
                    )
                    transcript = corrected

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
            ``escalation_request`` is non-None if the LLM called escalate.
        """
        import json as _json

        escalation: str | None = None
        working_messages = list(messages)  # local copy — don't mutate caller's list

        for _round_num in range(max_rounds):
            result = self._call_openai_compat(working_messages, tools=TOOL_DEFINITIONS)

            if isinstance(result, str):
                log.debug("LLM returned text (no tool call): %.100s", result)

                # Quality gate: no tool was called, so ask the local LLM
                # whether the response actually satisfied the user's request.
                # If not, escalate instead of returning a vague/hedging answer.
                if self._should_escalate_response(working_messages, result):
                    log.info(
                        "Quality gate escalation — response judged incomplete: %.60s",
                        result,
                    )
                    user_msg = ""
                    for m in reversed(working_messages):
                        if m["role"] == "user":
                            user_msg = m["content"]
                            break
                    return "", user_msg or "User asked a question that needs escalation"

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

                if fn_name == "escalate":
                    escalation = fn_args.get("request", "")
                    # Return immediately — don't loop back for another LLM
                    # round that would generate chatty hedging text.
                    return "", escalation
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

        Normalizes text for speech before synthesis (strips markdown, expands
        symbols/units, humanizes identifiers).

        Args:
            response_text: Text to synthesise.
            user_id:       Optional user ID for logging.

        Returns:
            WAV audio bytes, or ``b""`` on failure.
        """
        from openclaw_voice.tts_normalizer import normalize_for_tts

        normalized = normalize_for_tts(response_text)
        if normalized != response_text:
            log.debug("TTS normalized: %r → %r", response_text[:80], normalized[:80])

        t_start = time.monotonic()
        audio = self._synthesize(normalized)
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

    def _should_escalate_response(
        self,
        messages: list[dict],
        response: str,
    ) -> bool:
        """Use the local LLM as a quality gate on no-tool-use responses.

        Returns True if the response is judged incomplete/unsatisfying and
        the interaction should be escalated to the main agent.
        """
        # Extract last user message
        user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break

        if not user_msg:
            return False

        # Simple greetings / acknowledgements don't need escalation
        user_lower = user_msg.strip().lower().rstrip("!?.")
        greeting_words = {"hi", "hey", "hello", "thanks", "thank you", "bye", "goodbye", "ok", "okay", "sure", "yes", "no", "yep", "nope"}
        if user_lower in greeting_words:
            return False

        judge_prompt = (
            "You are a quality checker for a voice assistant. A user said something "
            "and the assistant responded WITHOUT using any tools (no web search, no "
            "weather lookup, no escalation). Your job: did the assistant actually "
            "answer the user's question or fulfill their request?\n\n"
            "Reply with ONLY 'COMPLETE' or 'INCOMPLETE'.\n"
            "- COMPLETE: The response directly answers the question or naturally "
            "continues the conversation.\n"
            "- INCOMPLETE: The response hedges, deflects, says it will check, "
            "admits it doesn't know, or fails to answer what was asked.\n\n"
            f"User: {user_msg}\n"
            f"Assistant: {response}\n\n"
            "Verdict:"
        )

        try:
            judge_messages = [{"role": "user", "content": judge_prompt}]
            judge_result = self._call_openai_compat(judge_messages, tools=None)
            if isinstance(judge_result, str):
                verdict = judge_result.strip().upper()
                log.debug("Quality gate verdict: %s (user: %.40s, response: %.40s)",
                          verdict, user_msg, response)
                return "INCOMPLETE" in verdict
        except Exception as exc:
            log.warning("Quality gate LLM call failed, allowing response: %s", exc)

        return False

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
# Word correction helpers
# ---------------------------------------------------------------------------


def _load_corrections(path: str) -> dict[str, str]:
    """Load word corrections from a TOML file.

    Expected format::

        [corrections]
        "wrong" = "right"

    Args:
        path: Path to the TOML corrections file.

    Returns:
        Dict mapping lowercase wrong → right replacement strings.
    """
    p = Path(path)
    if not p.is_file():
        log.warning("Corrections file not found: %s", path)
        return {}
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-redef]
        data = tomllib.loads(p.read_text())
        corrections = data.get("corrections", {})
        # Normalise keys to lowercase for case-insensitive matching
        result = {k.lower(): v for k, v in corrections.items()}
        log.info("Loaded %d word corrections from %s", len(result), path)
        return result
    except Exception as exc:
        log.error("Failed to load corrections from %s: %s", path, exc)
        return {}


def _apply_corrections(text: str, corrections: dict[str, str]) -> str:
    """Apply word-boundary-aware corrections to transcribed text.

    Matching is case-insensitive. The replacement preserves no surrounding
    case context — it uses the replacement value as-is (since corrections
    are deliberate mappings like "Bell" → "Bel").

    Args:
        text:        Input transcript text.
        corrections: Dict of lowercase_wrong → right.

    Returns:
        Corrected text.
    """
    if not corrections:
        return text
    for wrong, right in corrections.items():
        # Word-boundary-aware, case-insensitive replacement
        pattern = re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE)
        text = pattern.sub(right, text)
    return text


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
