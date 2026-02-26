"""
VoiceSession — per-guild voice session state container.

Holds all mutable state for one active voice channel session:
  - The ConversationLog (shared source of truth)
  - The system prompt (built at join time with channel context)
  - Worker task handles (STT, LLM, TTS, escalation)
  - Playback queue and cancellation primitives

One VoiceSession is created when the bot joins a voice channel (/join) and
destroyed when it leaves (/leave). All workers hold a reference to the session
and read/write through it.

Usage::

    session = VoiceSession(system_prompt="You are Chip...")
    session.log.append(LogEntry(ts=time.monotonic(), kind="user_speech", ...))
    session.pending_llm_trigger.set()   # wake the LLM worker
"""

from __future__ import annotations

import asyncio
import logging
import time

from openclaw_voice.conversation_log import ConversationLog, LogEntry

log = logging.getLogger("openclaw_voice.voice_session")


class VoiceSession:
    """Per-guild voice session state.

    Attributes:
        log:                  Shared append-only conversation log.
        system_prompt:        System prompt built at join time (includes channel context).
        guild_id:             Discord guild ID this session belongs to.

        stt_task:             asyncio.Task for the STT worker, or None.
        llm_task:             asyncio.Task for the LLM worker, or None.
        tts_task:             asyncio.Task for the TTS worker, or None.
        escalation_tasks:     Dict of active escalation workers keyed by request ID.

        playback_queue:       Async queue of WAV bytes to play in the voice channel.
        tts_cancel:           Event: set when new user speech arrives, cancels TTS.
        pending_llm_trigger:  Event: set when STT produces a new user_speech entry.
        last_llm_version:     Log version when the LLM worker last ran. Used to detect
                              whether new speech arrived while the LLM was busy.
    """

    def __init__(
        self,
        system_prompt: str,
        guild_id: int = 0,
        playback_queue_maxsize: int = 10,
    ) -> None:
        self.guild_id = guild_id
        self.system_prompt = system_prompt

        # Shared conversation log
        self.log: ConversationLog = ConversationLog()

        # Worker task handles
        self.stt_task: asyncio.Task | None = None
        self.llm_task: asyncio.Task | None = None
        self.tts_task: asyncio.Task | None = None
        self.escalation_tasks: dict[str, asyncio.Task] = {}

        # Playback
        self.playback_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=playback_queue_maxsize)
        self.tts_cancel: asyncio.Event = asyncio.Event()
        self.tts_pause: asyncio.Event = asyncio.Event()  # pause playback on noise

        # LLM trigger / dedup
        self.pending_llm_trigger: asyncio.Event = asyncio.Event()
        self.last_llm_version: int = 0

        log.info(
            "VoiceSession created",
            extra={"guild_id": guild_id},
        )

    def signal_new_speech(self) -> None:
        """Called when new user speech arrives.

        Sets the TTS cancel event (stops in-flight TTS) and the LLM trigger
        event (wakes the LLM worker). Does NOT cancel LLM or escalation tasks.
        """
        self.tts_cancel.set()
        self.pending_llm_trigger.set()
        log.debug("VoiceSession: new speech signal sent", extra={"guild_id": self.guild_id})

    def clear_tts_cancel(self) -> None:
        """Clear the TTS cancel flag once the TTS worker has responded to it."""
        self.tts_cancel.clear()

    def is_active(self) -> bool:
        """Return True if any worker is running."""
        tasks = [self.stt_task, self.llm_task, self.tts_task]
        return any(t is not None and not t.done() for t in tasks)

    async def stop_workers(self) -> None:
        """Cancel all workers and wait for them to finish.

        Called on /leave. Does not disconnect from Discord — caller handles that.
        """
        tasks_to_cancel: list[asyncio.Task] = []

        for t in (self.stt_task, self.llm_task, self.tts_task):
            if t is not None and not t.done():
                tasks_to_cancel.append(t)

        for t in list(self.escalation_tasks.values()):
            if not t.done():
                tasks_to_cancel.append(t)

        for t in tasks_to_cancel:
            t.cancel()

        # Wait for all cancellations to complete
        import contextlib

        for t in tasks_to_cancel:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t

        self.stt_task = None
        self.llm_task = None
        self.tts_task = None
        self.escalation_tasks.clear()

        log.info("VoiceSession workers stopped", extra={"guild_id": self.guild_id})

    def inject_system_entry(self, text: str, speaker: str = "system") -> int:
        """Append a system entry to the log (e.g. channel context, keepalive).

        Returns:
            New log version.
        """
        entry = LogEntry(
            ts=time.monotonic(),
            kind="system",
            speaker=speaker,
            text=text,
        )
        return self.log.append(entry)

    def __repr__(self) -> str:
        return (
            f"VoiceSession(guild_id={self.guild_id}, log={self.log!r}, active={self.is_active()})"
        )
