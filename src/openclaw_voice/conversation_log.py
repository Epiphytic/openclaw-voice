"""
ConversationLog — append-only, thread-safe conversation history.

Each entry in the log represents one event in the voice conversation:
  - user_speech: transcribed speech from a human user
  - assistant: spoken response from the bot
  - tool_result: result of a tool call (weather, web search, etc.)
  - escalation_result: response from the main agent (Bel) after escalation
  - system: internal system events (keepalives, context injections)

The log is the single source of truth for conversation state. Workers read
from snapshots and write new entries via append(). The version number increments
on each append, enabling efficient change detection by workers.

Usage::

    log = ConversationLog()
    entry = LogEntry(ts=time.monotonic(), kind="user_speech", speaker="Alice", text="Hey Assistant")
    version = log.append(entry)
    entries, version = log.snapshot()
    messages = log.to_messages()    # OpenAI chat format
    version = await log.wait_for(version)   # async: waits until log advances past version
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("openclaw_voice.conversation_log")

# Valid entry kinds
ENTRY_KINDS = frozenset({"user_speech", "assistant", "tool_result", "escalation_result", "system"})


@dataclass
class LogEntry:
    """A single event in the conversation log.

    Attributes:
        ts:      Monotonic timestamp (time.monotonic()) when the entry was created.
        kind:    Entry type — one of: user_speech, assistant, tool_result,
                 escalation_result, system.
        speaker: Who produced this entry (user display name, bot name, tool name, etc.).
        text:    The content of the entry.
        meta:    Optional metadata dict (stt_ms, llm_ms, tool_name, seq, etc.).
    """

    ts: float
    kind: str
    speaker: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ENTRY_KINDS:
            raise ValueError(
                f"Invalid log entry kind {self.kind!r}. Expected one of: {sorted(ENTRY_KINDS)}"
            )


class ConversationLog:
    """Append-only conversation log. Thread-safe. Async-awaitable.

    Designed for use across the asyncio + threading boundary:
      - ``append()`` is safe to call from any thread (uses a threading lock).
      - ``wait_for()`` is async-native (uses an asyncio.Condition).
      - ``snapshot()`` returns a stable copy — safe to read without holding locks.

    The ``version`` counter starts at 0 (empty). Each ``append()`` increments it
    by 1 and returns the new version. Workers can use ``wait_for(v)`` to block
    until the log contains more entries than version ``v``.
    """

    def __init__(self) -> None:
        self._entries: list[LogEntry] = []
        self._lock = threading.Lock()
        self._version: int = 0

        # asyncio.Condition for async waiting — must be used from the event loop
        # where wait_for() is called. Created lazily to avoid loop-binding issues.
        self._condition: asyncio.Condition | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_condition(self) -> asyncio.Condition:
        """Get or create the asyncio.Condition for this log.

        Called from async context (event loop). First call binds this log to
        the running event loop.
        """
        if self._condition is None:
            self._condition = asyncio.Condition()
            with contextlib.suppress(RuntimeError):
                self._loop = asyncio.get_running_loop()
        return self._condition

    def append(self, entry: LogEntry) -> int:
        """Append an entry to the log.

        Thread-safe. Signals any waiters in the event loop.

        Args:
            entry: The LogEntry to append.

        Returns:
            The new version number (= number of entries in the log).
        """
        with self._lock:
            self._entries.append(entry)
            self._version += 1
            version = self._version

        log.debug(
            "ConversationLog append",
            extra={
                "kind": entry.kind,
                "speaker": entry.speaker,
                "text_preview": entry.text[:80],
                "version": version,
            },
        )

        # Signal async waiters from any thread using call_soon_threadsafe.
        # This is safe even if called from a sync thread.
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._notify_async)

        return version

    def _notify_async(self) -> None:
        """Notify all async waiters — must be called from the event loop."""
        if self._condition is None:
            return

        async def _do_notify() -> None:
            async with self._condition:
                self._condition.notify_all()

        asyncio.ensure_future(_do_notify())

    def snapshot(self) -> tuple[list[LogEntry], int]:
        """Return a stable (entries_copy, version) snapshot.

        The returned list is a shallow copy — safe to iterate without holding
        the lock. LogEntry objects themselves are immutable by convention.

        Returns:
            Tuple of (entries, version).
        """
        with self._lock:
            return list(self._entries), self._version

    def to_messages(self) -> list[dict]:
        """Convert the log to OpenAI chat messages format.

        Maps entry kinds to roles:
          - user_speech   → {"role": "user", "content": "Speaker: text"}
          - assistant     → {"role": "assistant", "content": text}
          - tool_result   → {"role": "system", "content": "[tool_result] speaker: text"}
          - escalation_result → {"role": "system", "content": "[escalation_result] speaker: text"}
          - system        → {"role": "system", "content": text}

        Returns:
            List of message dicts suitable for passing to the LLM.
        """
        entries, _ = self.snapshot()
        messages: list[dict] = []

        for entry in entries:
            if entry.kind == "user_speech":
                messages.append({"role": "user", "content": f"{entry.speaker}: {entry.text}"})
            elif entry.kind == "assistant":
                messages.append({"role": "assistant", "content": entry.text})
            elif entry.kind in ("tool_result", "escalation_result"):
                messages.append(
                    {
                        "role": "system",
                        "content": f"[{entry.kind}] {entry.speaker}: {entry.text}",
                    }
                )
            elif entry.kind == "system":
                messages.append({"role": "system", "content": entry.text})

        return messages

    async def wait_for(self, version: int) -> int:
        """Async: block until the log advances past the given version.

        Binds the log to the running event loop on first call.

        Args:
            version: Wait until log version > this value.

        Returns:
            The new version (guaranteed > version argument).
        """
        condition = self._get_condition()
        async with condition:
            while True:
                with self._lock:
                    current = self._version
                if current > version:
                    return current
                await condition.wait()

    @property
    def version(self) -> int:
        """Current log version (number of entries appended so far)."""
        with self._lock:
            return self._version

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __repr__(self) -> str:
        with self._lock:
            return f"ConversationLog(entries={len(self._entries)}, version={self._version})"
