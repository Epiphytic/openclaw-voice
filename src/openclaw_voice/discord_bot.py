"""
Discord Voice Channel Bot — openclaw-voice (v2 worker architecture).

Joins Discord voice channels and provides interactive voice conversation:
  - Listens to users via per-user audio streams (pycord voice receive)
  - VAD segments speech into utterances
  - Four independent async worker loops per guild: STT → LLM → TTS + Escalation
  - ConversationLog is the single source of truth for conversation state
  - New speech cancels TTS only; LLM and escalation work survives interrupts
  - Plays back TTS audio in the voice channel

Slash commands:
  /join         — bot joins the user's current voice channel
  /leave        — bot disconnects from voice
  /voice <name> — set the TTS voice

Token resolution order (highest priority first):
  1. CLI --token argument
  2. OPENCLAW_VOICE_DISCORD_TOKEN environment variable
  3. Config file [discord] token key

Usage via CLI::

    openclaw-voice discord-bot --token <TOKEN> --guild-id <GID>
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
import time
from pathlib import Path

from openclaw_voice.conversation_log import ConversationLog, LogEntry
from openclaw_voice.vad import FRAME_SIZE, SAMPLE_RATE, SAMPLE_WIDTH, VoiceActivityDetector
from openclaw_voice.voice_pipeline import PipelineConfig, VoicePipeline
from openclaw_voice.voice_session import VoiceSession

log = logging.getLogger("openclaw_voice.discord_bot")

# ---------------------------------------------------------------------------
# Guard — pycord is an optional dependency
# ---------------------------------------------------------------------------
try:
    import discord
    from discord.sinks import Sink  # type: ignore[attr-defined]

    _PYCORD_AVAILABLE = True
except ImportError:
    _PYCORD_AVAILABLE = False
    discord = None  # type: ignore[assignment]
    Sink = object  # type: ignore[assignment,misc]

# Sample rates
DISCORD_SAMPLE_RATE = 48_000  # Discord sends 48kHz stereo opus/PCM
WHISPER_SAMPLE_RATE = 16_000  # whisper.cpp requires 16kHz mono
DISCORD_CHANNELS = 2
DISCORD_SAMPLE_WIDTH = 2  # int16

# Max size of the response playback queue
PLAYBACK_QUEUE_MAXSIZE = 10

# Default VAD settings — tuned for natural speech with brief pauses
DEFAULT_VAD_SILENCE_MS = 1500
DEFAULT_VAD_MIN_SPEECH_MS = 500

# Escalation worker timing
ESCALATION_KEEPALIVE_S = 20
ESCALATION_MAX_WAIT_S = 120


# ---------------------------------------------------------------------------
# VAD Sink (pycord voice receive)
# ---------------------------------------------------------------------------


class VoiceSink(Sink):  # type: ignore[misc]
    """Custom pycord Sink that feeds per-user audio through VAD.

    When a complete utterance is detected for a user (via debounce timer), it
    is placed on the session's ``stt_queue`` for async processing by the STT worker.

    New speech immediately signals ``tts_cancel`` so in-flight TTS is interrupted.

    Args:
        stt_queue:       asyncio.Queue for (user_id, pcm_bytes, enqueued_at, seq) tuples.
        loop:            The running asyncio event loop.
        tts_cancel:      asyncio.Event to set when new speech arrives (cancels TTS).
        vad_silence_ms:  Silence threshold in ms before flushing an utterance.
        vad_min_speech_ms: Minimum speech ms to be considered a real utterance.
        bot_user_id:     Bot's own user ID (audio from self is ignored).
    """

    def __init__(
        self,
        stt_queue: asyncio.Queue,  # type: ignore[type-arg]
        loop: asyncio.AbstractEventLoop,
        tts_cancel: asyncio.Event | None = None,
        vad_silence_ms: int = DEFAULT_VAD_SILENCE_MS,
        vad_min_speech_ms: int = DEFAULT_VAD_MIN_SPEECH_MS,
        bot_user_id: int = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._stt_queue = stt_queue
        self._loop = loop
        self._tts_cancel = tts_cancel
        self._vad_silence_ms = vad_silence_ms
        self._vad_min_speech_ms = vad_min_speech_ms
        self._bot_user_id = bot_user_id
        # Per-user accumulated PCM audio (16kHz mono)
        self._user_audio: dict[int, bytearray] = {}
        # Per-user raw frame buffers (to handle partial frames from resampling)
        self._buffers: dict[int, bytes] = {}
        # Per-user monotonic sequence counter — incremented each utterance
        self._user_seqs: dict[int, int] = {}
        # Per-user debounce timer handles (asyncio.TimerHandle)
        self._flush_timers: dict[int, asyncio.TimerHandle] = {}
        # Debounce delay: flush this many seconds after last packet
        self._flush_delay_s = 0.25
        # Keep _vad_instances for mute-detection compatibility
        self._vad_instances: dict[int, VoiceActivityDetector] = {}

    def write(self, data: bytes, user: int) -> None:  # type: ignore[override]
        """Called by pycord for each audio chunk from a user.

        Accumulates resampled 16kHz mono PCM and resets a per-user debounce
        timer. When no packets arrive for ``_flush_delay_s`` (250ms), the
        timer fires and emits the accumulated audio as an utterance.

        Signals tts_cancel immediately on new speech so the TTS worker is
        interrupted as soon as the user starts talking.
        """
        # Ignore our own audio (bot hearing itself through Discord)
        if user == self._bot_user_id:
            return

        # Signal new speech → cancel any in-flight TTS immediately (thread-safe)
        if self._tts_cancel is not None:
            self._loop.call_soon_threadsafe(self._tts_cancel.set)

        # Accumulate raw data in resample buffer
        buf = self._buffers.get(user, b"") + data
        self._buffers[user] = buf

        # Downsample 48kHz stereo → 16kHz mono PCM
        try:
            mono_pcm = _resample_48k_stereo_to_16k_mono(buf)
        except Exception as exc:
            log.debug("Audio resample error for user %s: %s", user, exc)
            return

        self._buffers[user] = b""

        # Append to per-user audio accumulator
        if user not in self._user_audio:
            self._user_audio[user] = bytearray()
            log.debug("Speech started for user %s", user)
        self._user_audio[user].extend(mono_pcm)

        # Reset the debounce timer. write() is called from pycord's audio
        # thread, so all timer ops go through call_soon_threadsafe.
        self._loop.call_soon_threadsafe(self._schedule_flush, user)

    def _schedule_flush(self, user: int) -> None:
        """Schedule (or reschedule) the debounce timer on the event loop."""
        existing = self._flush_timers.pop(user, None)
        if existing is not None:
            existing.cancel()
        self._flush_timers[user] = self._loop.call_later(
            self._flush_delay_s,
            self._debounce_flush,
            user,
        )

    # Minimum audio duration (ms) to consider an utterance real speech.
    # Anything shorter is likely a noise blip that Whisper will hallucinate on.
    MIN_UTTERANCE_MS = 800

    def _debounce_flush(self, user: int) -> None:
        """Called on the event loop when the debounce timer fires for a user."""
        audio = self._user_audio.pop(user, None)
        self._flush_timers.pop(user, None)

        if audio is None:
            return

        duration_ms = len(audio) // (SAMPLE_RATE * SAMPLE_WIDTH // 1000)
        if duration_ms < self.MIN_UTTERANCE_MS:
            log.debug("Debounce flush — too short (%d ms) for user %s, discarding", duration_ms, user)
            return

        seq = self._user_seqs.get(user, 0) + 1
        self._user_seqs[user] = seq
        duration_ms = len(audio) // (SAMPLE_RATE * SAMPLE_WIDTH // 1000)
        log.info(
            "Debounce flush",
            extra={"user_id": user, "duration_ms": duration_ms, "bytes": len(audio), "seq": seq},
        )
        self._stt_queue.put_nowait(
            (user, bytes(audio), time.monotonic(), seq),
        )

    def flush_user(self, user_id: int) -> bytes | None:
        """Force-flush accumulated audio for a user (e.g. on mute).

        Returns the raw PCM bytes, or None if nothing buffered.
        """
        # Cancel pending debounce timer
        timer = self._flush_timers.pop(user_id, None)
        if timer is not None:
            timer.cancel()

        audio = self._user_audio.pop(user_id, None)
        if audio is None or len(audio) < FRAME_SIZE * 10:
            return None

        duration_ms = len(audio) // (SAMPLE_RATE * SAMPLE_WIDTH // 1000)
        log.info(
            "Utterance force-flushed",
            extra={"duration_ms": duration_ms, "bytes": len(audio)},
        )
        return bytes(audio)

    def cleanup(self) -> None:  # type: ignore[override]
        """Clean up all state when recording stops."""
        for timer in self._flush_timers.values():
            timer.cancel()
        self._flush_timers.clear()
        self._user_audio.clear()
        self._buffers.clear()
        self._user_seqs.clear()
        self._vad_instances.clear()
        log.debug("VoiceSink cleaned up")


# ---------------------------------------------------------------------------
# Voice Bot
# ---------------------------------------------------------------------------


class VoiceBot(discord.Bot if _PYCORD_AVAILABLE else object):  # type: ignore[misc]
    """Discord bot that listens in voice channels and responds via TTS.

    v2 architecture uses four independent async worker loops per guild:
      - STT worker: transcribes audio, appends user_speech to ConversationLog
      - LLM worker: waits for trigger, calls LLM, appends assistant entry
      - TTS worker: synthesises speech, queues audio for playback
      - Escalation worker: handles gateway calls (spawned as needed)

    ConversationLog is the single source of truth. New speech cancels TTS only.

    Args:
        pipeline_config:      VoicePipeline configuration.
        guild_ids:            List of guild IDs to register slash commands for.
        transcript_channel_id: Discord channel ID to post conversation transcripts.
                              Set to None to disable transcript posting.
        vad_silence_ms:       VAD silence threshold in ms (default 1500).
        vad_min_speech_ms:    VAD minimum speech duration in ms (default 500).
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig | None = None,
        guild_ids: list[int] | None = None,
        transcript_channel_id: int | None = None,
        vad_silence_ms: int = DEFAULT_VAD_SILENCE_MS,
        vad_min_speech_ms: int = DEFAULT_VAD_MIN_SPEECH_MS,
        **kwargs,
    ) -> None:
        if not _PYCORD_AVAILABLE:
            raise ImportError(
                "py-cord is required for VoiceBot. "
                "Install with: pip install 'py-cord[voice]' PyNaCl"
            )

        super().__init__(**kwargs)

        self._pipeline_config = pipeline_config or PipelineConfig()
        self._guild_ids = guild_ids or []
        self._transcript_channel_id = transcript_channel_id
        # Per-guild text channel where /join was invoked (overrides global transcript channel)
        self._guild_text_channels: dict[int, int] = {}  # guild_id → channel_id
        self._guild_context: dict[int, dict] = {}  # guild_id → {guild_name, channel, etc.}
        self._vad_silence_ms = vad_silence_ms
        self._vad_min_speech_ms = vad_min_speech_ms

        # Per-guild voice pipelines (one per active voice channel)
        self._pipelines: dict[int, VoicePipeline] = {}

        # Per-guild voice sessions (own all worker tasks + conversation log)
        self._sessions: dict[int, VoiceSession] = {}

        # Per-guild sink refs for mute detection
        self._sinks: dict[int, VoiceSink] = {}

        # Per-guild playback tasks (separate from session workers — plays audio queue)
        self._playback_tasks: dict[int, asyncio.Task] = {}  # type: ignore[type-arg]

        self._register_commands()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_ready(self) -> None:
        log.info("VoiceBot ready", extra={"user": str(self.user)})

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Detect when a user mutes — treat as end of utterance."""
        if member.id == self.user.id:
            return  # ignore our own state changes

        # User just muted (self-mute or server-mute)
        was_muted = before.self_mute or before.mute
        is_muted = after.self_mute or after.mute

        if not was_muted and is_muted:
            guild_id = member.guild.id
            user_id = str(member.id)
            sink = self._sinks.get(guild_id)
            if sink is not None:
                utterance = sink.flush_user(int(user_id))
                if utterance:
                    log.info(
                        "Mute detected — flushing utterance",
                        extra={"user_id": user_id, "bytes": len(utterance)},
                    )
                    session = self._sessions.get(guild_id)
                    if session is not None:
                        uid_int = int(user_id)
                        sink._user_seqs[uid_int] = sink._user_seqs.get(uid_int, 0) + 1
                        session.tts_cancel.set()
                        await session.stt_queue.put(
                            (uid_int, utterance, time.monotonic(), sink._user_seqs[uid_int])
                        )

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def _register_commands(self) -> None:
        """Register slash commands on the bot."""

        @self.slash_command(
            name="join",
            description="Join your current voice channel",
            guild_ids=self._guild_ids or None,
        )
        async def join_cmd(ctx: discord.ApplicationContext) -> None:
            await self._cmd_join(ctx)

        @self.slash_command(
            name="leave",
            description="Disconnect from the voice channel",
            guild_ids=self._guild_ids or None,
        )
        async def leave_cmd(ctx: discord.ApplicationContext) -> None:
            await self._cmd_leave(ctx)

        @self.slash_command(
            name="voice",
            description="Set the TTS voice",
            guild_ids=self._guild_ids or None,
        )
        async def voice_cmd(
            ctx: discord.ApplicationContext,
            name: str,
        ) -> None:
            await self._cmd_voice(ctx, name)

    async def _cmd_join(self, ctx: discord.ApplicationContext) -> None:
        """Handle /join — connect to user's voice channel."""
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.respond("You need to be in a voice channel first.", ephemeral=True)
            return

        channel = ctx.author.voice.channel
        guild_id = ctx.guild_id

        # Disconnect from any existing channel in this guild
        existing = ctx.guild.voice_client
        if existing:
            await self._stop_listening(guild_id)
            await existing.disconnect(force=True)

        try:
            vc = await channel.connect()
        except Exception as exc:
            log.error("Failed to connect to voice channel: %s", exc)
            await ctx.respond(f"Failed to join: {exc}", ephemeral=True)
            return

        # Remember the text channel where /join was invoked for transcripts
        self._guild_text_channels[guild_id] = ctx.channel_id

        await self._start_listening(guild_id, vc)
        await ctx.respond(f"Joined **{channel.name}**. I'm listening! 🎙️")
        log.info(
            "Joined voice channel",
            extra={"guild_id": guild_id, "channel": channel.name},
        )

    async def _cmd_leave(self, ctx: discord.ApplicationContext) -> None:
        """Handle /leave — disconnect from voice channel."""
        guild_id = ctx.guild_id
        vc = ctx.guild.voice_client

        if not vc:
            await ctx.respond("I'm not in a voice channel.", ephemeral=True)
            return

        await self._stop_listening(guild_id)
        self._guild_text_channels.pop(guild_id, None)
        self._guild_context.pop(guild_id, None)
        await vc.disconnect(force=True)
        await ctx.respond("Disconnected. Bye! 👋")
        log.info("Disconnected from voice channel", extra={"guild_id": guild_id})

    async def _cmd_voice(self, ctx: discord.ApplicationContext, name: str) -> None:
        """Handle /voice <name> — change TTS voice."""
        guild_id = ctx.guild_id
        if guild_id in self._pipelines:
            self._pipelines[guild_id].config.tts_voice = name
        else:
            self._pipeline_config.tts_voice = name

        await ctx.respond(f"TTS voice set to **{name}**.")
        log.info("TTS voice changed", extra={"guild_id": guild_id, "voice": name})

    # ------------------------------------------------------------------
    # Voice recording / processing
    # ------------------------------------------------------------------

    def _load_channel_memory(self, channel_id: int) -> str:
        """Load channel memory summary if it exists."""
        memory_path = Path.home() / ".openclaw" / "workspace" / "memory" / f"discord-{channel_id}.md"
        if memory_path.is_file():
            try:
                text = memory_path.read_text().strip()
                if text:
                    log.info("Loaded channel memory from %s (%d chars)", memory_path, len(text))
                    return text
            except Exception as exc:
                log.warning("Failed to read channel memory %s: %s", memory_path, exc)
        return ""

    async def _start_listening(self, guild_id: int, vc: discord.VoiceClient) -> None:
        """Start recording and processing audio in the voice channel."""
        # Build channel context for the pipeline
        guild = vc.guild
        voice_channel = vc.channel
        guild_name = guild.name if guild else str(guild_id)
        channel_name = voice_channel.name if voice_channel else "unknown"

        # Store context for escalation messages
        self._guild_context[guild_id] = {
            "guild_name": guild_name,
            "guild_id": guild_id,
            "voice_channel": channel_name,
            "text_channel_id": self._guild_text_channels.get(guild_id),
        }

        # Load channel memory (from the text channel where /join was invoked)
        text_channel_id = self._guild_text_channels.get(guild_id, guild_id)
        channel_memory = self._load_channel_memory(text_channel_id)

        # Create pipeline for this guild with channel context
        pipeline = VoicePipeline(
            config=self._pipeline_config,
            channel_id=str(guild_id),
        )

        # Build system prompt with channel context injected
        system_prompt = pipeline.build_system_prompt()
        context_addition = f"\n\nYou are in the '{channel_name}' voice channel on the '{guild_name}' Discord server."
        if channel_memory:
            summary = channel_memory[-1500:] if len(channel_memory) > 1500 else channel_memory
            context_addition += f"\n\nRecent channel context:\n{summary}"
        system_prompt += context_addition

        self._pipelines[guild_id] = pipeline

        # Create VoiceSession (holds log, worker tasks, playback queue, events)
        session = VoiceSession(system_prompt=system_prompt, guild_id=guild_id)

        # Attach queue and event not yet in VoiceSession dataclass
        session.stt_queue: asyncio.Queue = asyncio.Queue()  # type: ignore[attr-defined]
        session.pending_tts: asyncio.Event = asyncio.Event()  # type: ignore[attr-defined]

        self._sessions[guild_id] = session

        # Start the voice sink — feeds audio into session.stt_queue
        loop = asyncio.get_event_loop()
        sink = VoiceSink(
            stt_queue=session.stt_queue,
            loop=loop,
            tts_cancel=session.tts_cancel,
            vad_silence_ms=self._vad_silence_ms,
            vad_min_speech_ms=self._vad_min_speech_ms,
            bot_user_id=self.user.id,
        )
        self._sinks[guild_id] = sink
        vc.start_recording(sink, self._on_recording_finished, guild_id)

        # Start the three persistent workers (STT, LLM, TTS)
        session.stt_task = asyncio.create_task(
            self._stt_worker(guild_id, session),
            name=f"stt-{guild_id}",
        )
        session.llm_task = asyncio.create_task(
            self._llm_worker(guild_id, session),
            name=f"llm-{guild_id}",
        )
        session.tts_task = asyncio.create_task(
            self._tts_worker(guild_id, session),
            name=f"tts-{guild_id}",
        )

        # Playback worker (plays audio from session.playback_queue into voice channel)
        self._playback_tasks[guild_id] = asyncio.create_task(
            self._playback_worker(guild_id, session, vc),
            name=f"playback-{guild_id}",
        )

        log.info("Started listening in guild %s (v2 workers)", guild_id)

    async def _stop_listening(self, guild_id: int) -> None:
        """Stop recording and clean up resources for a guild."""
        # Stop all session workers (STT, LLM, TTS, escalation)
        session = self._sessions.pop(guild_id, None)
        if session is not None:
            await session.stop_workers()

        # Cancel playback task (not managed by VoiceSession)
        playback_task = self._playback_tasks.pop(guild_id, None)
        if playback_task and not playback_task.done():
            playback_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await playback_task

        # Clean up remaining per-guild state
        self._pipelines.pop(guild_id, None)
        self._sinks.pop(guild_id, None)

        log.info("Stopped listening in guild %s", guild_id)

    async def _on_recording_finished(self, sink: VoiceSink, guild_id: int) -> None:
        """Callback when pycord stops recording (e.g. bot disconnect)."""
        sink.cleanup()
        log.debug("Recording finished for guild %s", guild_id)

    # ------------------------------------------------------------------
    # Worker loops (v2)
    # ------------------------------------------------------------------

    async def _stt_worker(self, guild_id: int, session: VoiceSession) -> None:
        """STT worker: pull audio from stt_queue, transcribe, append to log.

        Never cancelled by new speech. Runs until /leave.
        """
        pipeline = self._pipelines.get(guild_id)
        if pipeline is None:
            log.error("STT worker started but no pipeline for guild %s", guild_id)
            return

        loop = asyncio.get_event_loop()
        log.debug("STT worker started for guild %s", guild_id)

        try:
            while True:
                user_id, pcm_bytes, enqueued_at, seq = await session.stt_queue.get()

                log.debug(
                    "STT worker: received audio",
                    extra={"guild_id": guild_id, "user_id": user_id, "seq": seq},
                )

                display_name = self._resolve_display_name(guild_id, user_id)

                # Transcribe in executor (blocking Whisper HTTP call)
                transcript = await loop.run_in_executor(
                    None, pipeline.run_stt, pcm_bytes, str(user_id)
                )

                if not transcript:
                    log.debug(
                        "STT: empty/hallucination (user=%s seq=%d)", user_id, seq
                    )
                    continue

                log.info(
                    "Transcribed utterance",
                    extra={
                        "guild_id": guild_id,
                        "user_id": user_id,
                        "seq": seq,
                        "transcript": transcript,
                    },
                )

                # Append to conversation log (single source of truth)
                entry = LogEntry(
                    ts=time.monotonic(),
                    kind="user_speech",
                    speaker=display_name,
                    text=transcript,
                    meta={"seq": seq, "user_id": str(user_id)},
                )
                session.log.append(entry)

                # Signal TTS cancel + wake LLM worker
                session.tts_cancel.set()
                session.pending_llm_trigger.set()

                # Post transcript to Discord text channel
                await self._post_to_channel(
                    guild_id, f"🎙️ **{display_name}**: {transcript}"
                )

        except asyncio.CancelledError:
            log.debug("STT worker cancelled for guild %s", guild_id)
        except Exception as exc:
            log.error("STT worker error for guild %s: %s", guild_id, exc, exc_info=True)

    async def _llm_worker(self, guild_id: int, session: VoiceSession) -> None:
        """LLM worker: wait for trigger, call LLM, append response to log.

        Never cancelled by new speech. Stale responses (log advanced while LLM
        was running) are discarded and the worker re-triggers itself.
        """
        pipeline = self._pipelines.get(guild_id)
        if pipeline is None:
            log.error("LLM worker started but no pipeline for guild %s", guild_id)
            return

        loop = asyncio.get_event_loop()
        log.debug("LLM worker started for guild %s", guild_id)

        try:
            while True:
                # Wait for a new user_speech entry
                await session.pending_llm_trigger.wait()
                session.pending_llm_trigger.clear()

                # Snapshot log version before the LLM call
                _, pre_version = session.log.snapshot()

                # Build message list: system prompt + full conversation history
                messages = [{"role": "system", "content": session.system_prompt}]
                messages += session.log.to_messages()

                log.debug(
                    "LLM worker: calling LLM",
                    extra={
                        "guild_id": guild_id,
                        "messages": len(messages),
                        "log_version": pre_version,
                    },
                )

                # LLM call with tool support (blocking, run in executor)
                response_text, escalation = await loop.run_in_executor(
                    None, pipeline.call_llm_with_tools, messages
                )

                # Check if log advanced while LLM was running (new speech arrived)
                # BUT: don't discard if we already got an escalation or tool result —
                # the work is valuable even if new speech arrived.
                _, post_version = session.log.snapshot()
                if post_version > pre_version and not escalation and not response_text:
                    log.info(
                        "LLM response stale (log v%d → v%d), discarding",
                        pre_version,
                        post_version,
                        extra={"guild_id": guild_id},
                    )
                    # STT worker already set pending_llm_trigger; loop back to wait
                    continue
                if post_version > pre_version and (escalation or response_text):
                    log.info(
                        "Log advanced (v%d → v%d) but LLM produced useful output, keeping",
                        pre_version,
                        post_version,
                        extra={"guild_id": guild_id},
                    )

                bot_name = pipeline.config.bot_name
                agent_name = pipeline.config.main_agent_name

                # Handle escalation tool call
                if escalation:
                    interim_text = f"Let me check with {agent_name} on that."
                    entry = LogEntry(
                        ts=time.monotonic(),
                        kind="assistant",
                        speaker=bot_name,
                        text=interim_text,
                    )
                    session.log.append(entry)
                    session.pending_tts.set()

                    await self._post_to_channel(
                        guild_id, f"🫖 **{bot_name}**: {interim_text}"
                    )

                    # Spawn escalation worker as independent task
                    esc_id = f"esc-{time.monotonic():.3f}"
                    esc_task = asyncio.create_task(
                        self._escalation_worker(session, escalation, guild_id),
                        name=f"escalation-{guild_id}-{esc_id}",
                    )
                    session.escalation_tasks[esc_id] = esc_task

                    # Prune completed escalation tasks
                    session.escalation_tasks = {
                        k: t
                        for k, t in session.escalation_tasks.items()
                        if not t.done()
                    }
                    continue

                if not response_text:
                    log.warning(
                        "LLM returned empty response for guild %s", guild_id
                    )
                    continue

                # Append assistant entry to log and wake TTS worker
                entry = LogEntry(
                    ts=time.monotonic(),
                    kind="assistant",
                    speaker=bot_name,
                    text=response_text,
                )
                session.log.append(entry)
                session.pending_tts.set()

                await self._post_to_channel(
                    guild_id, f"🫖 **{bot_name}**: {response_text}"
                )

        except asyncio.CancelledError:
            log.debug("LLM worker cancelled for guild %s", guild_id)
        except Exception as exc:
            log.error("LLM worker error for guild %s: %s", guild_id, exc, exc_info=True)

    async def _tts_worker(self, guild_id: int, session: VoiceSession) -> None:
        """TTS worker: wait for signal, synthesise latest assistant entry, queue audio.

        This is the ONLY worker that gets interrupted by new speech.
        Checks tts_cancel between synthesis and playback.
        """
        pipeline = self._pipelines.get(guild_id)
        if pipeline is None:
            log.error("TTS worker started but no pipeline for guild %s", guild_id)
            return

        loop = asyncio.get_event_loop()
        log.debug("TTS worker started for guild %s", guild_id)

        try:
            while True:
                # Wait for new TTS signal from LLM worker
                await session.pending_tts.wait()
                session.pending_tts.clear()
                session.clear_tts_cancel()

                # Get the latest assistant entry from the log
                entries, _ = session.log.snapshot()
                assistant_entries = [e for e in entries if e.kind == "assistant"]
                if not assistant_entries:
                    continue
                text = assistant_entries[-1].text

                log.debug(
                    "TTS worker: synthesising response",
                    extra={"guild_id": guild_id, "text_preview": text[:60]},
                )

                # Synthesise in executor (blocking TTS HTTP call)
                audio = await loop.run_in_executor(
                    None, pipeline.synthesize_response, text
                )

                # Check if new speech arrived during synthesis
                if session.tts_cancel.is_set():
                    session.clear_tts_cancel()
                    log.info(
                        "TTS cancelled by new speech for guild %s", guild_id
                    )
                    continue

                if not audio:
                    log.warning("TTS returned empty audio for guild %s", guild_id)
                    continue

                # Queue for playback
                try:
                    session.playback_queue.put_nowait(audio)
                except asyncio.QueueFull:
                    log.warning(
                        "Playback queue full for guild %s, dropping TTS audio", guild_id
                    )

        except asyncio.CancelledError:
            log.debug("TTS worker cancelled for guild %s", guild_id)
        except Exception as exc:
            log.error("TTS worker error for guild %s: %s", guild_id, exc, exc_info=True)

    async def _escalation_worker(
        self,
        session: VoiceSession,
        request_text: str,
        guild_id: int,
    ) -> None:
        """Escalation worker: call main agent via gateway, feed result back to log.

        Runs to completion (or 120s timeout). Never cancelled by new speech.
        Every 20s sends a keepalive audio message to the voice channel.
        On response: appends escalation_result → sets pending_llm_trigger so
        the LLM generates a natural spoken response.
        """
        pipeline = self._pipelines.get(guild_id)
        if pipeline is None:
            log.warning("Escalation worker: no pipeline for guild %s", guild_id)
            return

        bot_name = pipeline.config.bot_name
        agent_name = pipeline.config.main_agent_name
        loop = asyncio.get_event_loop()

        log.info(
            "Escalation worker started",
            extra={"guild_id": guild_id, "request": request_text[:80]},
        )

        try:
            # Post escalation notice to channel
            await self._post_to_channel(
                guild_id,
                f"🫖 **{bot_name} → {agent_name}**: {request_text}",
            )

            # Start gateway task
            gateway_task = asyncio.create_task(
                self._send_to_bel(request_text, "voice user", guild_id)
            )

            keepalive_messages = [
                f"{agent_name}'s thinking about that, one moment.",
                f"Still waiting on {agent_name}, hang tight.",
                f"{agent_name}'s taking a bit — still working on it.",
            ]
            keepalive_count = 0
            elapsed_s = 0
            bel_response: str | None = None

            while elapsed_s < ESCALATION_MAX_WAIT_S:
                try:
                    bel_response = await asyncio.wait_for(
                        asyncio.shield(gateway_task),
                        timeout=ESCALATION_KEEPALIVE_S,
                    )
                    break  # Got a response
                except asyncio.TimeoutError:
                    elapsed_s += ESCALATION_KEEPALIVE_S
                    keepalive_count += 1
                    keepalive_text = keepalive_messages[
                        min(keepalive_count - 1, len(keepalive_messages) - 1)
                    ]
                    log.info(
                        "Escalation keepalive #%d for guild %s", keepalive_count, guild_id
                    )

                    # Append keepalive to log (context for LLM)
                    session.log.append(
                        LogEntry(
                            ts=time.monotonic(),
                            kind="system",
                            speaker="system",
                            text=f"Still waiting for {agent_name}...",
                        )
                    )

                    # Synthesise and queue keepalive audio
                    audio = await loop.run_in_executor(
                        None, pipeline.synthesize_response, keepalive_text
                    )
                    if audio:
                        with contextlib.suppress(asyncio.QueueFull):
                            session.playback_queue.put_nowait(audio)

            # If still running past timeout, cancel and grab any partial result
            if not gateway_task.done():
                gateway_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await gateway_task
            elif not gateway_task.cancelled():
                try:
                    bel_response = bel_response or gateway_task.result()
                except Exception as exc:
                    log.warning("Gateway task raised exception: %s", exc)

            # Truncate very long responses
            if bel_response and len(bel_response) > 1500:
                bel_response = bel_response[:1500] + "..."

            if not bel_response:
                await self._post_to_channel(
                    guild_id,
                    f"🔔 **{agent_name} → {bot_name}**: ⚠️ *(no response — may be busy)*",
                )
                # Append fallback result so LLM can acknowledge the failure
                session.log.append(
                    LogEntry(
                        ts=time.monotonic(),
                        kind="escalation_result",
                        speaker=agent_name,
                        text=f"No response from {agent_name} — may be unavailable right now.",
                    )
                )
                session.pending_llm_trigger.set()
                return

            # Post response to Discord
            display_text = bel_response[:1800] if len(bel_response) > 1800 else bel_response
            await self._post_to_channel(
                guild_id, f"🔔 **{agent_name} → {bot_name}**: {display_text}"
            )

            # Append escalation_result to log → trigger LLM to generate spoken response
            session.log.append(
                LogEntry(
                    ts=time.monotonic(),
                    kind="escalation_result",
                    speaker=agent_name,
                    text=bel_response,
                )
            )
            session.pending_llm_trigger.set()

        except asyncio.CancelledError:
            log.debug("Escalation worker cancelled for guild %s", guild_id)
        except Exception as exc:
            log.error(
                "Escalation worker error for guild %s: %s", guild_id, exc, exc_info=True
            )

    # ------------------------------------------------------------------
    # Gateway (escalation send)
    # ------------------------------------------------------------------

    async def _send_to_bel(self, request: str, user_name: str, guild_id: int = 0) -> str | None:
        """Send a request to Bel via the OpenClaw gateway WebSocket.

        Returns Bel's response text, or None on failure/timeout.
        """
        from openclaw_voice.gateway_client import send_to_bel

        bot_name = self._pipeline_config.bot_name
        agent_name = self._pipeline_config.main_agent_name

        # Build context header with channel info
        ctx = self._guild_context.get(guild_id, {})
        guild_name = ctx.get("guild_name", "unknown server")
        voice_channel = ctx.get("voice_channel", "unknown channel")
        text_channel_id = ctx.get("text_channel_id")

        context_parts = [
            f"[Voice escalation from {bot_name}]",
            f"Discord guild: {guild_name} (ID: {guild_id})",
            f"Voice channel: {voice_channel}",
        ]
        if text_channel_id:
            context_parts.append(f"Text channel ID: {text_channel_id}")

        message = (
            f"{' | '.join(context_parts)}\n"
            f"{user_name} asked via voice: {request}\n\n"
            f"IMPORTANT: Reply with a SHORT answer (1-3 sentences max) suitable for "
            f"text-to-speech. No markdown, no links, no code. {bot_name} will speak "
            f"your response aloud."
        )
        return await send_to_bel(message, timeout_s=90.0)

    # ------------------------------------------------------------------
    # Playback worker
    # ------------------------------------------------------------------

    async def _playback_worker(
        self,
        guild_id: int,
        session: VoiceSession,
        vc: discord.VoiceClient,
    ) -> None:
        """Play audio responses from session.playback_queue into the voice channel."""
        log.debug("Playback worker started for guild %s", guild_id)

        while True:
            try:
                wav_bytes = await session.playback_queue.get()

                if not vc.is_connected():
                    log.debug("Voice client disconnected, dropping playback")
                    session.playback_queue.task_done()
                    continue

                # Write WAV to a temp file for FFmpeg
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tf.write(wav_bytes)
                    tmp_path = tf.name

                try:
                    # Wait for any current playback to finish
                    while vc.is_playing():
                        await asyncio.sleep(0.1)

                    source = discord.FFmpegPCMAudio(tmp_path)
                    vc.play(source)

                    # Wait for playback to complete
                    while vc.is_playing():
                        await asyncio.sleep(0.1)

                except Exception as exc:
                    log.error("Playback error for guild %s: %s", guild_id, exc)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

                session.playback_queue.task_done()

            except asyncio.CancelledError:
                log.debug("Playback worker cancelled for guild %s", guild_id)
                break
            except Exception as exc:
                log.error("Unexpected playback error for guild %s: %s", guild_id, exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_display_name(self, guild_id: int, user_id: int | str) -> str:
        """Resolve a Discord user ID to their display name."""
        uid = int(user_id)
        try:
            guild = self.get_guild(guild_id)
            if guild:
                member = guild.get_member(uid)
                if member:
                    return member.display_name
        except Exception:
            pass
        return str(user_id)

    async def _post_to_channel(self, guild_id: int, message: str) -> None:
        """Post a message to the text channel where /join was invoked.

        Falls back to the global transcript channel if no per-guild channel
        was recorded.
        """
        channel_id = self._guild_text_channels.get(guild_id) or self._transcript_channel_id
        if not channel_id:
            log.warning("No text channel for guild %s, skipping post", guild_id)
            return

        try:
            channel = self.get_channel(channel_id)
            if channel is None:
                channel = await self.fetch_channel(channel_id)
            # Truncate to Discord's 2000 char limit
            if len(message) > 2000:
                message = message[:1997] + "..."
            await channel.send(message)
            log.debug("Posted to channel %s: %s", channel_id, message[:80])
        except Exception as exc:
            log.warning("Failed to post to channel %s: %s", channel_id, exc, exc_info=True)

    def _rephrase_for_speech(self, pipeline: VoicePipeline, bel_response: str) -> str:
        """Use the LLM to rephrase the main agent's response for natural speech.

        Delegates to the pipeline's own rephrase_for_speech() method which
        handles the LLM call and think-tag stripping.
        """
        return pipeline.rephrase_for_speech(bel_response)


# ---------------------------------------------------------------------------
# Audio resampling
# ---------------------------------------------------------------------------


def _resample_48k_stereo_to_16k_mono(pcm: bytes) -> bytes:
    """Downsample 48kHz stereo int16 PCM to 16kHz mono int16 PCM.

    Uses audioop (stdlib) for efficiency. Falls back to a simple decimation
    if audioop is not available (Python 3.13+ removed audioop).

    Args:
        pcm: Raw 48kHz stereo int16 PCM bytes.

    Returns:
        16kHz mono int16 PCM bytes.
    """
    if not pcm:
        return b""

    try:
        import audioop  # type: ignore[import]

        # Stereo → mono
        mono = audioop.tomono(pcm, 2, 0.5, 0.5)
        # 48kHz → 16kHz (factor 3)
        resampled, _ = audioop.ratecv(mono, 2, 1, 48_000, 16_000, None)
        return resampled
    except ImportError:
        pass

    # Fallback: manual decimation (every 3rd sample from averaged stereo)
    import struct

    n_samples = len(pcm) // 4  # 4 bytes per stereo sample (2ch * int16)
    stereo = struct.unpack(f"<{n_samples * 2}h", pcm[: n_samples * 4])

    # Average stereo channels → mono
    mono_samples = [(stereo[i * 2] + stereo[i * 2 + 1]) // 2 for i in range(n_samples)]

    # Decimate by 3 (48kHz / 3 = 16kHz)
    decimated = mono_samples[::3]
    return struct.pack(f"<{len(decimated)}h", *decimated)


# ---------------------------------------------------------------------------
# Bot factory
# ---------------------------------------------------------------------------


def create_bot(
    pipeline_config: PipelineConfig | None = None,
    guild_ids: list[int] | None = None,
    transcript_channel_id: int | None = None,
    vad_silence_ms: int = DEFAULT_VAD_SILENCE_MS,
    vad_min_speech_ms: int = DEFAULT_VAD_MIN_SPEECH_MS,
) -> VoiceBot:
    """Create a configured VoiceBot instance.

    Args:
        pipeline_config:      VoicePipeline configuration.
        guild_ids:            Guild IDs for slash command registration.
        transcript_channel_id: Discord channel ID for transcript posting. None to disable.
        vad_silence_ms:       VAD silence threshold in ms (default 1500).
        vad_min_speech_ms:    VAD minimum speech duration in ms (default 500).

    Returns:
        Configured VoiceBot ready to run.
    """
    if not _PYCORD_AVAILABLE:
        raise ImportError("py-cord is required. Install with: pip install 'py-cord[voice]' PyNaCl")

    intents = discord.Intents.default()
    intents.voice_states = True

    return VoiceBot(
        pipeline_config=pipeline_config,
        guild_ids=guild_ids,
        transcript_channel_id=transcript_channel_id,
        vad_silence_ms=vad_silence_ms,
        vad_min_speech_ms=vad_min_speech_ms,
        intents=intents,
    )
