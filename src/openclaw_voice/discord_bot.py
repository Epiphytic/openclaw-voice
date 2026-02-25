"""
Discord Voice Channel Bot — openclaw-voice.

Joins Discord voice channels and provides interactive voice conversation:
  - Listens to users via per-user audio streams (pycord voice receive)
  - VAD segments speech into utterances
  - Runs each utterance through VoicePipeline (STT → LLM → TTS)
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

from openclaw_voice.vad import FRAME_SIZE, SAMPLE_RATE, SAMPLE_WIDTH, VoiceActivityDetector
from openclaw_voice.voice_pipeline import PipelineConfig, VoicePipeline

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

# If a pipeline response is older than this (seconds since utterance was enqueued)
# by the time it would play, discard it — the user has moved on.
MAX_RESPONSE_AGE_S: float = 120.0

# Default VAD settings — tuned for natural speech with brief pauses
DEFAULT_VAD_SILENCE_MS = 1500
DEFAULT_VAD_MIN_SPEECH_MS = 500


# ---------------------------------------------------------------------------
# VAD Sink (pycord voice receive)
# ---------------------------------------------------------------------------


class VoiceSink(Sink):  # type: ignore[misc]
    """Custom pycord Sink that feeds per-user audio through VAD.

    When a complete utterance is detected for a user, it is placed on
    the shared ``utterance_queue`` for async processing.

    Args:
        utterance_queue: asyncio.Queue for (user_id, pcm_bytes, enqueued_at, seq) tuples.
        loop:            The running asyncio event loop.
        vad_silence_ms:  Silence threshold in ms before flushing an utterance.
        vad_min_speech_ms: Minimum speech ms to be considered a real utterance.
    """

    def __init__(
        self,
        utterance_queue: asyncio.Queue,  # type: ignore[type-arg]
        loop: asyncio.AbstractEventLoop,
        vad_silence_ms: int = DEFAULT_VAD_SILENCE_MS,
        vad_min_speech_ms: int = DEFAULT_VAD_MIN_SPEECH_MS,
        bot_user_id: int = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._utterance_queue = utterance_queue
        self._loop = loop
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

        If new speech arrives while a previous utterance is being processed,
        the dispatcher in VoiceBot handles cancellation and concatenation.
        """
        # Ignore our own audio (bot hearing itself through Discord)
        if user == self._bot_user_id:
            return



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
        self._utterance_queue.put_nowait(
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

        # Per-guild utterance queues and sink references
        self._utterance_queues: dict[int, asyncio.Queue] = {}  # type: ignore[type-arg]
        self._processing_tasks: dict[int, asyncio.Task] = {}  # type: ignore[type-arg]
        self._playback_queues: dict[int, asyncio.Queue] = {}  # type: ignore[type-arg]
        self._playback_tasks: dict[int, asyncio.Task] = {}  # type: ignore[type-arg]
        self._sinks: dict[int, VoiceSink] = {}  # per-guild sink refs for mute detection

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
                    q = self._utterance_queues.get(guild_id)
                    if q is not None:
                        uid_int = int(user_id)
                        sink._user_seqs[uid_int] = sink._user_seqs.get(uid_int, 0) + 1
                        await q.put((user_id, utterance, time.monotonic(), sink._user_seqs[uid_int]))

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

        # Inject channel context into the system prompt
        context_addition = f"\n\nYou are in the '{channel_name}' voice channel on the '{guild_name}' Discord server."
        if channel_memory:
            # Take last ~500 chars of channel memory for context
            summary = channel_memory[-1500:] if len(channel_memory) > 1500 else channel_memory
            context_addition += f"\n\nRecent channel context:\n{summary}"
        pipeline._system_prompt += context_addition

        self._pipelines[guild_id] = pipeline

        # Create queues
        utterance_q: asyncio.Queue = asyncio.Queue()  # type: ignore[type-arg]
        playback_q: asyncio.Queue = asyncio.Queue(maxsize=PLAYBACK_QUEUE_MAXSIZE)  # type: ignore[type-arg]
        self._utterance_queues[guild_id] = utterance_q
        self._playback_queues[guild_id] = playback_q

        # Start the voice sink with configurable VAD parameters
        loop = asyncio.get_event_loop()
        sink = VoiceSink(
            utterance_q,
            loop,
            vad_silence_ms=self._vad_silence_ms,
            vad_min_speech_ms=self._vad_min_speech_ms,
            bot_user_id=self.user.id,
        )
        self._sinks[guild_id] = sink
        vc.start_recording(sink, self._on_recording_finished, guild_id)

        # Start processing and playback tasks
        self._processing_tasks[guild_id] = asyncio.create_task(
            self._process_utterances(guild_id, vc)
        )
        self._playback_tasks[guild_id] = asyncio.create_task(self._playback_worker(guild_id, vc))

        log.info("Started listening in guild %s", guild_id)

    async def _stop_listening(self, guild_id: int) -> None:
        """Stop recording and clean up resources for a guild."""
        # Cancel processing task
        for tasks in (self._processing_tasks, self._playback_tasks):
            task = tasks.pop(guild_id, None)
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Clean up pipelines and queues
        self._pipelines.pop(guild_id, None)
        self._utterance_queues.pop(guild_id, None)
        self._playback_queues.pop(guild_id, None)
        self._sinks.pop(guild_id, None)

        log.info("Stopped listening in guild %s", guild_id)

    async def _on_recording_finished(self, sink: VoiceSink, guild_id: int) -> None:
        """Callback when pycord stops recording (e.g. bot disconnect)."""
        sink.cleanup()
        log.debug("Recording finished for guild %s", guild_id)

    async def _process_utterances(
        self,
        guild_id: int,
        vc: discord.VoiceClient,
    ) -> None:
        """Dispatch per-user pipeline tasks from the utterance queue.

        Each user gets their own asyncio Task so that a new utterance from
        the same user immediately cancels any in-flight response — preventing
        stale responses from piling up when speech is fragmented.

        Utterance tuples from VoiceSink: ``(user_id, pcm_bytes, enqueued_at, seq)``
        """
        if guild_id not in self._utterance_queues:
            log.error("Utterance queue missing for guild %s", guild_id)
            return

        log.debug("Utterance dispatcher started for guild %s", guild_id)
        utterance_q = self._utterance_queues[guild_id]

        # Per-user in-flight pipeline tasks and accumulated audio
        user_tasks: dict[int, asyncio.Task] = {}  # type: ignore[type-arg]
        user_audio: dict[int, bytearray] = {}  # accumulated PCM across utterances

        try:
            while True:
                user_id, pcm_bytes, enqueued_at, seq = await utterance_q.get()
                log.debug(
                    "Utterance received",
                    extra={"guild_id": guild_id, "user_id": user_id, "seq": seq},
                )

                # Always accumulate audio
                if user_id not in user_audio:
                    user_audio[user_id] = bytearray()
                user_audio[user_id].extend(pcm_bytes)

                # If there's an in-flight task that hasn't started TTS yet,
                # cancel it — we'll restart with the combined audio.
                old_task = user_tasks.get(user_id)
                if old_task and not old_task.done():
                    log.info(
                        "New speech from user %s (seq=%d); cancelling in-flight pipeline to append",
                        user_id,
                        seq,
                    )
                    old_task.cancel()
                    await asyncio.sleep(0)

                # Start a new task with ALL accumulated audio for this user
                combined_audio = bytes(user_audio[user_id])

                # Create a wrapper that clears accumulated audio on success
                async def _run_and_clear(
                    uid: int = user_id,
                    audio: bytes = combined_audio,
                    eat: float = enqueued_at,
                    s: int = seq,
                ) -> None:
                    await self._run_single_utterance(guild_id, vc, uid, audio, eat, s)
                    # Pipeline completed (TTS played) — clear accumulated audio
                    user_audio.pop(uid, None)

                task = asyncio.create_task(
                    _run_and_clear(),
                    name=f"pipeline-{guild_id}-{user_id}-{seq}",
                )
                user_tasks[user_id] = task

                # Clean up completed tasks to avoid memory growth
                user_tasks = {uid: t for uid, t in user_tasks.items() if not t.done()}

                utterance_q.task_done()

        except asyncio.CancelledError:
            log.debug("Utterance dispatcher cancelled for guild %s", guild_id)
            # Cancel all in-flight user tasks on shutdown
            for task in user_tasks.values():
                if not task.done():
                    task.cancel()

    async def _run_single_utterance(
        self,
        guild_id: int,
        vc: discord.VoiceClient,
        user_id: int,
        pcm_bytes: bytes,
        enqueued_at: float,
        seq: int,
    ) -> None:
        """Run the STT → LLM → TTS pipeline for a single user utterance.

        Checks response age before queuing playback. If cancelled (due to a
        newer utterance arriving), exits cleanly without queuing audio.
        Also posts a transcript summary to the configured transcript channel.
        """
        pipeline = self._pipelines.get(guild_id)
        playback_q = self._playback_queues.get(guild_id)

        if pipeline is None or playback_q is None:
            log.error("Pipeline or playback queue gone for guild %s", guild_id)
            return

        display_name = self._resolve_display_name(guild_id, user_id)
        bot_name = pipeline.config.bot_name
        agent_name = pipeline.config.main_agent_name

        try:
            # ── Phase 1: STT + LLM (cancellable) ────────────────────────────
            loop = asyncio.get_event_loop()
            transcript, response_text = await loop.run_in_executor(
                None,
                pipeline.process_utterance,
                pcm_bytes,
                str(user_id),
            )

            if not transcript:
                return  # nothing heard

            # Post user's speech to channel
            await self._post_to_channel(
                guild_id, f"🎙️ **{display_name}**: {transcript}"
            )

            if not response_text:
                await self._post_to_channel(
                    guild_id, f"🫖 **{bot_name}**: ⚠️ *(no response)*"
                )
                return

            # ── Cancellation checkpoint ──────────────────────────────────────
            # If a new utterance arrived while we were doing STT+LLM, this
            # task will be cancelled HERE — before we waste time on TTS.
            await asyncio.sleep(0)

            # ── Response age guard ───────────────────────────────────────────
            response_age_s = time.monotonic() - enqueued_at
            if response_age_s > MAX_RESPONSE_AGE_S:
                log.info(
                    "Response discarded — too old (%.1fs) for user %s seq=%d",
                    response_age_s, user_id, seq,
                )
                await self._post_to_channel(
                    guild_id, f"🫖 **{bot_name}**: {response_text} ⚠️ *stale — not spoken*"
                )
                return

            # ── Phase 2: TTS ─────────────────────────────────────────────────
            response_audio = await loop.run_in_executor(
                None,
                pipeline.synthesize_response,
                response_text,
                str(user_id),
            )

            # Post Chip's response to channel
            await self._post_to_channel(
                guild_id, f"🫖 **{bot_name}**: {response_text}"
            )

        except asyncio.CancelledError:
            log.info(
                "Pipeline task cancelled (user %s seq=%d) — newer utterance superseded it",
                user_id, seq,
            )
            raise

        except Exception as exc:
            log.error(
                "Error in pipeline for guild %s user %s: %s",
                guild_id, user_id, exc,
            )
            return

        if not response_audio:
            return

        # ── Queue audio for playback ─────────────────────────────────────────
        try:
            playback_q.put_nowait(response_audio)
        except asyncio.QueueFull:
            log.warning(
                "Playback queue full for guild %s, dropping response (user %s)",
                guild_id, user_id,
            )

        # ── Escalation to Bel ────────────────────────────────────────────────
        # Spawn as independent task so it survives cancellation from new speech
        escalation = pipeline.last_escalation
        if escalation:
            asyncio.create_task(
                self._handle_escalation(
                    guild_id, vc, pipeline, playback_q, display_name, escalation
                ),
                name=f"escalation-{guild_id}-{user_id}",
            )

    async def _handle_escalation(
        self,
        guild_id: int,
        vc: discord.VoiceClient,
        pipeline,
        playback_q: asyncio.Queue,
        display_name: str,
        escalation_request: str,
    ) -> None:
        """Send an escalation request to the main agent via OpenClaw gateway.

        Runs the gateway call with a keepalive — if it takes longer than
        KEEPALIVE_INTERVAL_S, synthesizes a "still thinking" message so the
        user isn't left hanging.
        """
        KEEPALIVE_INTERVAL_S = 20

        bot_name = pipeline.config.bot_name
        agent_name = pipeline.config.main_agent_name

        try:
            await self._post_to_channel(
                guild_id,
                f"🫖 **{bot_name} → {agent_name}**: {escalation_request}",
            )

            # Run gateway call with keepalive loop
            gateway_task = asyncio.create_task(
                self._send_to_bel(escalation_request, display_name, guild_id)
            )

            keepalive_count = 0
            keepalive_messages = [
                f"{agent_name}'s thinking about that, one moment.",
                f"Still waiting on {agent_name}, hang tight.",
                f"{agent_name}'s taking a bit — still working on it.",
            ]

            while not gateway_task.done():
                try:
                    bel_response = await asyncio.wait_for(
                        asyncio.shield(gateway_task), timeout=KEEPALIVE_INTERVAL_S
                    )
                    break  # Got a response
                except asyncio.TimeoutError:
                    # Gateway still working — send keepalive
                    keepalive_count += 1
                    msg = keepalive_messages[
                        min(keepalive_count - 1, len(keepalive_messages) - 1)
                    ]
                    log.info(
                        "Escalation keepalive #%d for guild %s", keepalive_count, guild_id
                    )
                    loop = asyncio.get_event_loop()
                    audio = await loop.run_in_executor(
                        None, pipeline.synthesize_response, msg,
                    )
                    if audio:
                        try:
                            playback_q.put_nowait(audio)
                        except asyncio.QueueFull:
                            pass

                    if keepalive_count >= 4:
                        # Give up after ~80s
                        gateway_task.cancel()
                        bel_response = None
                        break
            else:
                bel_response = gateway_task.result()

            # Truncate long responses
            if bel_response and len(bel_response) > 1500:
                bel_response = bel_response[:1500] + "..."

            if not bel_response:
                await self._post_to_channel(
                    guild_id,
                    f"🔔 **{agent_name} → {bot_name}**: ⚠️ *(no response — may be busy)*",
                )
                loop = asyncio.get_event_loop()
                audio = await loop.run_in_executor(
                    None,
                    pipeline.synthesize_response,
                    f"Sorry, I couldn't reach {agent_name} right now. Try again in a moment.",
                )
                if audio:
                    try:
                        playback_q.put_nowait(audio)
                    except asyncio.QueueFull:
                        pass
                return

            # Post Bel's response to channel
            bel_display = bel_response[:1800] if len(bel_response) > 1800 else bel_response
            await self._post_to_channel(
                guild_id, f"🔔 **{agent_name} → {bot_name}**: {bel_display}"
            )

            # Rephrase for speech and synthesize
            loop = asyncio.get_event_loop()
            spoken_response = await loop.run_in_executor(
                None, self._rephrase_for_speech, pipeline, bel_response,
            )

            audio = await loop.run_in_executor(
                None, pipeline.synthesize_response, spoken_response,
            )

            if audio:
                await self._post_to_channel(
                    guild_id, f"🫖 **{bot_name}**: {spoken_response}"
                )
                try:
                    playback_q.put_nowait(audio)
                except asyncio.QueueFull:
                    pass

        except Exception as exc:
            log.error("Escalation handler error for guild %s: %s", guild_id, exc)
            try:
                loop = asyncio.get_event_loop()
                audio = await loop.run_in_executor(
                    None,
                    pipeline.synthesize_response,
                    f"Sorry, something went wrong with the escalation.",
                )
                if audio:
                    playback_q.put_nowait(audio)
            except Exception:
                pass

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

    def _rephrase_for_speech(self, pipeline, bel_response: str) -> str:
        """Use the LLM to rephrase the main agent's response for natural speech."""
        agent_name = pipeline.config.main_agent_name
        bot_name = pipeline.config.bot_name
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

        result = pipeline._call_openai_compat(messages)
        if isinstance(result, str) and result:
            # Strip think tags
            if "<think>" in result:
                import re
                result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            return result
        return bel_response  # fallback: use Bel's raw response

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
            return

        try:
            channel = self.get_channel(channel_id)
            if channel is None:
                channel = await self.fetch_channel(channel_id)
            await channel.send(message)
        except Exception as exc:
            log.warning("Failed to post to channel %s: %s", channel_id, exc)

    async def _playback_worker(
        self,
        guild_id: int,
        vc: discord.VoiceClient,
    ) -> None:
        """Play audio responses from the playback queue."""
        playback_q = self._playback_queues.get(guild_id)
        if playback_q is None:
            return

        log.debug("Playback worker started for guild %s", guild_id)

        while True:
            try:
                wav_bytes = await playback_q.get()

                if not vc.is_connected():
                    log.debug("Voice client disconnected, dropping playback")
                    playback_q.task_done()
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

                playback_q.task_done()

            except asyncio.CancelledError:
                log.debug("Playback worker cancelled for guild %s", guild_id)
                break
            except Exception as exc:
                log.error("Unexpected playback error for guild %s: %s", guild_id, exc)


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
