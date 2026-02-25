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
import collections
import contextlib
import json
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

_VOICE_STATE_FILE = Path.home() / ".openclaw" / "workspace" / "openclaw-voice" / "voice_state.json"
_PID_FILE = Path.home() / ".openclaw" / "workspace" / "openclaw-voice" / "chip.pid"


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

    # Default pre-buffer: 300ms of audio at 20ms/frame = 15 frames
    DEFAULT_PRE_BUFFER_MS = 300
    # Discord sends ~20ms frames
    _FRAME_DURATION_MS = 20

    def __init__(
        self,
        stt_queue: asyncio.Queue,  # type: ignore[type-arg]
        loop: asyncio.AbstractEventLoop,
        tts_cancel: asyncio.Event | None = None,
        vad_silence_ms: int = DEFAULT_VAD_SILENCE_MS,
        vad_min_speech_ms: int = DEFAULT_VAD_MIN_SPEECH_MS,
        bot_user_id: int = 0,
        pre_buffer_ms: int = DEFAULT_PRE_BUFFER_MS,
        speech_end_delay_ms: int = 1000,
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
        # Pre-buffer config
        self._pre_buffer_frames = max(1, pre_buffer_ms // self._FRAME_DURATION_MS)
        # Per-user rolling pre-buffer of resampled 16kHz mono PCM chunks
        self._pre_buffers: dict[int, collections.deque[bytes]] = {}
        # Per-user accumulated PCM audio (16kHz mono)
        self._user_audio: dict[int, bytearray] = {}
        # Per-user raw frame buffers (to handle partial frames from resampling)
        self._buffers: dict[int, bytes] = {}
        # Per-user monotonic sequence counter — incremented each utterance
        self._user_seqs: dict[int, int] = {}
        # Per-user debounce timer handles (asyncio.TimerHandle)
        self._flush_timers: dict[int, asyncio.TimerHandle] = {}
        # Debounce delay: flush this many seconds after last packet
        self._flush_delay_s = speech_end_delay_ms / 1000.0
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

        # Feed the rolling pre-buffer (always, even during active speech)
        if user not in self._pre_buffers:
            self._pre_buffers[user] = collections.deque(maxlen=self._pre_buffer_frames)
        self._pre_buffers[user].append(mono_pcm)

        # Append to per-user audio accumulator
        if user not in self._user_audio:
            # Speech just started — prepend pre-buffer to capture lead-in audio
            self._user_audio[user] = bytearray()
            pre_buf = self._pre_buffers.get(user)
            if pre_buf and len(pre_buf) > 1:
                # All frames except the last (which is the current chunk)
                for chunk in list(pre_buf)[:-1]:
                    self._user_audio[user].extend(chunk)
            log.debug(
                "Speech started for user %s (pre-buffered %d frames)",
                user,
                len(pre_buf) - 1 if pre_buf and len(pre_buf) > 1 else 0,
            )
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
        self._pre_buffers.clear()
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
        speech_end_delay_ms: int = 1000,
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
        self._speech_end_delay_ms = speech_end_delay_ms

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
        # Kill any previous instance and write our PID
        self._claim_pid_file()
        # Build cast of characters for all guilds
        await self._build_cast()
        log.info("VoiceBot ready", extra={"user": str(self.user)})
        await self._restore_voice_state()

    # ------------------------------------------------------------------
    # Cast of characters — built on startup for fast message handling
    # ------------------------------------------------------------------

    # Roles for on_message routing:
    #   "self"   — this bot (Chip), skip entirely
    #   "agent"  — main agent bot (belthanior/Bel), skip in bridge (handled by escalation TTS)
    #   "bot"    — other bots, identify by name in TTS bridge
    #   "user"   — human users, skip if in voice channel, otherwise identify
    _cast: dict[int, dict] = {}  # user_id → {"name": str, "role": str}

    async def _build_cast(self) -> None:
        """Build a lookup of all members and bots across guilds."""
        main_name = self._pipeline_config.main_agent_name.lower()
        self_id = self.user.id

        for guild in self.guilds:
            # Fetch full member list (works even without members intent for small guilds)
            try:
                members = guild.members
                if not members or len(members) < 2:
                    # Cache might be empty — try fetching
                    members = []
                    async for member in guild.fetch_members(limit=200):
                        members.append(member)
            except Exception as exc:
                log.warning("Failed to fetch members for guild %s: %s", guild.id, exc)
                members = guild.members or []

            for member in members:
                if member.id == self_id:
                    role = "self"
                elif member.bot:
                    name_lower = (member.display_name or member.name).lower()
                    if name_lower in (main_name, "belthanior", "bel"):
                        role = "agent"
                    else:
                        role = "bot"
                else:
                    role = "user"

                self._cast[member.id] = {
                    "name": member.display_name or member.name,
                    "role": role,
                }

            log.info(
                "Cast of characters for guild %s: %d members (%d users, %d bots, agent=%s)",
                guild.name,
                len(members),
                sum(1 for v in self._cast.values() if v["role"] == "user"),
                sum(1 for v in self._cast.values() if v["role"] in ("bot", "agent", "self")),
                next((v["name"] for v in self._cast.values() if v["role"] == "agent"), "none"),
            )

    def _cast_lookup(self, user_id: int) -> dict:
        """Look up a user in the cast. Returns {"name": str, "role": str}."""
        return self._cast.get(user_id, {"name": str(user_id), "role": "unknown"})

    @staticmethod
    def _claim_pid_file() -> None:
        """Write our PID and kill any stale previous instance."""
        import os
        import signal

        if _PID_FILE.is_file():
            try:
                old_pid = int(_PID_FILE.read_text().strip())
                if old_pid != os.getpid():
                    os.kill(old_pid, signal.SIGTERM)
                    log.info("Killed previous instance (PID %d)", old_pid)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        _PID_FILE.write_text(str(os.getpid()))

    async def _restore_voice_state(self) -> None:
        """Auto-reconnect to the last voice channel on restart."""
        if not _VOICE_STATE_FILE.is_file():
            return
        try:
            state = json.loads(_VOICE_STATE_FILE.read_text())
            guild_id = state["guild_id"]
            voice_channel_id = state["voice_channel_id"]
            text_channel_id = state["text_channel_id"]

            guild = self.get_guild(guild_id)
            if guild is None:
                log.warning("Restore voice state: guild %s not found", guild_id)
                return

            channel = guild.get_channel(voice_channel_id)
            if channel is None:
                log.warning("Restore voice state: voice channel %s not found", voice_channel_id)
                return

            vc = await channel.connect()
            self._guild_text_channels[guild_id] = text_channel_id
            await self._start_listening(guild_id, vc)
            log.info(
                "Restored voice connection",
                extra={"guild_id": guild_id, "channel": channel.name},
            )
        except Exception as exc:
            log.warning("Failed to restore voice state: %s", exc)

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

    async def on_message(self, message: discord.Message) -> None:
        """Bridge text channel messages to voice TTS using the cast of characters.

        Routing by role:
          "self"    → skip (Chip's own messages)
          "agent"   → skip (Bel's posts — already handled by escalation TTS)
          "user"    → skip if in voice channel, otherwise read aloud with attribution
          "bot"     → read aloud with attribution
          "unknown" → read aloud with attribution (new member not in startup cast)
        """
        if not message.guild:
            return

        guild_id = message.guild.id
        session = self._sessions.get(guild_id)
        if session is None:
            return

        # Must match the linked text channel
        text_channel_id = self._guild_text_channels.get(guild_id)
        if not text_channel_id or message.channel.id != text_channel_id:
            return

        # Look up in cast
        cast_entry = self._cast_lookup(message.author.id)
        role = cast_entry["role"]
        display_name = cast_entry["name"]

        # If not in cast yet (joined after startup), add them now
        if role == "unknown":
            display_name = message.author.display_name or message.author.name
            if message.author.bot:
                # Check if this is the main agent
                name_lower = display_name.lower()
                main_name = self._pipeline_config.main_agent_name.lower()
                if name_lower in (main_name, "belthanior", "bel"):
                    role = "agent"
                else:
                    role = "bot"
            else:
                role = "user"
            self._cast[message.author.id] = {"name": display_name, "role": role}
            log.info("Added to cast: %s (id=%d, role=%s)", display_name, message.author.id, role)

        # Route by role
        if role == "self":
            return
        if role == "agent":
            log.debug("on_message: skipping agent message from %s", display_name)
            return

        # Users in voice can hear themselves type — skip
        if role == "user":
            vc = message.guild.voice_client if message.guild else None
            if vc and vc.channel:
                voice_member_ids = {m.id for m in vc.channel.members}
                if message.author.id in voice_member_ids:
                    log.debug("on_message: skipping — %s is in voice channel", display_name)
                    return

        content = message.content.strip()
        if not content:
            return

        pipeline = self._pipelines.get(guild_id)
        if pipeline is None or not pipeline.config.tts_read_channel:
            return

        loop = asyncio.get_event_loop()
        word_count = len(content.split())

        # Summarize long messages via LLM so TTS stays concise
        if word_count > 250:
            try:
                summary_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Summarize the following message in 2 sentences. "
                            "Be concise and factual. /no_think"
                        ),
                    },
                    {"role": "user", "content": content},
                ]
                summary = await loop.run_in_executor(
                    None, pipeline._call_openai_compat, summary_messages
                )
                if summary and isinstance(summary, str):
                    content = summary.strip()
                else:
                    content = " ".join(content.split()[:250]) + "…"
            except Exception as exc:
                log.warning("Failed to summarize channel message: %s", exc)
                content = " ".join(content.split()[:250]) + "…"

        # Attribution: other speakers get identified, self/agent never reach here
        tts_text = f"{display_name} says: {content}"

        log.info(
            "Text-to-voice bridge: reading message",
            extra={"guild_id": guild_id, "author": display_name, "role": role, "words": word_count},
        )

        audio = await loop.run_in_executor(
            None, pipeline.synthesize_response, tts_text
        )
        if audio:
            with contextlib.suppress(asyncio.QueueFull):
                session.playback_queue.put_nowait((audio, None))

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

        # Persist voice state for auto-reconnect on restart
        try:
            _VOICE_STATE_FILE.write_text(json.dumps({
                "guild_id": guild_id,
                "voice_channel_id": channel.id,
                "text_channel_id": ctx.channel_id,
            }))
        except Exception as exc:
            log.warning("Failed to save voice state: %s", exc)

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

        # Clear persisted voice state
        try:
            _VOICE_STATE_FILE.unlink(missing_ok=True)
        except Exception as exc:
            log.warning("Failed to clear voice state: %s", exc)
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
        text_channel_id = self._guild_text_channels.get(guild_id)
        text_channel_obj = self.get_channel(text_channel_id) if text_channel_id else None
        text_channel_name = getattr(text_channel_obj, "name", None) or "unknown"
        self._guild_context[guild_id] = {
            "guild_name": guild_name,
            "guild_id": guild_id,
            "voice_channel": channel_name,
            "text_channel_id": text_channel_id,
            "text_channel_name": text_channel_name,
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
        context_addition = f"\n\nYou are in the '{channel_name}' voice channel on the '{guild_name}' Discord server. The linked text channel is #{text_channel_name}."
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
            speech_end_delay_ms=self._speech_end_delay_ms,
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

                # Use cast first, fall back to async resolution
                cast_entry = self._cast_lookup(int(user_id))
                if cast_entry["role"] != "unknown":
                    display_name = cast_entry["name"]
                else:
                    display_name = await self._resolve_display_name_async(guild_id, user_id)
                    # Cache in cast for next time
                    self._cast[int(user_id)] = {"name": display_name, "role": "user"}

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

                # Inject recent text channel messages into the system prompt
                system_content = session.system_prompt
                n_ctx = pipeline.config.channel_context_messages
                if n_ctx > 0:
                    text_channel_id = self._guild_text_channels.get(guild_id)
                    if text_channel_id:
                        try:
                            text_channel = self.get_channel(text_channel_id)
                            if text_channel:
                                ctx_lines: list[str] = []
                                async for msg in text_channel.history(limit=n_ctx):
                                    if msg.author.id == self.user.id:
                                        continue
                                    if msg.content:
                                        ctx_lines.append(
                                            f"{msg.author.display_name}: {msg.content}"
                                        )
                                ctx_lines.reverse()  # chronological order
                                if ctx_lines:
                                    ch_name = getattr(text_channel, "name", str(text_channel_id))
                                    header = f"[Text channel #{ch_name} — last {len(ctx_lines)} messages]"
                                    system_content = (
                                        session.system_prompt
                                        + f"\n\n{header}\n"
                                        + "\n".join(ctx_lines)
                                    )
                        except Exception as _ctx_exc:
                            log.warning(
                                "Failed to fetch channel context for guild %s: %s",
                                guild_id,
                                _ctx_exc,
                            )

                # Build message list: system prompt + full conversation history
                messages = [{"role": "system", "content": system_content}]
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
                    interim_text = "One moment."
                    entry = LogEntry(
                        ts=time.monotonic(),
                        kind="assistant",
                        speaker=bot_name,
                        text=interim_text,
                    )
                    session.log.append(entry)
                    session._pending_channel_post = f"🫖 **{bot_name}**: {interim_text}"
                    session.pending_tts.set()

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
                # Store the channel post text on the session so playback worker
                # can post AFTER audio plays (not before)
                session._pending_channel_post = f"🫖 **{bot_name}**: {response_text}"
                session.pending_tts.set()

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

                # Queue for playback — tuple of (audio, channel_post_text)
                # Channel post text is posted AFTER playback completes
                channel_post = getattr(session, "_pending_channel_post", None)
                session._pending_channel_post = None
                try:
                    session.playback_queue.put_nowait((audio, channel_post))
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
        """Escalation worker: send user question to main agent, TTS the response.

        Simple passthrough — no LLM rephrase, no channel posting by the voice
        bot. The main agent posts to the channel itself as part of its response.
        Voice agent just renders the response as TTS audio.

        Flow:
          1. Voice agent already said "One moment" (queued by LLM worker)
          2. Send user question to main agent via gateway
          3. Main agent responds (and posts to channel on its own)
          4. Voice agent TTS-renders the response -> playback
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
            # Send to main agent via gateway
            bel_response = await asyncio.wait_for(
                self._send_to_bel(request_text, "voice user", guild_id),
                timeout=ESCALATION_MAX_WAIT_S,
            )

            # Filter non-answers (NO_REPLY, bare "NO", etc.)
            if bel_response:
                stripped = bel_response.strip().upper()
                if stripped in ("NO_REPLY", "NO", "NOREPLY", "N/A", ""):
                    log.warning("Escalation returned non-answer: %r — treating as empty", bel_response)
                    bel_response = None

            if not bel_response:
                fail_text = f"Sorry, I couldn't reach {agent_name}."
                audio = await loop.run_in_executor(
                    None, pipeline.synthesize_response, fail_text
                )
                if audio:
                    with contextlib.suppress(asyncio.QueueFull):
                        session.playback_queue.put_nowait((audio, None))
                session.log.append(
                    LogEntry(
                        ts=time.monotonic(),
                        kind="escalation_result",
                        speaker=agent_name,
                        text=f"No response from {agent_name}.",
                    )
                )
                return

            # Truncate for TTS
            tts_text = bel_response[:1500] if len(bel_response) > 1500 else bel_response

            log.info(
                "Escalation response received, rendering TTS",
                extra={"guild_id": guild_id, "response_len": len(bel_response)},
            )

            # Record in conversation log for context
            session.log.append(
                LogEntry(
                    ts=time.monotonic(),
                    kind="escalation_result",
                    speaker=agent_name,
                    text=bel_response,
                )
            )

            # Post Bel's response to the text channel so users have a text record
            await self._post_to_channel(
                guild_id, f"🜂 **{agent_name}**: {bel_response}"
            )

            # TTS render directly — no rephrase
            audio = await loop.run_in_executor(
                None, pipeline.synthesize_response, tts_text
            )
            if audio:
                with contextlib.suppress(asyncio.QueueFull):
                    session.playback_queue.put_nowait((audio, None))

        except asyncio.TimeoutError:
            log.warning("Escalation timed out for guild %s", guild_id)
            fail_text = f"{agent_name} is taking too long, try again later."
            audio = await loop.run_in_executor(
                None, pipeline.synthesize_response, fail_text
            )
            if audio:
                with contextlib.suppress(asyncio.QueueFull):
                    session.playback_queue.put_nowait((audio, None))
        except asyncio.CancelledError:
            log.debug("Escalation worker cancelled for guild %s", guild_id)
        except Exception as exc:
            log.error(
                "Escalation worker error for guild %s: %s", guild_id, exc, exc_info=True
            )

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

        text_channel_name = ctx.get("text_channel_name", "unknown")
        context_parts = [
            f"[Voice escalation from {bot_name}]",
            f"Discord guild: {guild_name} (ID: {guild_id})",
            f"Voice channel: {voice_channel}",
        ]
        if text_channel_id:
            context_parts.append(f"Text channel: #{text_channel_name} (ID: {text_channel_id})")

        # Build the instruction block, including channel-post directive if available
        instructions = (
            "Your response will be spoken aloud via TTS — keep it SHORT "
            "(1-3 sentences), no markdown, no links, no code blocks.\n"
            "Do NOT say NO_REPLY — always respond with text for the voice user."
        )
        if text_channel_id:
            instructions += (
                f"\nAlso post your full response as a Discord message to "
                f"#{text_channel_name} (channel ID {text_channel_id}) so text participants can read it."
            )

        message = (
            f"{' | '.join(context_parts)}\n"
            f"{user_name} asked via voice: {request}\n\n"
            f"INSTRUCTIONS:\n"
            f"{instructions}"
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
                item = await session.playback_queue.get()

                # Unpack tuple: (audio_bytes, channel_post_text_or_none)
                if isinstance(item, tuple):
                    wav_bytes, channel_post = item
                else:
                    wav_bytes, channel_post = item, None

                if not vc.is_connected():
                    log.debug("Voice client disconnected, dropping playback")
                    session.playback_queue.task_done()
                    continue

                # Check barge-in before even starting
                if session.tts_cancel.is_set():
                    log.info("Playback skipped — user is speaking (barge-in)")
                    session.playback_queue.task_done()
                    continue

                # Write WAV to a temp file for FFmpeg
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tf.write(wav_bytes)
                    tmp_path = tf.name

                barged_in = False
                try:
                    # Wait for any current playback to finish
                    while vc.is_playing():
                        if session.tts_cancel.is_set():
                            vc.stop()
                            log.info("Barge-in: stopped current playback")
                            barged_in = True
                            break
                        await asyncio.sleep(0.05)

                    if barged_in or session.tts_cancel.is_set():
                        # Drain remaining queue — user interrupted
                        drained = 0
                        while not session.playback_queue.empty():
                            try:
                                session.playback_queue.get_nowait()
                                session.playback_queue.task_done()
                                drained += 1
                            except asyncio.QueueEmpty:
                                break
                        if drained:
                            log.info("Barge-in: drained %d queued audio chunks", drained)
                        continue

                    source = discord.FFmpegPCMAudio(tmp_path)
                    vc.play(source)

                    # Wait for playback to complete, checking barge-in
                    while vc.is_playing():
                        if session.tts_cancel.is_set():
                            vc.stop()
                            log.info("Barge-in: interrupted playback mid-stream")
                            barged_in = True
                            break
                        await asyncio.sleep(0.05)

                    # Post to channel AFTER successful playback (not before)
                    if not barged_in and channel_post:
                        await self._post_to_channel(guild_id, channel_post)

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

    # Cache for resolved display names (user_id → name)
    _name_cache: dict[int, str] = {}

    def _resolve_display_name(self, guild_id: int, user_id: int | str) -> str:
        """Resolve a Discord user ID to their display name (sync, cache-only)."""
        uid = int(user_id)
        if uid in self._name_cache:
            return self._name_cache[uid]
        try:
            guild = self.get_guild(guild_id)
            if guild:
                member = guild.get_member(uid)
                if member:
                    self._name_cache[uid] = member.display_name
                    return member.display_name
        except Exception:
            pass
        return str(user_id)

    async def _resolve_display_name_async(self, guild_id: int, user_id: int | str) -> str:
        """Resolve a Discord user ID to their display name with API fallback."""
        uid = int(user_id)
        # Check cache first
        if uid in self._name_cache:
            return self._name_cache[uid]
        # Try local cache
        try:
            guild = self.get_guild(guild_id)
            if guild:
                member = guild.get_member(uid)
                if member:
                    self._name_cache[uid] = member.display_name
                    return member.display_name
                # Cache miss — fetch from API
                try:
                    member = await guild.fetch_member(uid)
                    if member:
                        self._name_cache[uid] = member.display_name
                        return member.display_name
                except Exception:
                    pass
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
    speech_end_delay_ms: int = 1000,
) -> VoiceBot:
    """Create a configured VoiceBot instance.

    Args:
        pipeline_config:      VoicePipeline configuration.
        guild_ids:            Guild IDs for slash command registration.
        transcript_channel_id: Discord channel ID for transcript posting. None to disable.
        vad_silence_ms:       VAD silence threshold in ms (default 1500).
        vad_min_speech_ms:    VAD minimum speech duration in ms (default 500).
        speech_end_delay_ms:  Silence duration before finalizing speech (default 1000).

    Returns:
        Configured VoiceBot ready to run.
    """
    if not _PYCORD_AVAILABLE:
        raise ImportError("py-cord is required. Install with: pip install 'py-cord[voice]' PyNaCl")

    intents = discord.Intents.default()
    intents.voice_states = True
    intents.members = True
    # Message Content is a privileged intent — must be enabled in Discord
    # Developer Portal. Only request it if tts_read_channel is enabled.
    if pipeline_config and pipeline_config.tts_read_channel:
        intents.message_content = True
        intents.messages = True

    return VoiceBot(
        pipeline_config=pipeline_config,
        guild_ids=guild_ids,
        transcript_channel_id=transcript_channel_id,
        vad_silence_ms=vad_silence_ms,
        vad_min_speech_ms=vad_min_speech_ms,
        speech_end_delay_ms=speech_end_delay_ms,
        intents=intents,
    )
