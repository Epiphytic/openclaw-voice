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
import io
import logging
import os
import queue
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

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
    Sink = object   # type: ignore[assignment,misc]

from openclaw_voice.vad import FRAME_SIZE, FRAME_DURATION_MS, VoiceActivityDetector
from openclaw_voice.voice_pipeline import PipelineConfig, VoicePipeline

# Sample rates
DISCORD_SAMPLE_RATE = 48_000   # Discord sends 48kHz stereo opus/PCM
WHISPER_SAMPLE_RATE = 16_000   # whisper.cpp requires 16kHz mono
DISCORD_CHANNELS = 2
DISCORD_SAMPLE_WIDTH = 2       # int16

# Max size of the response playback queue
PLAYBACK_QUEUE_MAXSIZE = 10


# ---------------------------------------------------------------------------
# VAD Sink (pycord voice receive)
# ---------------------------------------------------------------------------

class VoiceSink(Sink):  # type: ignore[misc]
    """Custom pycord Sink that feeds per-user audio through VAD.

    When a complete utterance is detected for a user, it is placed on
    the shared ``utterance_queue`` for async processing.

    Args:
        utterance_queue: asyncio.Queue for (user_id, pcm_bytes) tuples.
        loop:            The running asyncio event loop.
    """

    def __init__(
        self,
        utterance_queue: asyncio.Queue,  # type: ignore[type-arg]
        loop: asyncio.AbstractEventLoop,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._utterance_queue = utterance_queue
        self._loop = loop
        # Per-user VAD instances keyed by Discord user ID
        self._vad_instances: dict[int, VoiceActivityDetector] = {}
        # Per-user raw frame buffers (to handle partial frames)
        self._buffers: dict[int, bytes] = {}

    def _get_vad(self, user_id: int) -> VoiceActivityDetector:
        """Get or create a VAD instance for a user."""
        if user_id not in self._vad_instances:
            self._vad_instances[user_id] = VoiceActivityDetector(
                aggressiveness=3,
                silence_threshold_ms=800,
            )
            log.debug("Created VAD for user %s", user_id)
        return self._vad_instances[user_id]

    def write(self, data: bytes, user: int) -> None:  # type: ignore[override]
        """Called by pycord for each audio chunk from a user.

        pycord provides raw 48kHz stereo int16 PCM. We need to downsample
        to 16kHz mono before feeding to webrtcvad.
        """
        # Accumulate data in per-user buffer
        buf = self._buffers.get(user, b"") + data
        self._buffers[user] = buf

        # Downsample 48kHz stereo → 16kHz mono PCM
        try:
            mono_pcm = _resample_48k_stereo_to_16k_mono(buf)
        except Exception as exc:
            log.debug("Audio resample error for user %s: %s", user, exc)
            return

        # Clear the buffer (we consumed it all)
        self._buffers[user] = b""

        # Feed 20ms frames to VAD
        vad = self._get_vad(user)
        offset = 0
        while offset + FRAME_SIZE <= len(mono_pcm):
            frame = mono_pcm[offset : offset + FRAME_SIZE]
            offset += FRAME_SIZE
            try:
                utterance = vad.process(frame)
                if utterance is not None:
                    # Schedule utterance delivery on the event loop
                    self._loop.call_soon_threadsafe(
                        self._utterance_queue.put_nowait,
                        (user, utterance),
                    )
            except Exception as exc:
                log.warning("VAD error for user %s: %s", user, exc)

    def cleanup(self) -> None:  # type: ignore[override]
        """Clean up all VAD state when recording stops."""
        self._vad_instances.clear()
        self._buffers.clear()
        log.debug("VoiceSink cleaned up")


# ---------------------------------------------------------------------------
# Voice Bot
# ---------------------------------------------------------------------------

class VoiceBot(discord.Bot if _PYCORD_AVAILABLE else object):  # type: ignore[misc]
    """Discord bot that listens in voice channels and responds via TTS.

    Args:
        pipeline_config: VoicePipeline configuration.
        guild_ids:       List of guild IDs to register slash commands for.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig | None = None,
        guild_ids: list[int] | None = None,
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

        # Per-guild voice pipelines (one per active voice channel)
        self._pipelines: dict[int, VoicePipeline] = {}

        # Per-guild utterance queues and sink references
        self._utterance_queues: dict[int, asyncio.Queue] = {}  # type: ignore[type-arg]
        self._processing_tasks: dict[int, asyncio.Task] = {}  # type: ignore[type-arg]
        self._playback_queues: dict[int, asyncio.Queue] = {}  # type: ignore[type-arg]
        self._playback_tasks: dict[int, asyncio.Task] = {}  # type: ignore[type-arg]

        self._register_commands()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_ready(self) -> None:
        log.info("VoiceBot ready", extra={"user": str(self.user)})

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

    async def _start_listening(self, guild_id: int, vc: discord.VoiceClient) -> None:
        """Start recording and processing audio in the voice channel."""
        # Create pipeline for this guild
        self._pipelines[guild_id] = VoicePipeline(
            config=self._pipeline_config,
            channel_id=str(guild_id),
        )

        # Create queues
        utterance_q: asyncio.Queue = asyncio.Queue()  # type: ignore[type-arg]
        playback_q: asyncio.Queue = asyncio.Queue(maxsize=PLAYBACK_QUEUE_MAXSIZE)  # type: ignore[type-arg]
        self._utterance_queues[guild_id] = utterance_q
        self._playback_queues[guild_id] = playback_q

        # Start the voice sink
        loop = asyncio.get_event_loop()
        sink = VoiceSink(utterance_q, loop)
        vc.start_recording(sink, self._on_recording_finished, guild_id)

        # Start processing and playback tasks
        self._processing_tasks[guild_id] = asyncio.create_task(
            self._process_utterances(guild_id, vc)
        )
        self._playback_tasks[guild_id] = asyncio.create_task(
            self._playback_worker(guild_id, vc)
        )

        log.info("Started listening in guild %s", guild_id)

    async def _stop_listening(self, guild_id: int) -> None:
        """Stop recording and clean up resources for a guild."""
        # Cancel processing task
        for tasks in (self._processing_tasks, self._playback_tasks):
            task = tasks.pop(guild_id, None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Clean up pipelines and queues
        self._pipelines.pop(guild_id, None)
        self._utterance_queues.pop(guild_id, None)
        self._playback_queues.pop(guild_id, None)

        log.info("Stopped listening in guild %s", guild_id)

    def _on_recording_finished(self, sink: VoiceSink, guild_id: int) -> None:
        """Callback when pycord stops recording (e.g. bot disconnect)."""
        sink.cleanup()
        log.debug("Recording finished for guild %s", guild_id)

    async def _process_utterances(
        self,
        guild_id: int,
        vc: discord.VoiceClient,
    ) -> None:
        """Process utterances from the queue and schedule playback."""
        pipeline = self._pipelines.get(guild_id)
        playback_q = self._playback_queues.get(guild_id)

        if pipeline is None or playback_q is None:
            log.error("Pipeline or playback queue missing for guild %s", guild_id)
            return

        log.debug("Utterance processor started for guild %s", guild_id)
        utterance_q = self._utterance_queues[guild_id]

        while True:
            try:
                user_id, pcm_bytes = await utterance_q.get()
                log.debug(
                    "Processing utterance",
                    extra={"guild_id": guild_id, "user_id": user_id},
                )

                # Run synchronous pipeline in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                response_text, response_audio = await loop.run_in_executor(
                    None,
                    pipeline.process_utterance,
                    pcm_bytes,
                    str(user_id),
                )

                if response_audio:
                    try:
                        playback_q.put_nowait(response_audio)
                    except asyncio.QueueFull:
                        log.warning(
                            "Playback queue full for guild %s, dropping response",
                            guild_id,
                        )

                utterance_q.task_done()

            except asyncio.CancelledError:
                log.debug("Utterance processor cancelled for guild %s", guild_id)
                break
            except Exception as exc:
                log.error(
                    "Error processing utterance for guild %s: %s",
                    guild_id,
                    exc,
                )

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
    mono_samples = [
        (stereo[i * 2] + stereo[i * 2 + 1]) // 2 for i in range(n_samples)
    ]

    # Decimate by 3 (48kHz / 3 = 16kHz)
    decimated = mono_samples[::3]
    return struct.pack(f"<{len(decimated)}h", *decimated)


# ---------------------------------------------------------------------------
# Bot factory
# ---------------------------------------------------------------------------


def create_bot(
    pipeline_config: PipelineConfig | None = None,
    guild_ids: list[int] | None = None,
) -> "VoiceBot":
    """Create a configured VoiceBot instance.

    Args:
        pipeline_config: VoicePipeline configuration.
        guild_ids:       Guild IDs for slash command registration.

    Returns:
        Configured VoiceBot ready to run.
    """
    if not _PYCORD_AVAILABLE:
        raise ImportError(
            "py-cord is required. Install with: pip install 'py-cord[voice]' PyNaCl"
        )

    intents = discord.Intents.default()
    intents.voice_states = True

    return VoiceBot(
        pipeline_config=pipeline_config,
        guild_ids=guild_ids,
        intents=intents,
    )
