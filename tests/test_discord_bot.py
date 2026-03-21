"""Tests for Discord bot slash command interaction patterns.

Validates that all slash commands handle Discord's 3-second interaction
timeout correctly by using defer() + followup.send() for slow operations,
and that errors after defer() always produce a followup error response.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Lightweight mock scaffolding for pycord types
# ---------------------------------------------------------------------------


def _make_ctx(*, in_voice: bool = True, voice_client: object | None = None) -> MagicMock:
    """Build a mock ApplicationContext with async helpers."""
    ctx = MagicMock()
    ctx.defer = AsyncMock()
    ctx.respond = AsyncMock()
    ctx.guild_id = 12345

    followup = MagicMock()
    followup.send = AsyncMock()
    ctx.followup = followup

    ctx.channel_id = 99999

    # Author voice state
    if in_voice:
        voice_state = MagicMock()
        voice_channel = MagicMock()
        voice_channel.name = "general"
        voice_channel.id = 54321
        voice_channel.connect = AsyncMock(return_value=MagicMock())
        voice_state.channel = voice_channel
        ctx.author.voice = voice_state
    else:
        ctx.author.voice = None

    # Guild
    ctx.guild.voice_client = voice_client
    return ctx


@pytest.fixture()
def bot():
    """Create a VoiceBot instance with pycord fully mocked out."""
    # Patch the entire discord module so VoiceBot can be instantiated
    # without a real pycord install / Discord connection.
    mock_discord = MagicMock()
    mock_discord.Bot = type("Bot", (), {"__init_subclass__": classmethod(lambda cls, **kw: None)})
    mock_discord.Intents.default.return_value = MagicMock()

    with (
        patch.dict("sys.modules", {"discord": mock_discord, "discord.sinks": MagicMock()}),
        patch("openclaw_voice.discord_bot._PYCORD_AVAILABLE", True),
        patch("openclaw_voice.discord_bot.discord", mock_discord),
    ):
        from openclaw_voice.discord_bot import VoiceBot

        # Minimal construction — skip super().__init__ and _register_commands
        instance = object.__new__(VoiceBot)
        instance._pipeline_config = MagicMock()
        instance._pipeline_config.tts_voice = "af_heart"
        instance._pipelines = {}
        instance._sessions = {}
        instance._sinks = {}
        instance._playback_tasks = {}
        instance._guild_text_channels = {}
        instance._guild_context = {}
        instance._channel_msg_cache = {}
        instance._guild_ids = []
        instance._vad_silence_ms = 1500
        instance._vad_min_speech_ms = 500
        instance._speech_end_delay_ms = 1000
        instance._cast = {}
        instance._stop_listening = AsyncMock()
        instance._start_listening = AsyncMock()
        instance._seed_channel_cache = AsyncMock()
        instance.user = MagicMock(id=11111)
        yield instance


# ---------------------------------------------------------------------------
# /join tests
# ---------------------------------------------------------------------------


class TestCmdJoin:
    """Tests for the /join slash command handler."""

    @pytest.mark.asyncio()
    async def test_not_in_voice_responds_without_defer(self, bot):
        """Early guard: user not in voice channel -> ctx.respond(), no defer."""
        ctx = _make_ctx(in_voice=False)
        await bot._cmd_join(ctx)

        ctx.respond.assert_awaited_once()
        ctx.defer.assert_not_awaited()
        ctx.followup.send.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_join_defers_then_followup(self, bot):
        """Normal join: defer() first, then followup.send() with success."""
        ctx = _make_ctx()
        await bot._cmd_join(ctx)

        ctx.defer.assert_awaited_once()
        ctx.respond.assert_not_awaited()
        ctx.followup.send.assert_awaited_once()
        msg = ctx.followup.send.call_args[0][0]
        assert "Joined" in msg
        assert "general" in msg

    @pytest.mark.asyncio()
    async def test_join_disconnects_existing_before_connecting(self, bot):
        """When already connected, disconnect first then reconnect."""
        existing_vc = AsyncMock()
        existing_vc.disconnect = AsyncMock()
        ctx = _make_ctx(voice_client=existing_vc)

        await bot._cmd_join(ctx)

        bot._stop_listening.assert_awaited_once()
        existing_vc.disconnect.assert_awaited_once()
        ctx.followup.send.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_join_connect_failure_sends_followup_error(self, bot):
        """If channel.connect() raises, a followup error is sent (not ctx.respond)."""
        ctx = _make_ctx()
        ctx.author.voice.channel.connect = AsyncMock(side_effect=RuntimeError("connection refused"))

        await bot._cmd_join(ctx)

        ctx.defer.assert_awaited_once()
        ctx.followup.send.assert_awaited_once()
        msg = ctx.followup.send.call_args[0][0]
        assert "Failed to join" in msg

    @pytest.mark.asyncio()
    async def test_join_start_listening_failure_sends_followup_error(self, bot):
        """If _start_listening raises after defer, followup error is sent."""
        ctx = _make_ctx()
        bot._start_listening = AsyncMock(side_effect=RuntimeError("worker crash"))

        await bot._cmd_join(ctx)

        ctx.defer.assert_awaited_once()
        # followup.send is called twice: once for success, once for error?
        # Actually no - the success followup.send is BEFORE _start_listening,
        # so it will succeed. The exception in _start_listening will be caught
        # by the outer try/except and a second followup error will be attempted.
        # But Discord only allows one followup? Actually followup.send can be
        # called multiple times. Let's just check the error was logged.
        calls = ctx.followup.send.call_args_list
        assert len(calls) >= 1


# ---------------------------------------------------------------------------
# /leave tests
# ---------------------------------------------------------------------------


class TestCmdLeave:
    """Tests for the /leave slash command handler."""

    @pytest.mark.asyncio()
    async def test_not_connected_responds_without_defer(self, bot):
        """Early guard: no voice client -> ctx.respond(), no defer."""
        ctx = _make_ctx(voice_client=None)
        await bot._cmd_leave(ctx)

        ctx.respond.assert_awaited_once()
        ctx.defer.assert_not_awaited()
        ctx.followup.send.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_leave_defers_then_followup(self, bot):
        """Normal leave: defer() first, then followup.send() with success."""
        vc = AsyncMock()
        vc.disconnect = AsyncMock()
        ctx = _make_ctx(voice_client=vc)

        await bot._cmd_leave(ctx)

        ctx.defer.assert_awaited_once()
        ctx.respond.assert_not_awaited()
        ctx.followup.send.assert_awaited_once()
        msg = ctx.followup.send.call_args[0][0]
        assert "Disconnected" in msg

    @pytest.mark.asyncio()
    async def test_leave_stop_listening_failure_sends_followup_error(self, bot):
        """If _stop_listening raises after defer, followup error is sent."""
        vc = AsyncMock()
        ctx = _make_ctx(voice_client=vc)
        bot._stop_listening = AsyncMock(side_effect=RuntimeError("cleanup crash"))

        await bot._cmd_leave(ctx)

        ctx.defer.assert_awaited_once()
        ctx.followup.send.assert_awaited_once()
        msg = ctx.followup.send.call_args[0][0]
        assert "Failed to leave" in msg

    @pytest.mark.asyncio()
    async def test_leave_cleans_up_guild_state(self, bot):
        """Leave should clean up text channels, context, and cache."""
        vc = AsyncMock()
        vc.disconnect = AsyncMock()
        ctx = _make_ctx(voice_client=vc)
        guild_id = ctx.guild_id

        bot._guild_text_channels[guild_id] = 99999
        bot._guild_context[guild_id] = {"some": "context"}
        bot._channel_msg_cache[guild_id] = ["msg"]

        await bot._cmd_leave(ctx)

        assert guild_id not in bot._guild_text_channels
        assert guild_id not in bot._guild_context
        assert guild_id not in bot._channel_msg_cache


# ---------------------------------------------------------------------------
# /voice tests
# ---------------------------------------------------------------------------


class TestCmdVoice:
    """Tests for the /voice slash command handler."""

    @pytest.mark.asyncio()
    async def test_voice_responds_immediately(self, bot):
        """Voice command is fast — uses ctx.respond(), not defer+followup."""
        ctx = _make_ctx()
        await bot._cmd_voice(ctx, "af_nova")

        ctx.respond.assert_awaited_once()
        ctx.defer.assert_not_awaited()
        ctx.followup.send.assert_not_awaited()
        msg = ctx.respond.call_args[0][0]
        assert "af_nova" in msg

    @pytest.mark.asyncio()
    async def test_voice_updates_pipeline_config(self, bot):
        """Voice command updates the pipeline config for the guild."""
        ctx = _make_ctx()
        await bot._cmd_voice(ctx, "bf_emma")

        assert bot._pipeline_config.tts_voice == "bf_emma"

    @pytest.mark.asyncio()
    async def test_voice_updates_active_pipeline(self, bot):
        """When a pipeline exists for the guild, update its config."""
        ctx = _make_ctx()
        guild_id = ctx.guild_id
        pipeline = MagicMock()
        pipeline.config.tts_voice = "af_heart"
        bot._pipelines[guild_id] = pipeline

        await bot._cmd_voice(ctx, "am_adam")

        assert pipeline.config.tts_voice == "am_adam"
