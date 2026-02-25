"""
openclaw-voice CLI — unified entry point.

Sub-commands:
  stt        Start the Wyoming STT bridge (→ whisper.cpp)
  tts        Start the Wyoming TTS bridge (→ Kokoro)
  speaker-id Start the Speaker ID HTTP server (Resemblyzer)
  all        Start all three services concurrently

Configuration sources (in priority order, highest first):
  1. CLI flags
  2. Environment variables (OPENCLAW_VOICE_*)
  3. TOML config file (--config / OPENCLAW_VOICE_CONFIG)
  4. Built-in defaults

Examples:
  openclaw-voice stt --port 10300 --whisper-url http://localhost:8001/inference
  openclaw-voice tts --port 10200 --kokoro-url http://localhost:8002/v1/audio/speech
  openclaw-voice speaker-id --port 8003 --profiles-dir ./profiles
  openclaw-voice all --config /etc/openclaw-voice/config.toml
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import click

from openclaw_voice.logging_config import setup_logging as _setup_logging_json

log = logging.getLogger("openclaw_voice")


# ---------------------------------------------------------------------------
# TOML config loader (stdlib tomllib on 3.11+, else tomli)
# ---------------------------------------------------------------------------


def _load_toml_config(path: Path) -> dict:
    """Load a TOML config file and return the parsed dict."""
    try:
        if sys.version_info >= (3, 11):
            import tomllib

            return tomllib.loads(path.read_text())
        else:
            import tomli  # type: ignore[import]

            return tomli.loads(path.read_text())
    except FileNotFoundError:
        log.warning("Config file not found: %s", path)
        return {}
    except Exception as exc:
        log.error("Failed to parse config file %s: %s", path, exc)
        return {}


def _load_openclaw_config() -> dict:
    """Load voice config from the main OpenClaw config (~/.openclaw/openclaw.json).

    Looks for a ``voice`` key in the top-level config.  Returns the dict
    or empty dict if not found.
    """
    import json as _json

    candidates = [
        Path.home() / ".openclaw" / "openclaw.json",
        Path(os.environ.get("OPENCLAW_HOME", "")) / "openclaw.json",
    ]
    for p in candidates:
        if p.is_file():
            try:
                data = _json.loads(p.read_text())
                voice_cfg = data.get("voice", {})
                if voice_cfg:
                    log.debug("Loaded voice config from %s", p)
                    return voice_cfg
            except Exception as exc:
                log.debug("Could not read voice config from %s: %s", p, exc)
    return {}


def _env(key: str, default: str | None = None) -> str | None:
    """Read an environment variable with the OPENCLAW_VOICE_ prefix."""
    return os.environ.get(f"OPENCLAW_VOICE_{key.upper()}", default)


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

_config_option = click.option(
    "--config",
    "-c",
    type=click.Path(dir_okay=False),
    default=lambda: _env("CONFIG"),
    help="Path to TOML config file",
    show_default=False,
)

_log_level_option = click.option(
    "--log-level",
    default=lambda: _env("LOG_LEVEL", "INFO"),
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging verbosity",
    show_default=True,
)


def _setup_logging(level: str) -> None:
    """Configure structured JSON logging at the requested level.

    Respects OPENCLAW_VOICE_DEBUG=true for forced debug output.
    """
    _setup_logging_json(level)


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------


@click.group()
@click.version_option()
def main() -> None:
    """openclaw-voice: Wyoming bridges for Home Assistant voice."""


# ---------------------------------------------------------------------------
# stt sub-command
# ---------------------------------------------------------------------------


@main.command("stt")
@_config_option
@_log_level_option
@click.option("--host", default=lambda: _env("STT_HOST", "0.0.0.0"), help="Bind host")
@click.option("--port", default=lambda: _env("STT_PORT", "10300"), type=int, help="Bind port")
@click.option(
    "--whisper-url",
    default=lambda: _env("WHISPER_URL", "http://localhost:8001/inference"),
    help="whisper.cpp /inference endpoint",
)
@click.option(
    "--speaker-id/--no-speaker-id",
    default=False,
    help="Enable speaker identification",
)
@click.option(
    "--speaker-id-url",
    default=lambda: _env("SPEAKER_ID_URL", "http://localhost:8003/identify"),
    help="Speaker ID /identify endpoint",
)
@click.option(
    "--speaker-id-threshold",
    default=0.75,
    type=float,
    help="Cosine similarity threshold for speaker ID",
)
@click.option(
    "--transcript-log",
    default=lambda: _env("TRANSCRIPT_LOG"),
    type=click.Path(dir_okay=False),
    help="Path to JSONL transcript log (optional)",
)
@click.option("--model-name", default="large-v3-turbo", help="Whisper model name (for Describe)")
@click.option("--whisper-timeout", default=30.0, type=float, help="Whisper HTTP timeout (s)")
def stt_cmd(
    config: str | None,
    log_level: str,
    host: str,
    port: int,
    whisper_url: str,
    speaker_id: bool,
    speaker_id_url: str,
    speaker_id_threshold: float,
    transcript_log: str | None,
    model_name: str,
    whisper_timeout: float,
) -> None:
    """Start the Wyoming STT bridge (whisper.cpp backend)."""
    _setup_logging(log_level)

    cfg_file: dict = {}
    if config:
        cfg_file = _load_toml_config(Path(config)).get("stt", {})

    # CLI flags take priority over config file
    from openclaw_voice.stt_bridge import STTConfig, run_stt_bridge

    stt_config = STTConfig(
        host=host or cfg_file.get("host", "0.0.0.0"),
        port=int(port or cfg_file.get("port", 10300)),
        whisper_url=whisper_url or cfg_file.get("whisper_url", "http://localhost:8001/inference"),
        speaker_id_url=speaker_id_url
        or cfg_file.get("speaker_id_url", "http://localhost:8003/identify"),
        enable_speaker_id=speaker_id or cfg_file.get("enable_speaker_id", False),
        speaker_id_threshold=speaker_id_threshold or cfg_file.get("speaker_id_threshold", 0.75),
        transcript_log=Path(transcript_log)
        if transcript_log
        else (Path(cfg_file["transcript_log"]) if "transcript_log" in cfg_file else None),
        model_name=model_name or cfg_file.get("model_name", "large-v3-turbo"),
        whisper_timeout=whisper_timeout or cfg_file.get("whisper_timeout", 30.0),
    )
    asyncio.run(run_stt_bridge(stt_config))


# ---------------------------------------------------------------------------
# tts sub-command
# ---------------------------------------------------------------------------


@main.command("tts")
@_config_option
@_log_level_option
@click.option("--host", default=lambda: _env("TTS_HOST", "0.0.0.0"), help="Bind host")
@click.option("--port", default=lambda: _env("TTS_PORT", "10200"), type=int, help="Bind port")
@click.option(
    "--kokoro-url",
    default=lambda: _env("KOKORO_URL", "http://localhost:8002/v1/audio/speech"),
    help="Kokoro /v1/audio/speech endpoint",
)
@click.option(
    "--default-voice",
    default=lambda: _env("DEFAULT_VOICE", "af_heart"),
    help="Default Kokoro voice (native name or OpenAI alias)",
)
@click.option("--speed", default=1.0, type=float, help="Speech speed (0.5–2.0)")
@click.option("--http-timeout", default=60.0, type=float, help="HTTP timeout for Kokoro (s)")
def tts_cmd(
    config: str | None,
    log_level: str,
    host: str,
    port: int,
    kokoro_url: str,
    default_voice: str,
    speed: float,
    http_timeout: float,
) -> None:
    """Start the Wyoming TTS bridge (Kokoro backend)."""
    _setup_logging(log_level)

    cfg_file: dict = {}
    if config:
        cfg_file = _load_toml_config(Path(config)).get("tts", {})

    from openclaw_voice.tts_bridge import TTSConfig, run_tts_bridge

    tts_config = TTSConfig(
        host=host or cfg_file.get("host", "0.0.0.0"),
        port=int(port or cfg_file.get("port", 10200)),
        kokoro_url=kokoro_url
        or cfg_file.get("kokoro_url", "http://localhost:8002/v1/audio/speech"),
        default_voice=default_voice or cfg_file.get("default_voice", "af_heart"),
        speed=speed or cfg_file.get("speed", 1.0),
        http_timeout=http_timeout or cfg_file.get("http_timeout", 60.0),
    )
    asyncio.run(run_tts_bridge(tts_config))


# ---------------------------------------------------------------------------
# speaker-id sub-command
# ---------------------------------------------------------------------------


@main.command("speaker-id")
@_config_option
@_log_level_option
@click.option("--host", default=lambda: _env("SPEAKER_ID_HOST", "0.0.0.0"), help="Bind host")
@click.option("--port", default=lambda: _env("SPEAKER_ID_PORT", "8003"), type=int, help="Bind port")
@click.option(
    "--profiles-dir",
    default=lambda: _env("PROFILES_DIR", "./speaker-profiles"),
    type=click.Path(file_okay=False),
    help="Directory for speaker profile JSON files",
)
@click.option(
    "--threshold",
    default=0.75,
    type=float,
    help="Cosine similarity threshold for identification",
)
@click.option(
    "--device",
    default=lambda: _env("RESEMBLYZER_DEVICE", "cpu"),
    type=click.Choice(["cpu", "cuda"]),
    help="Device for Resemblyzer inference",
)
def speaker_id_cmd(
    config: str | None,
    log_level: str,
    host: str,
    port: int,
    profiles_dir: str,
    threshold: float,
    device: str,
) -> None:
    """Start the Speaker ID server (Resemblyzer backend)."""
    _setup_logging(log_level)

    cfg_file: dict = {}
    if config:
        cfg_file = _load_toml_config(Path(config)).get("speaker_id", {})

    from openclaw_voice.speaker_id import SpeakerIDConfig, run_speaker_id

    sid_config = SpeakerIDConfig(
        profiles_dir=Path(profiles_dir or cfg_file.get("profiles_dir", "./speaker-profiles")),
        default_threshold=threshold or cfg_file.get("threshold", 0.75),
        device=device or cfg_file.get("device", "cpu"),
    )
    run_speaker_id(
        sid_config,
        host=host or cfg_file.get("host", "0.0.0.0"),
        port=int(port or cfg_file.get("port", 8003)),
    )


# ---------------------------------------------------------------------------
# all sub-command
# ---------------------------------------------------------------------------


@main.command("all")
@_config_option
@_log_level_option
@click.option("--stt-port", default=10300, type=int, help="Wyoming STT port")
@click.option("--tts-port", default=10200, type=int, help="Wyoming TTS port")
@click.option("--speaker-id-port", default=8003, type=int, help="Speaker ID HTTP port")
@click.option(
    "--whisper-url",
    default="http://localhost:8001/inference",
    help="whisper.cpp endpoint",
)
@click.option(
    "--kokoro-url",
    default="http://localhost:8002/v1/audio/speech",
    help="Kokoro endpoint",
)
@click.option(
    "--profiles-dir",
    default="./speaker-profiles",
    type=click.Path(file_okay=False),
    help="Speaker profiles directory",
)
@click.option("--speaker-id/--no-speaker-id", default=False, help="Enable speaker ID in STT")
def all_cmd(
    config: str | None,
    log_level: str,
    stt_port: int,
    tts_port: int,
    speaker_id_port: int,
    whisper_url: str,
    kokoro_url: str,
    profiles_dir: str,
    speaker_id: bool,
) -> None:
    """Start STT bridge, TTS bridge, and Speaker ID server concurrently."""
    _setup_logging(log_level)

    if config:
        _load_toml_config(Path(config))

    import threading

    from openclaw_voice.speaker_id import SpeakerIDConfig, run_speaker_id
    from openclaw_voice.stt_bridge import STTConfig, run_stt_bridge
    from openclaw_voice.tts_bridge import TTSConfig, run_tts_bridge

    stt_cfg = STTConfig(
        port=stt_port,
        whisper_url=whisper_url,
        enable_speaker_id=speaker_id,
        speaker_id_url=f"http://localhost:{speaker_id_port}/identify",
    )
    tts_cfg = TTSConfig(
        port=tts_port,
        kokoro_url=kokoro_url,
    )
    sid_cfg = SpeakerIDConfig(
        profiles_dir=Path(profiles_dir),
    )

    # Run speaker-id in a background thread (uvicorn is synchronous)
    sid_thread = threading.Thread(
        target=run_speaker_id,
        args=(sid_cfg,),
        kwargs={"host": "0.0.0.0", "port": speaker_id_port},
        daemon=True,
    )
    sid_thread.start()

    # Run both Wyoming bridges concurrently in the event loop
    async def _run_all() -> None:
        await asyncio.gather(
            run_stt_bridge(stt_cfg),
            run_tts_bridge(tts_cfg),
        )

    try:
        asyncio.run(_run_all())
    except KeyboardInterrupt:
        log.info("Shutting down")


# ---------------------------------------------------------------------------
# discord-bot sub-command
# ---------------------------------------------------------------------------


@main.command("discord-bot")
@_config_option
@_log_level_option
@click.option(
    "--token",
    default=lambda: _env("DISCORD_TOKEN"),
    help="Discord bot token (or OPENCLAW_VOICE_DISCORD_TOKEN env var)",
    show_default=False,
)
@click.option(
    "--guild-id",
    default=lambda: _env("GUILD_ID"),
    type=str,
    multiple=True,
    help="Guild ID(s) for slash command registration (repeatable; omit for global)",
)
@click.option(
    "--whisper-url",
    default=lambda: _env("WHISPER_URL", "http://localhost:8001/inference"),
    help="whisper.cpp /inference endpoint",
)
@click.option(
    "--kokoro-url",
    default=lambda: _env("KOKORO_URL", "http://localhost:8002/v1/audio/speech"),
    help="Kokoro /v1/audio/speech endpoint",
)
@click.option(
    "--llm-url",
    default=lambda: _env("LLM_URL", "http://localhost:8000/v1/chat/completions"),
    help="OpenAI-compatible LLM /v1/chat/completions endpoint",
)
@click.option(
    "--llm-model",
    default=lambda: _env("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
    help="LLM model name",
)
@click.option(
    "--tts-voice",
    default=lambda: _env("DEFAULT_VOICE", "af_heart"),
    help="Default Kokoro TTS voice",
)
@click.option(
    "--speaker-id/--no-speaker-id",
    default=False,
    help="Enable speaker identification",
)
@click.option(
    "--speaker-id-url",
    default=lambda: _env("SPEAKER_ID_URL", "http://localhost:8003/identify"),
    help="Speaker ID /identify endpoint",
)
@click.option(
    "--max-history",
    default=20,
    type=int,
    help="Max conversation turns to keep per channel",
)
@click.option(
    "--vad-silence-ms",
    default=lambda: int(_env("VAD_SILENCE_MS", "1500")),
    type=int,
    help="VAD silence threshold in ms before flushing an utterance (default 1500)",
)
@click.option(
    "--vad-min-speech-ms",
    default=lambda: int(_env("VAD_MIN_SPEECH_MS", "500")),
    type=int,
    help="VAD minimum speech duration in ms to emit an utterance (default 500)",
)
@click.option(
    "--transcript-channel-id",
    default=lambda: _env("TRANSCRIPT_CHANNEL_ID"),
    type=str,
    help="Discord channel ID for posting conversation transcripts (optional)",
)
def discord_bot_cmd(
    config: str | None,
    log_level: str,
    token: str | None,
    guild_id: tuple[str, ...],
    whisper_url: str,
    kokoro_url: str,
    llm_url: str,
    llm_model: str,
    tts_voice: str,
    speaker_id: bool,
    speaker_id_url: str,
    max_history: int,
    vad_silence_ms: int,
    vad_min_speech_ms: int,
    transcript_channel_id: str | None,
) -> None:
    """Start the Discord voice channel bot."""
    _setup_logging(log_level)

    cfg_file: dict = {}
    if config:
        cfg_file = _load_toml_config(Path(config)).get("discord", {})

    # Also check OpenClaw config for a [voice] section as fallback
    if not cfg_file:
        openclaw_cfg = _load_openclaw_config()
        if openclaw_cfg:
            cfg_file = openclaw_cfg

    # Resolve token (CLI > env > config file)
    resolved_token = token or cfg_file.get("token")
    if not resolved_token:
        import click as _click

        raise _click.ClickException(
            "Discord bot token is required. Pass --token or set OPENCLAW_VOICE_DISCORD_TOKEN."
        )

    # Resolve guild IDs
    resolved_guild_ids: list[int] = []
    if guild_id:
        resolved_guild_ids = [int(g) for g in guild_id]
    elif cfg_file.get("guild_id"):
        raw = cfg_file["guild_id"]
        resolved_guild_ids = [int(raw)] if isinstance(raw, (str, int)) else [int(g) for g in raw]

    from openclaw_voice.discord_bot import (
        DEFAULT_VAD_MIN_SPEECH_MS,
        DEFAULT_VAD_SILENCE_MS,
        PipelineConfig,
        create_bot,
    )

    # Config file values override CLI defaults (CLI explicit flags still win
    # via env vars, but when --config is given, its values take precedence
    # over hardcoded click defaults).
    def _resolve(cli_val, key: str, fallback=None):
        """Config file wins over CLI default; explicit CLI/env wins over config."""
        cfg_val = cfg_file.get(key)
        if cfg_val is not None:
            return cfg_val
        return cli_val if cli_val is not None else fallback

    pipeline_config = PipelineConfig(
        whisper_url=_resolve(whisper_url, "whisper_url", "http://localhost:8001/inference"),
        kokoro_url=_resolve(kokoro_url, "kokoro_url", "http://localhost:8002/v1/audio/speech"),
        llm_url=_resolve(llm_url, "llm_url", "http://localhost:8000/v1/chat/completions"),
        llm_model=_resolve(llm_model, "llm_model", "Qwen/Qwen2.5-32B-Instruct"),
        tts_voice=_resolve(tts_voice, "tts_voice", "af_heart"),
        enable_speaker_id=_resolve(speaker_id, "enable_speaker_id", False),
        speaker_id_url=_resolve(speaker_id_url, "speaker_id_url", "http://localhost:8003/identify"),
        max_history_turns=_resolve(max_history, "max_history_turns", 20),
        # Identity & context from config file (not hardcoded)
        bot_name=cfg_file.get("bot_name", "Assistant"),
        main_agent_name=cfg_file.get("main_agent_name", "main agent"),
        default_location=cfg_file.get("default_location", ""),
        default_timezone=cfg_file.get("default_timezone", "UTC"),
        extra_context=cfg_file.get("extra_context", ""),
        whisper_prompt=cfg_file.get("whisper_prompt", ""),
        corrections_file=cfg_file.get("corrections_file", ""),
        channel_context_messages=int(cfg_file.get("channel_context_messages", 10)),
        tts_read_channel=bool(cfg_file.get("tts_read_channel", True)),
    )

    # Resolve VAD and transcript settings (CLI > env > config file > defaults)
    resolved_vad_silence_ms: int = vad_silence_ms or int(
        cfg_file.get("vad_silence_ms", DEFAULT_VAD_SILENCE_MS)
    )
    resolved_vad_min_speech_ms: int = vad_min_speech_ms or int(
        cfg_file.get("vad_min_speech_ms", DEFAULT_VAD_MIN_SPEECH_MS)
    )
    resolved_speech_end_delay_ms: int = int(
        cfg_file.get("speech_end_delay_ms", 1000)
    )
    raw_channel_id = transcript_channel_id or cfg_file.get("transcript_channel_id")
    resolved_transcript_channel_id: int | None = int(raw_channel_id) if raw_channel_id else None

    bot = create_bot(
        pipeline_config=pipeline_config,
        guild_ids=resolved_guild_ids if resolved_guild_ids else None,
        transcript_channel_id=resolved_transcript_channel_id,
        vad_silence_ms=resolved_vad_silence_ms,
        vad_min_speech_ms=resolved_vad_min_speech_ms,
        speech_end_delay_ms=resolved_speech_end_delay_ms,
    )

    log.info(
        "Starting Discord voice bot",
        extra={
            "guild_ids": resolved_guild_ids,
            "llm_model": pipeline_config.llm_model,
            "tts_voice": pipeline_config.tts_voice,
            "vad_silence_ms": resolved_vad_silence_ms,
            "vad_min_speech_ms": resolved_vad_min_speech_ms,
            "speech_end_delay_ms": resolved_speech_end_delay_ms,
            "transcript_channel_id": resolved_transcript_channel_id,
        },
    )

    bot.run(resolved_token)


if __name__ == "__main__":
    main()
