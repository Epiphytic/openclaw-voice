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
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


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
        speaker_id_threshold=speaker_id_threshold
        or cfg_file.get("speaker_id_threshold", 0.75),
        transcript_log=Path(transcript_log) if transcript_log else (
            Path(cfg_file["transcript_log"]) if "transcript_log" in cfg_file else None
        ),
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
@click.option(
    "--port", default=lambda: _env("SPEAKER_ID_PORT", "8003"), type=int, help="Bind port"
)
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

    cfg_file: dict = {}
    if config:
        cfg_file = _load_toml_config(Path(config))

    from openclaw_voice.stt_bridge import STTConfig, run_stt_bridge
    from openclaw_voice.tts_bridge import TTSConfig, run_tts_bridge
    from openclaw_voice.speaker_id import SpeakerIDConfig, run_speaker_id
    import threading

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


if __name__ == "__main__":
    main()
