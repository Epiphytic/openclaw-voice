"""
Structured JSON logging configuration for openclaw-voice.

Usage:
    from openclaw_voice.logging_config import setup_logging

    setup_logging()          # uses OPENCLAW_VOICE_LOG_LEVEL env var (default: INFO)
    setup_logging("DEBUG")   # explicit level
    setup_logging(debug=True)# force DEBUG (e.g. from OPENCLAW_VOICE_DEBUG=true)

Set OPENCLAW_VOICE_DEBUG=true to enable verbose debug logging.
Set OPENCLAW_VOICE_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR to control verbosity.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # Include any extra fields attached to the record
        _standard_fields = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in _standard_fields and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


def setup_logging(
    level: str | None = None,
    *,
    debug: bool | None = None,
) -> None:
    """Configure structured JSON logging for the openclaw_voice package.

    Priority:
      1. ``debug=True`` kwarg → force DEBUG level
      2. ``level`` argument (explicit)
      3. ``OPENCLAW_VOICE_DEBUG=true`` env var → DEBUG
      4. ``OPENCLAW_VOICE_LOG_LEVEL`` env var
      5. Default: INFO

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR). Case-insensitive.
        debug: If True, force DEBUG level regardless of env vars.
    """
    # Resolve effective level
    if debug is True:
        effective_level = logging.DEBUG
    elif level is not None:
        effective_level = getattr(logging, level.upper(), logging.INFO)
    elif os.environ.get("OPENCLAW_VOICE_DEBUG", "").lower() in ("1", "true", "yes"):
        effective_level = logging.DEBUG
    else:
        env_level = os.environ.get("OPENCLAW_VOICE_LOG_LEVEL", "INFO").upper()
        effective_level = getattr(logging, env_level, logging.INFO)

    formatter = _JSONFormatter()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    # Configure root logger for the package
    pkg_logger = logging.getLogger("openclaw_voice")
    pkg_logger.setLevel(effective_level)
    # Avoid duplicate handlers if called multiple times
    if not pkg_logger.handlers:
        pkg_logger.addHandler(handler)
    else:
        pkg_logger.handlers[0] = handler

    # Suppress noisy third-party loggers unless in DEBUG mode
    if effective_level > logging.DEBUG:
        for noisy in ("httpx", "httpcore", "uvicorn.access"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    pkg_logger.debug(
        "Logging initialised",
        extra={"log_level": logging.getLevelName(effective_level)},
    )
