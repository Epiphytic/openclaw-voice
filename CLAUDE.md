# CLAUDE.md - openclaw-voice

## Project Overview
Wyoming protocol bridges connecting Home Assistant Voice Assist to local AI models:
- **STT**: whisper.cpp (via `wyoming-stt-bridge`)
- **TTS**: Kokoro (via `wyoming-tts-bridge` and OpenAI-compatible API)
- **Speaker ID**: Resemblyzer (via `speaker-id-server`)

## Architecture
- **Language**: Python 3.10+
- **Style**: Modern Python with type hints (`mypy` strict mode encouraged)
- **Entry Point**: `src/openclaw_voice/cli.py` (CLI via `click`)
- **Config**: TOML-based configuration or env vars (`OPENCLAW_VOICE_*`)

## Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,speaker-id]"
```

### Running
```bash
# Run specific service
openclaw-voice stt --port 10300
openclaw-voice tts --port 10200
openclaw-voice speaker-id --port 8003

# Run all services (threaded)
openclaw-voice all --config configs/config.example.toml
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=openclaw_voice

# Type checking
mypy src
```

### Linting / Formatting
```bash
ruff check .
ruff format .
```

## Structure
- `src/openclaw_voice/`: Source code
  - `stt_bridge.py`: Wyoming → whisper.cpp
  - `tts_bridge.py`: Wyoming → Kokoro
  - `speaker_id.py`: FastAPI server for Resemblyzer
  - `cli.py`: Unified CLI entry point
- `tests/`: Pytest suite
- `configs/`: Example configurations
- `docs/`: Architecture and guides

## Convention
- **Imports**: `from __future__ import annotations` at top of every file.
- **Types**: Use `|` for Union (Python 3.10+ style). Fully type-hint public functions.
- **Logging**: Use `logging.getLogger("openclaw_voice.submodule")`.
- **Async**: Use `asyncio` and `httpx` (async) for I/O bound tasks. STT/TTS bridges are async event handlers.
- **Error Handling**: Catch exceptions broadly at the top level of event loops to prevent bridge crashes.
- **Config**: Prefer CLI args/env vars over hardcoded values. Use `config.toml` for deployment.

## Git
- Main branch: `main`
- Merge strategy: Squash and merge
- Commit messages: Conventional Commits (feat, fix, docs, chore)
