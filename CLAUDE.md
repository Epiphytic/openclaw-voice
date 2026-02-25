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
  - `cli.py`: Unified CLI entry point
  - `stt_bridge.py`: Wyoming → whisper.cpp (via `facades/whisper.py`)
  - `tts_bridge.py`: Wyoming → Kokoro (via `facades/kokoro.py`)
  - `speaker_id.py`: FastAPI server for Resemblyzer (via `facades/resemblyzer.py`)
  - `logging_config.py`: Structured JSON logging setup
  - `facades/`: External service facades (whisper, kokoro, resemblyzer)
- `tests/`: Pytest suite
- `configs/`: Example configurations
- `docs/`: Architecture, guides, plans, and ADRs
  - `docs/adrs/`: Architectural Decision Records
  - `docs/plans/`: Implementation plans (HITL gate before execution)
- `MANIFEST.md`: Module & integration registry — update when adding modules

## Convention
- **Imports**: `from __future__ import annotations` at top of every file.
- **Types**: Use `|` for Union (Python 3.10+ style). Fully type-hint public functions.
- **Logging**: Use `logging.getLogger("openclaw_voice.submodule")`. All logging is structured JSON — use `setup_logging()` from `openclaw_voice.logging_config`, never `logging.basicConfig` directly. Set `OPENCLAW_VOICE_DEBUG=true` (in `.env.local`) for verbose output.
- **Async**: Use `asyncio` and `httpx` (async) for I/O bound tasks. STT/TTS bridges are async event handlers.
- **Error Handling**: Catch exceptions broadly at the top level of event loops to prevent bridge crashes.
- **Config**: Prefer CLI args/env vars over hardcoded values. Use `config.toml` for deployment. See `.env.example` for all supported `OPENCLAW_VOICE_*` variables.
- **Facade Pattern**: Never call external services (whisper.cpp, Kokoro, Resemblyzer) directly. Use the facades in `src/openclaw_voice/facades/`. This enables backend swapping. See CODING-STANDARDS.md §9.
- **Idempotency**: Scripts and functions must be safe to run twice. Avoid creating duplicate resources.
- **Circuit Breakers**: Halt and escalate after 3 consecutive failures on the same task. Do not loop indefinitely.

## Manifest
`MANIFEST.md` at the project root is the single source of truth for all modules, facades, and external integrations. **Update it whenever you create a new module, facade, or external integration.** This prevents agents from duplicating work.

## Plans & Decisions
- **`docs/plans/`** — Write concrete implementation plans here before starting complex work. Plans require explicit approval (HITL gate) before execution.
- **`docs/adrs/`** — Record architectural decisions (ADRs) when evaluating design options. Use ADR-NNN-short-title.md format.

## Git
- Main branch: `main`
- Merge strategy: Squash and merge
- **Commits**: Atomic, conventional commit messages (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`). One logical change per commit. No "big dump" commits.

## Coding Standards
Full standards are in `~/openclaw/workspace/CODING-STANDARDS.md`. Key rules for this project:
- Structured JSON logging via `logging_config.py`
- Facade pattern for all external service calls
- Idempotent scripts and functions
- Circuit breakers: escalate after 3 consecutive failures
- Atomic conventional commits
