# Agent & Module Manifest

**Purpose:** This file acts as the single source of truth for all coding agents. It records the existing agents, available internal modules, and external integrations to prevent duplication of effort and maintain architectural consistency.
**Rule:** Agents MUST update this registry whenever a new reusable module, agent role, or external facade is created.

---

## 🤖 1. Active Agent Teams & Roles
*Defines the specialized subagents currently configured for this project to prevent spawning redundant agents.*

| Agent Name | Primary Role | Trigger Condition | Capabilities / Scopes |
| :--- | :--- | :--- | :--- |
| *(None currently active)* | | | |

---

## 🧩 2. Module & Function Registry
*A directory of small, reusable modules. Always check here before writing a new utility function.*

| Module / Function | File Path | Description | Dependencies | Idempotent (Y/N) |
| :--- | :--- | :--- | :--- | :--- |
| `setup_logging()` | `src/openclaw_voice/logging_config.py` | Configures structured JSON logging with dynamic levels based on env vars. | `logging`, `json` | Y |
| `main()` | `src/openclaw_voice/cli.py` | Unified CLI entry point for STT, TTS, and Speaker ID services. | `click` | Y |
| `run_stt_bridge()` | `src/openclaw_voice/stt_bridge.py` | Runs the Wyoming STT bridge (whisper.cpp backend). | `wyoming`, `httpx` | N (Long-running) |
| `run_tts_bridge()` | `src/openclaw_voice/tts_bridge.py` | Runs the Wyoming TTS bridge (Kokoro backend). | `wyoming`, `httpx` | N (Long-running) |
| `run_speaker_id()` | `src/openclaw_voice/speaker_id.py` | Runs the Speaker ID HTTP server (Resemblyzer backend). | `fastapi`, `uvicorn` | N (Long-running) |

---

## 🔌 3. External Integrations (Facades)
*Registry of wrapped external libraries and APIs. Never call external libraries directly; use these facades.*

| Facade Name | File Path | Wrapped Library/API | Purpose |
| :--- | :--- | :--- | :--- |
| `WhisperFacade` | `src/openclaw_voice/facades/whisper.py` | `whisper.cpp` HTTP API | Wraps `/inference` calls to decouple from the specific backend API. |
| `KokoroFacade` | `src/openclaw_voice/facades/kokoro.py` | `Kokoro` HTTP API | Wraps `/v1/audio/speech` calls for TTS synthesis. |
| `resemblyzer` | `src/openclaw_voice/facades/resemblyzer.py` | `resemblyzer` | Wraps model loading (`get_encoder`) and inference (`embed_utterance`). |

---

## 📝 4. Global State & Conventions
* **Log Location:** Stdout/Stderr (Structured JSON).
* **Debug Flag:** Set `OPENCLAW_VOICE_DEBUG=true` in `.env.local` (Never commit this file).
* **Config:** `.env.example` shows all supported environment variables.
* **Architecture:** Wyoming protocol for STT/TTS; HTTP for Speaker ID.
