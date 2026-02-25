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
| `VoiceActivityDetector` | `src/openclaw_voice/vad.py` | VAD class: segments 16kHz int16 mono PCM audio into utterances using webrtcvad. Configurable aggressiveness, silence threshold (default 1500ms), and min speech duration (default 500ms). | `webrtcvad` | Y |
| `VoicePipeline` | `src/openclaw_voice/voice_pipeline.py` | Orchestrates STT → LLM → TTS pipeline for Discord voice. Maintains per-channel conversation history. Returns (transcript, response_text, response_audio). Logs per-stage timing (stt_ms, llm_ms, tts_ms, total_ms). | `httpx`, `WhisperFacade`, `KokoroFacade` | N (stateful) |
| `VoiceBot` | `src/openclaw_voice/discord_bot.py` | Discord voice channel bot (py-cord). Slash commands: /join, /leave, /voice. Per-user VAD + VoicePipeline per guild. Per-user Task dispatch with cancellation (new utterance cancels in-flight response). Max response age guard (5s). Transcript posting to a configurable Discord channel. | `py-cord[voice]`, `PyNaCl`, `VoicePipeline`, `VoiceActivityDetector` | N (Long-running) |
| `create_bot()` | `src/openclaw_voice/discord_bot.py` | Factory function for VoiceBot with sane intents defaults. Accepts transcript_channel_id, vad_silence_ms, vad_min_speech_ms. | `py-cord` | Y |

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
