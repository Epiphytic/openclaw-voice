# openclaw-voice

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Real-time Discord voice assistant with fully local AI processing. Also includes Wyoming protocol bridges for Home Assistant voice integration.

**Privacy-first**: all speech recognition, language model inference, and text-to-speech run locally — no audio or text leaves your machine.

## Features

### Discord Voice Bot
- 🎤 **Local STT** — whisper.cpp with GPU acceleration (12x realtime)
- 🧠 **Local LLM** — any OpenAI-compatible model (Qwen, GLM, etc.)
- 🗣️ **Local TTS** — Kokoro-82M for natural speech synthesis (5x realtime)
- 🔧 **Tool calling** — weather, time, web search handled directly
- 🔀 **Escalation** — complex requests route to your main OpenClaw agent seamlessly
- 💬 **Text-to-voice bridge** — text channel messages read aloud to voice participants
- 👤 **Speaker identification** — optional Resemblyzer-based speaker recognition
- ⚡ **Barge-in** — interrupt the bot mid-speech by talking
- 📝 **Conversation context** — maintains multi-turn history per channel
- 🔤 **Whisper vocabulary hints** — bias transcription toward project-specific terms
- 🔄 **Word corrections** — post-STT replacement dictionary for commonly misheard words

### Wyoming Protocol Bridges (Home Assistant)
- **STT bridge** — connects Home Assistant to whisper.cpp
- **TTS bridge** — connects Home Assistant to Kokoro TTS
- **Speaker ID server** — identifies speakers via Resemblyzer

## Installation

### Option 1: OpenClaw Plugin (recommended)

Install as an OpenClaw plugin for managed process lifecycle, native escalation, and agent tools:

```bash
# From npm (when published)
openclaw plugins install @epiphytic/discord-voice

# From local checkout (development)
openclaw plugins install --link ./plugin
```

Then configure in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "discord-voice": {
        "enabled": true,
        "config": {
          "botToken": "your-discord-bot-token",
          "guildIds": [123456789012345678],
          "botName": "Assistant",
          "llmUrl": "http://localhost:8000/v1/chat/completions",
          "llmModel": "Qwen/Qwen3-30B-A3B-Instruct-2507",
          "whisperUrl": "http://localhost:8001/inference",
          "kokoroUrl": "http://localhost:8002/v1/audio/speech",
          "defaultLocation": "Your City, Country",
          "defaultTimezone": "America/Vancouver",
          "whisperPrompt": "unusual names, project terms, place names"
        }
      }
    }
  }
}
```

Restart the gateway: `openclaw gateway restart`

### Option 2: Standalone

```bash
git clone https://github.com/Epiphytic/openclaw-voice.git
cd openclaw-voice
python -m venv .venv && source .venv/bin/activate
pip install ".[discord]"

# Copy and edit the config
cp config.example.toml config.toml
# Edit config.toml with your settings

# Run the Discord voice bot
openclaw-voice discord-bot --config config.toml --token YOUR_BOT_TOKEN
```

### Wyoming Bridges (Home Assistant)

```bash
pip install ".[speaker-id]"

# Run all bridges
openclaw-voice all --config config.toml

# Or individually
openclaw-voice stt --port 10300
openclaw-voice tts --port 10200
openclaw-voice speaker-id --port 8003
```

## Discord Bot Setup

1. Create a bot at [Discord Developer Portal](https://discord.com/developers/applications)
2. Enable these **Privileged Gateway Intents**:
   - **Server Members Intent** — for building member roster
   - **Message Content Intent** — for text-to-voice bridge
3. Enable **Voice** in the bot permissions
4. Invite the bot to your server with appropriate permissions

## Configuration Reference

All options can be set via TOML config (standalone), JSON plugin config (OpenClaw), CLI flags, or environment variables.

### Core Settings

| TOML Key | Plugin Key | Description | Default |
|----------|-----------|-------------|---------|
| `token` | `botToken` | Discord bot token | *(required)* |
| `guild_ids` | `guildIds` | Server IDs (array) | `[]` |
| `transcript_channel_id` | `transcriptChannelId` | Channel for posting transcripts | *(none)* |

### Model Endpoints

| TOML Key | Plugin Key | Description | Default |
|----------|-----------|-------------|---------|
| `llm_url` | `llmUrl` | OpenAI-compatible LLM endpoint | `http://localhost:8000/v1/chat/completions` |
| `llm_model` | `llmModel` | LLM model name | `Qwen/Qwen2.5-32B-Instruct` |
| `whisper_url` | `whisperUrl` | whisper.cpp endpoint | `http://localhost:8001/inference` |
| `kokoro_url` | `kokoroUrl` | Kokoro TTS endpoint | `http://localhost:8002/v1/audio/speech` |
| `tts_voice` | `ttsVoice` | TTS voice name | `af_heart` |

### Identity & Context

| TOML Key | Plugin Key | Description | Default |
|----------|-----------|-------------|---------|
| `bot_name` | `botName` | Bot's display name in conversations | `Assistant` |
| `main_agent_name` | `mainAgentName` | How the bot refers to its escalation target | `main agent` |
| `default_location` | `defaultLocation` | Location for weather queries | *(empty)* |
| `default_timezone` | `defaultTimezone` | Timezone for time queries | `UTC` |
| `extra_context` | `extraContext` | Freeform text appended to system prompt | *(empty)* |

### Speech Processing

| TOML Key | Plugin Key | Description | Default |
|----------|-----------|-------------|---------|
| `whisper_prompt` | `whisperPrompt` | Vocabulary hints for Whisper (comma-separated names, terms, places) | *(empty)* |
| `speech_end_delay_ms` | `speechEndDelayMs` | Silence duration (ms) before finalizing speech | `1000` |
| `vad_min_speech_ms` | `vadMinSpeechMs` | Minimum speech duration (ms) to process | `500` |
| `corrections_file` | `correctionsFile` | Path to TOML word corrections file | *(none)* |

### Channel Features

| TOML Key | Plugin Key | Description | Default |
|----------|-----------|-------------|---------|
| `channel_context_messages` | `channelContextMessages` | Number of recent text messages injected as LLM context | `10` |
| `tts_read_channel` | `ttsReadChannel` | Read text channel messages aloud in voice | `true` |

### Advanced

| TOML Key | Plugin Key | Description | Default |
|----------|-----------|-------------|---------|
| `control_port` | `controlPort` | HTTP port for health/control server | `18790` |

### Whisper Vocabulary Hints

The `whisper_prompt` field biases Whisper's transcription toward specific vocabulary. This is especially useful for:
- **People's names** that Whisper mishears (e.g., "Bel" → "Bell", "Bill")
- **Project names** (GIRT, OpenClaw)
- **Technical terms** (vLLM, ROCm, WASM)
- **Local place names** (Cassidy, Ladysmith)

Example:
```toml
whisper_prompt = "Liam, Bel, Chip, GIRT, OpenClaw, Cassidy, Vancouver Island"
```

### Word Corrections

For words that Whisper consistently gets wrong even with vocabulary hints, use a corrections file:

```toml
# corrections.toml
[corrections]
"Bell" = "Bel"
"Bill" = "Bel"
"Gurt" = "GIRT"
```

## Architecture

The bot runs as a single Python process with independent async workers:

- **STT Worker** — receives audio from Discord, sends to whisper.cpp
- **LLM Worker** — processes transcripts, calls tools or generates responses
- **TTS Worker** — synthesizes responses via Kokoro
- **Playback Worker** — streams audio back to Discord voice channel
- **Escalation Worker** — routes complex queries to the main OpenClaw agent

The bot appears as a **single seamless entity** to users — escalation to the main agent is transparent.

## Requirements

- **Python 3.10+**
- **Local model servers**: whisper.cpp (STT), any OpenAI-compatible LLM, Kokoro (TTS)
- **Discord bot** with Voice, Message Content, and Server Members intents
- **GPU recommended** for STT (Vulkan) and LLM (ROCm/CUDA)

## Development

```bash
pip install -e ".[dev,discord,speaker-id]"
pytest
ruff check .
```

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.
