# @epiphytic/openclaw-discord-voice

OpenClaw plugin for real-time Discord voice assistant with fully local AI processing.

## Features

- **Voice conversations** — join Discord voice channels and interact via speech
- **Local STT** — whisper.cpp with GPU acceleration
- **Local LLM** — any OpenAI-compatible model (Qwen, GLM, etc.)
- **Local TTS** — Kokoro-82M for natural speech synthesis
- **Tool calling** — weather, time, web search
- **Escalation** — complex requests route to the main OpenClaw agent seamlessly
- **Text-to-voice bridge** — text channel messages read aloud in voice
- **Speaker identification** — optional Resemblyzer-based speaker recognition

## Requirements

- Python 3.11+ with py-cord, whisper.cpp, Kokoro TTS
- Local LLM server (vLLM, llama.cpp, etc.) with OpenAI-compatible API
- Discord bot with Voice, Message Content, and Server Members intents

## Install

```bash
openclaw plugins install @epiphytic/openclaw-discord-voice
```

## Configuration

```json5
{
  plugins: {
    entries: {
      "discord-voice": {
        enabled: true,
        config: {
          botToken: "your-discord-bot-token",
          guildIds: [123456789],
          botName: "Assistant",
          llmUrl: "http://localhost:8000/v1/chat/completions",
          whisperUrl: "http://localhost:8001/inference",
          kokoroUrl: "http://localhost:8002/v1/audio/speech"
        }
      }
    }
  }
}
```

See the [repository](https://github.com/Epiphytic/openclaw-voice) for full configuration options and setup guide.

## License

MIT
