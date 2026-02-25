# openclaw-voice

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Wyoming protocol bridges for local Home Assistant voice:

- **STT**: Connects Home Assistant to **whisper.cpp** server
- **TTS**: Connects Home Assistant to **Kokoro** TTS server (via OpenAI-compatible API)
- **Speaker ID**: Identifies speakers via **Resemblyzer** for personalized responses

This project allows you to run high-quality, privacy-focused voice AI entirely locally, integrating seamlessly with Home Assistant's Voice Assist pipeline.

## Features

- 🎤 **Wyoming STT Bridge**: Fast, accurate speech-to-text via `whisper.cpp`.
- 🗣️ **Wyoming TTS Bridge**: Natural-sounding text-to-speech via `Kokoro` (OpenAI API compatible).
- 👤 **Speaker Identification**: Recognizes who is speaking to personalize responses (e.g., "Welcome back, Liam").
- 🚀 **Unified CLI**: Run all services with a single command or individually.
- 🐳 **Docker-ready**: Designed for containerized deployment (Dockerfile coming soon).

## Installation

Requires Python 3.10 or higher.

```bash
# Clone the repository
git clone https://github.com/epiphytic/openclaw-voice.git
cd openclaw-voice

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (including speaker-id support)
pip install ".[speaker-id]"
```

## Quick Start

1. **Start the backend servers** (whisper.cpp and Kokoro):
   Ensure `whisper.cpp` is running on port 8001 and Kokoro on port 8002.
   *(See their respective docs for setup instructions)*.

2. **Run openclaw-voice**:
   ```bash
   # Start all bridges with default settings
   openclaw-voice all \
     --whisper-url http://localhost:8001/inference \
     --kokoro-url http://localhost:8002/v1/audio/speech \
     --profiles-dir ./profiles
   ```

3. **Configure Home Assistant**:
   - Go to **Settings > Devices & Services > Add Integration**.
   - Search for **Wyoming Protocol**.
   - Add two Wyoming integrations:
     - **STT**: `0.0.0.0` (or host IP), port `10300`
     - **TTS**: `0.0.0.0` (or host IP), port `10200`
   - In your Voice Assistant pipeline settings, select these new providers.

## Configuration

You can configure `openclaw-voice` via CLI arguments, environment variables, or a TOML config file.

### CLI Example
```bash
openclaw-voice stt --port 10300 --speaker-id
openclaw-voice tts --port 10200 --speed 1.1
```

### Config File Example
Create `config.toml`:
```toml
[stt]
port = 10300
whisper_url = "http://whisper:8001/inference"
enable_speaker_id = true

[tts]
port = 10200
kokoro_url = "http://kokoro:8002/v1/audio/speech"
default_voice = "af_heart"
```

Run with: `openclaw-voice all --config config.toml`

## Speaker Identification

To enable speaker ID:
1. Ensure `openclaw-voice speaker-id` is running (port 8003).
2. Enroll speakers using the API or CLI tools (see [docs/speaker-enrollment.md](docs/speaker-enrollment.md)).
3. Enable speaker ID in the STT bridge configuration.

When a known speaker is detected, transcripts sent to Home Assistant will be prefixed with metadata:
`[speaker:Liam|conf:0.95] Turn on the lights.`

You can use this metadata in your LLM prompt to personalize interactions.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
