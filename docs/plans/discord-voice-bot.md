# Plan: Discord Voice Channel Bot

## Goal
Build a Discord bot that joins voice channels and provides interactive voice conversation — listen to users, transcribe, send to LLM, synthesize response, play back audio.

## Architecture

```
Discord Voice Channel
  ↓ (opus audio per user)
Voice Bot (pycord)
  ↓ (PCM → WAV)
  ├─→ whisper.cpp (:8001) → transcript
  ├─→ Speaker ID (:8003) → who's talking (parallel)
  ↓
LLM (Qwen :8000, OpenAI-compatible) → response text
  ↓
Kokoro TTS (:8002) → response audio (WAV)
  ↓ (WAV → opus)
Discord Voice Channel (playback)
```

## Components

### 1. Voice Receive
- Use `py-cord` (discord.ext.voice) with audio receive
- Per-user audio streams (Discord provides separate streams per speaker)
- Voice Activity Detection (VAD) using `webrtcvad` or silence detection
- Buffer audio until speech ends, then process

### 2. Processing Pipeline
- Convert opus → PCM → WAV (16kHz mono for whisper.cpp)
- Parallel: STT + Speaker ID
- Send transcript + speaker context to LLM
- LLM generates response
- TTS generates audio
- Play audio back via Discord voice client

### 3. Conversation Context
- Maintain per-channel conversation history
- Include speaker names in context
- System prompt awareness of who's talking

### 4. Commands
- `/join` — bot joins user's voice channel
- `/leave` — bot leaves
- `/voices` — list available TTS voices
- `/voice <name>` — set bot's speaking voice
- Auto-join when someone enters the configured voice channel (optional)

## Technical Decisions
- **py-cord over discord.py**: py-cord maintained voice receive support
- **webrtcvad for VAD**: Already installed as resemblyzer dep, fast C extension
- **ffmpeg for audio conversion**: Already installed on belthanior
- **Existing facades**: Use WhisperFacade, KokoroFacade from openclaw-voice

## Files to Create
- `src/openclaw_voice/discord_bot.py` — Main bot module
- `src/openclaw_voice/voice_pipeline.py` — Audio processing pipeline
- `src/openclaw_voice/vad.py` — Voice activity detection
- `tests/test_voice_pipeline.py` — Unit tests
- `tests/test_vad.py` — VAD tests

## Dependencies to Add
- py-cord[voice] (Discord API + voice support)
- PyNaCl (voice encryption)
- webrtcvad (voice activity detection)
