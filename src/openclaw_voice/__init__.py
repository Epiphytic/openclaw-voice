"""
openclaw-voice — Wyoming protocol bridges for Home Assistant voice.

Components:
- stt_bridge: Wyoming STT → whisper.cpp
- tts_bridge: Wyoming TTS → Kokoro (OpenAI-compatible)
- speaker_id: Speaker identification via Resemblyzer (FastAPI server)
"""

__version__ = "0.1.0"
__all__ = ["stt_bridge", "tts_bridge", "speaker_id", "cli"]
