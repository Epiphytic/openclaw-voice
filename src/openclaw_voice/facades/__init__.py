"""
Facade modules for external service integrations.

Each facade wraps direct calls to external libraries or HTTP APIs, providing
a stable internal interface that can be swapped without touching business logic.

Facades:
  - whisper.py   — wraps whisper.cpp HTTP /inference calls
  - kokoro.py    — wraps Kokoro TTS HTTP /v1/audio/speech calls
  - resemblyzer.py — wraps Resemblyzer model loading and inference

See CODING-STANDARDS.md §9 "Facade Pattern for Externals".
"""
