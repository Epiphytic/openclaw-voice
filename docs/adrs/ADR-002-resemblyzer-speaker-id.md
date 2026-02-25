# ADR-002: Use Resemblyzer for Speaker Identification

**Status:** Accepted  
**Date:** 2026-02-24  
**Author:** Belthanior

---

## Context

openclaw-voice needs speaker identification to tag transcripts with a speaker name and access level. This enables Home Assistant automations to behave differently based on *who* is speaking (e.g., trusted household members vs. guests).

Three approaches were evaluated:

1. **Resemblyzer** — a Python library providing a pretrained `VoiceEncoder` (d-vector / generalized end-to-end model) that embeds utterances into a 256-dimensional speaker space.
2. **SpeechBrain x-vector / ECAPA-TDNN** — heavier models from the SpeechBrain framework, typically used in professional speaker diarisation pipelines.
3. **Cloud API (e.g., Azure Speaker Recognition, AWS Transcribe)** — managed services that handle speaker enrollment and identification server-side.

## Decision

**Use Resemblyzer for speaker identification.**

## Rationale

### Resemblyzer (Chosen)
- **Local and private**: All inference runs on the home server. No audio ever leaves the network — critical for a home voice assistant.
- **Lightweight**: The pretrained `VoiceEncoder` is a small LSTM model (~17 MB). Inference takes <100ms on CPU, making it viable on a mid-range home server without a GPU.
- **Good enough for small household scale**: For 2–10 enrolled speakers, cosine similarity on d-vectors provides reliable identification (typically >90% accuracy with ≥3 enroll samples per speaker).
- **Simple API**: `encoder.embed_utterance(wav)` returns a numpy array. No complex pipeline setup.
- **Averaging strategy**: Multiple enrollment samples can be averaged for better accuracy over time, which fits the incremental enrollment model (`/enroll` endpoint).
- **Permissive license**: MIT licensed.

### SpeechBrain ECAPA-TDNN (Rejected)
- Significantly larger model footprint (hundreds of MB).
- More complex setup: requires SpeechBrain, which pulls in a large dependency tree.
- Overkill for a household with <10 speakers — marginal accuracy gains don't justify the overhead.
- Slower inference on CPU; typically requires GPU for real-time use.

### Cloud API (Rejected)
- Privacy non-starter: voice audio would leave the home network.
- Adds cloud dependency and cost.
- Latency is unpredictable (network round-trip on every identification).
- Breaks functionality when internet is unavailable (exactly when local voice control matters most).

## Consequences

- `resemblyzer` and `soundfile` are added as optional dependencies (extra group `[speaker-id]`).
- `librosa` is an optional soft dependency for audio resampling; if absent, only 16 kHz WAV input is accepted.
- Speaker profiles are stored as JSON files on disk (one per speaker) containing the 256-dim embedding as a list. This is simple, portable, and human-inspectable.
- The `VoiceEncoder` is a lazy-loaded singleton (loaded on first request, then cached for the process lifetime) to avoid repeated model load overhead.
- The facade in `src/openclaw_voice/facades/resemblyzer.py` wraps all direct Resemblyzer calls, enabling future backend swaps (e.g., to ECAPA-TDNN or a cloud API) without touching business logic.
- Identification confidence is exposed in the API response so callers can tune the threshold.

## Related ADRs

- ADR-001: Use Wyoming Protocol for Home Assistant Integration
