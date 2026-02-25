# ADR-001: Use Wyoming Protocol for Home Assistant Integration

**Status:** Accepted  
**Date:** 2026-02-24  
**Author:** Liam (Belthanior)

---

## Context

openclaw-voice needs to integrate with Home Assistant's Voice Assist pipeline. Three approaches were evaluated:

1. **Wyoming Protocol** — an open, lightweight TCP socket protocol specifically designed for streaming audio events between Home Assistant and satellite voice services.
2. **Direct HA API** — use Home Assistant's REST/WebSocket API to push transcripts and receive TTS audio directly.
3. **Custom Protocol** — design a bespoke HTTP/WebSocket protocol tailored to this project's requirements.

## Decision

**Use the Wyoming Protocol.**

## Rationale

### Wyoming Protocol (Chosen)
- **Native HA integration**: Wyoming is the first-class protocol for Home Assistant Voice Assist. Satellites declared as Wyoming endpoints appear automatically in the HA UI and work with any HA pipeline (Whisper, Piper, etc.).
- **Streaming support**: Wyoming is designed for bidirectional audio streaming using lightweight TCP events (`AudioChunk`, `AudioStop`, `Transcript`, `Synthesize`). This matches our real-time use case.
- **Active ecosystem**: The `wyoming` Python package is maintained by the HA team. The protocol is used by `wyoming-faster-whisper`, `wyoming-piper`, and others — giving us proven implementation patterns.
- **Low overhead**: Pure TCP, minimal framing. No HTTP request/response overhead on the hot path.
- **Future compatibility**: HA's voice roadmap is built around Wyoming. Adopting it positions openclaw-voice to benefit from upstream improvements automatically.

### Direct HA API (Rejected)
- HA's REST/WebSocket API is designed for home automation, not for streaming voice pipelines.
- Would require polling or complex WebSocket management for audio delivery.
- No standard way to register a custom STT/TTS provider via API.
- Tight coupling to HA's internal data model — breakage risk on HA updates.

### Custom Protocol (Rejected)
- High implementation cost with no upstream benefit.
- No tooling or community support.
- Future integrators would need to learn a non-standard protocol.
- Maintenance burden falls entirely on this project.

## Consequences

- The project depends on the `wyoming` Python package (`>= 1.5.3`).
- All STT and TTS services expose Wyoming TCP servers on their configured ports.
- whisper.cpp and Kokoro are accessed via HTTP internally (wrapped by facades), while Wyoming is the external-facing protocol.
- Speaker ID uses HTTP (not Wyoming) since it is an auxiliary service not directly in the HA voice pipeline.

## Related ADRs

- ADR-002: Use Resemblyzer for Speaker Identification
