"""
Voice Activity Detection (VAD) using webrtcvad with RMS energy gating.

Accepts PCM audio frames (20ms chunks at 16kHz, 16-bit mono) and tracks
speech / silence state. When enough silence follows a speech segment, the
accumulated audio is returned as a complete utterance byte buffer.

An RMS energy threshold is applied *before* webrtcvad: frames below the
threshold are treated as silence regardless of what webrtcvad thinks.
This filters out low-level background noise (fans, farm ambience, etc.)
that can fool webrtcvad into thinking speech is ongoing.

Usage::

    detector = VoiceActivityDetector(aggressiveness=3, silence_threshold_ms=1500)
    for chunk in audio_frames:          # 20ms PCM chunks
        utterance = detector.process(chunk)
        if utterance is not None:
            # complete utterance ready for STT
            process(utterance)
"""

from __future__ import annotations

import logging
import struct

import webrtcvad  # type: ignore[import]

log = logging.getLogger("openclaw_voice.vad")

# Expected frame parameters
SAMPLE_RATE = 16_000  # Hz
SAMPLE_WIDTH = 2  # bytes (int16)
CHANNELS = 1
FRAME_DURATION_MS = 20  # ms per frame
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * SAMPLE_WIDTH  # 640 bytes


def _frame_rms(frame: bytes) -> float:
    """Compute the RMS (root-mean-square) amplitude of a 16-bit PCM frame.

    Returns a value roughly in 0–32768. Typical speech is 1000–10000;
    quiet room background noise is 50–500.
    """
    n_samples = len(frame) // SAMPLE_WIDTH
    if n_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{n_samples}h", frame)
    sum_sq = sum(s * s for s in samples)
    return (sum_sq / n_samples) ** 0.5


class VoiceActivityDetector:
    """Stateful VAD that segments audio into discrete utterances.

    Args:
        aggressiveness:       webrtcvad aggressiveness level 0–3 (default 3).
        silence_threshold_ms: Milliseconds of continuous silence after speech
                              that triggers an utterance flush (default 1500 ms).
                              Increased from 800 ms to prevent mid-sentence splits.
        min_speech_ms:        Minimum speech duration to emit an utterance
                              (default 500 ms). Filters breath sounds and false
                              triggers. Raised from 100 ms for production use.
        rms_threshold:        Minimum RMS amplitude for a frame to be considered
                              speech (default 300). Frames below this are forced
                              to silence before webrtcvad runs. Filters background
                              noise (fans, farm ambience). Typical speech: 1000+.

    Each call to :meth:`process` accepts exactly one 20 ms PCM frame.
    When a complete utterance is detected it returns the raw PCM bytes
    (all frames from utterance start to end), otherwise ``None``.
    """

    def __init__(
        self,
        aggressiveness: int = 3,
        silence_threshold_ms: int = 1500,
        min_speech_ms: int = 500,
        rms_threshold: int = 300,
    ) -> None:
        if aggressiveness not in range(4):
            raise ValueError(f"aggressiveness must be 0–3, got {aggressiveness}")

        self._vad = webrtcvad.Vad(aggressiveness)
        self._aggressiveness = aggressiveness
        self._silence_threshold_ms = silence_threshold_ms
        self._min_speech_ms = min_speech_ms
        self._rms_threshold = rms_threshold

        # Frames of silence needed to trigger utterance flush
        self._silence_frames_needed = max(1, silence_threshold_ms // FRAME_DURATION_MS)
        # Minimum speech frames required
        self._min_speech_frames = max(1, min_speech_ms // FRAME_DURATION_MS)

        # State
        self._speaking: bool = False
        self._speech_frames: list[bytes] = []  # frames collected during speech
        self._silence_count: int = 0  # consecutive silent frames seen

        log.debug(
            "VoiceActivityDetector initialised",
            extra={
                "aggressiveness": aggressiveness,
                "silence_threshold_ms": silence_threshold_ms,
                "silence_frames_needed": self._silence_frames_needed,
                "rms_threshold": rms_threshold,
            },
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def aggressiveness(self) -> int:
        """Current webrtcvad aggressiveness level."""
        return self._aggressiveness

    @property
    def silence_threshold_ms(self) -> int:
        """Silence duration (ms) needed to flush an utterance."""
        return self._silence_threshold_ms

    @property
    def is_speaking(self) -> bool:
        """True while speech is in progress (utterance not yet flushed)."""
        return self._speaking

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def process(self, frame: bytes) -> bytes | None:
        """Process one 20ms PCM frame and return an utterance if complete.

        Args:
            frame: Exactly ``FRAME_SIZE`` (640) bytes of 16kHz int16 mono PCM.

        Returns:
            Concatenated PCM bytes of the complete utterance, or ``None`` if
            the utterance is still ongoing / no speech detected yet.

        Raises:
            ValueError: If ``frame`` is not exactly ``FRAME_SIZE`` bytes.
        """
        if len(frame) != FRAME_SIZE:
            raise ValueError(
                f"Expected {FRAME_SIZE}-byte frame, got {len(frame)}. "
                "Frames must be 20ms at 16kHz int16 mono."
            )

        # RMS energy gate: reject frames below threshold as silence
        rms = _frame_rms(frame)

        # Log RMS every 50 frames (~1s) for tuning
        if not hasattr(self, "_rms_log_counter"):
            self._rms_log_counter = 0
            self._rms_max_since_log = 0.0
            self._rms_min_since_log = 99999.0
        self._rms_log_counter += 1
        self._rms_max_since_log = max(self._rms_max_since_log, rms)
        self._rms_min_since_log = min(self._rms_min_since_log, rms)
        if self._rms_log_counter >= 50:
            log.debug(
                "RMS stats (1s window)",
                extra={
                    "rms_min": round(self._rms_min_since_log, 1),
                    "rms_max": round(self._rms_max_since_log, 1),
                    "rms_current": round(rms, 1),
                    "threshold": self._rms_threshold,
                    "speaking": self._speaking,
                },
            )
            self._rms_log_counter = 0
            self._rms_max_since_log = 0.0
            self._rms_min_since_log = 99999.0

        if rms < self._rms_threshold:
            is_speech = False
        else:
            try:
                is_speech = self._vad.is_speech(frame, SAMPLE_RATE)
            except Exception as exc:
                log.warning("webrtcvad error on frame: %s", exc)
                is_speech = False

        if is_speech:
            if not self._speaking:
                log.debug("Speech started")
            self._speaking = True
            self._silence_count = 0
            self._speech_frames.append(frame)
            return None

        # Silent frame
        if self._speaking:
            self._speech_frames.append(frame)  # include trailing silence in buffer
            self._silence_count += 1

            if self._silence_count >= self._silence_frames_needed:
                return self._flush()

        return None

    def _flush(self) -> bytes | None:
        """Flush accumulated speech frames as an utterance.

        Trims trailing silence (beyond one extra frame) before returning.
        Returns ``None`` if utterance is shorter than ``min_speech_ms``.
        """
        # Strip trailing silence (keep one frame for naturalness)
        keep = max(0, len(self._speech_frames) - self._silence_count + 1)
        frames_to_emit = self._speech_frames[:keep]

        # Reset state
        self._speaking = False
        self._speech_frames = []
        self._silence_count = 0

        if len(frames_to_emit) < self._min_speech_frames:
            log.debug(
                "Utterance too short (%d frames), discarding",
                len(frames_to_emit),
            )
            return None

        utterance = b"".join(frames_to_emit)
        duration_ms = len(frames_to_emit) * FRAME_DURATION_MS
        log.debug(
            "Utterance flushed",
            extra={"duration_ms": duration_ms, "bytes": len(utterance)},
        )
        return utterance

    def flush(self) -> bytes | None:
        """Force-flush any buffered speech (e.g. on mute).

        Returns the accumulated utterance bytes, or None if nothing buffered.
        """
        if not self._speech_frames:
            return None
        utterance = b"".join(self._speech_frames)
        duration_ms = len(self._speech_frames) * FRAME_DURATION_MS
        self._speaking = False
        self._speech_frames = []
        self._silence_count = 0
        if duration_ms < self._min_speech_ms:
            log.debug("Flush discarded — too short (%d ms)", duration_ms)
            return None
        log.info(
            "Utterance force-flushed",
            extra={"duration_ms": duration_ms, "bytes": len(utterance)},
        )
        return utterance

    def reset(self) -> None:
        """Reset state, discarding any buffered speech."""
        self._speaking = False
        self._speech_frames = []
        self._silence_count = 0
        log.debug("VoiceActivityDetector reset")
