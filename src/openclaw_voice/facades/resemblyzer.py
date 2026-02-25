"""
Facade for Resemblyzer voice encoder (speaker identification).

Wraps all direct Resemblyzer calls (model loading and inference) so that the
speaker ID module depends only on this facade — not on the ``resemblyzer``
package directly. Swapping to a different speaker embedding backend (e.g.
SpeechBrain ECAPA-TDNN, a cloud API) only requires updating this module.

See ADR-002: Use Resemblyzer for Speaker Identification.

The VoiceEncoder is a lazy-loaded process-level singleton.  Loading it once
and reusing it avoids repeated model initialisation overhead across requests.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("openclaw_voice.facades.resemblyzer")

# Process-level singleton — None until first use
_encoder: Any = None


def get_encoder(device: str = "cpu") -> Any:
    """Return the lazy-loaded Resemblyzer VoiceEncoder singleton.

    The encoder is initialised on first call and cached for the lifetime
    of the process.  Thread-safe in CPython due to the GIL; for multi-process
    deployments, each worker initialises its own copy.

    Args:
        device: Inference device ("cpu" or "cuda").

    Returns:
        A ``resemblyzer.VoiceEncoder`` instance.

    Raises:
        ImportError: If the ``resemblyzer`` package is not installed.
    """
    global _encoder
    if _encoder is None:
        log.info(
            "Loading Resemblyzer VoiceEncoder",
            extra={"device": device},
        )
        from resemblyzer import VoiceEncoder  # type: ignore[import]

        _encoder = VoiceEncoder(device)
        log.info("Resemblyzer VoiceEncoder ready", extra={"device": device})
    return _encoder


def preprocess(audio: np.ndarray, source_sr: int = 16000) -> np.ndarray:
    """Apply Resemblyzer's standard audio pre-processing pipeline.

    Args:
        audio:     Mono float32 numpy array.
        source_sr: Source sample rate (Hz); will be resampled to 16 kHz if needed.

    Returns:
        Pre-processed numpy array ready for embedding.

    Raises:
        ImportError: If the ``resemblyzer`` package is not installed.
    """
    from resemblyzer import preprocess_wav  # type: ignore[import]

    return preprocess_wav(audio, source_sr=source_sr)


def embed_utterance(encoder: Any, processed_audio: np.ndarray) -> np.ndarray:
    """Compute a speaker embedding (d-vector) for the given audio.

    Args:
        encoder:         A ``VoiceEncoder`` instance (from ``get_encoder()``).
        processed_audio: Pre-processed audio array (from ``preprocess()``).

    Returns:
        256-dimensional float32 embedding vector.
    """
    embedding: np.ndarray = encoder.embed_utterance(processed_audio)
    log.debug(
        "Computed speaker embedding",
        extra={"embedding_shape": list(embedding.shape)},
    )
    return embedding
