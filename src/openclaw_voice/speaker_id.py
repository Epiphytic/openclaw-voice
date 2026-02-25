"""
Speaker Identification Server — Resemblyzer backend.

Exposes a FastAPI HTTP server that matches voice embeddings against enrolled
speaker profiles. Used by the STT bridge to tag transcripts with speaker names.

Endpoints:
  POST   /identify         — Identify speaker from audio (returns name + confidence)
  POST   /enroll           — Enroll / update a speaker (name + audio sample)
  GET    /speakers         — List enrolled speakers
  DELETE /speakers/{name}  — Remove a speaker profile
  GET    /health           — Health + model status

Speaker profiles are stored as JSON files on disk (one per speaker) containing
a 256-dimensional Resemblyzer embedding. Multiple enroll calls average embeddings
for better accuracy.

Usage:
    from openclaw_voice.speaker_id import create_app, SpeakerIDConfig
    import uvicorn

    config = SpeakerIDConfig(
        profiles_dir=Path("./profiles"),
        default_threshold=0.75,
    )
    app = create_app(config)
    uvicorn.run(app, host="0.0.0.0", port=8003)
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    pass

log = logging.getLogger("openclaw_voice.speaker_id")

# Minimum audio lengths
MIN_IDENTIFY_SAMPLES = 1600   # ~0.1s at 16kHz
MIN_ENROLL_SAMPLES = 8000     # ~0.5s at 16kHz
TARGET_SAMPLE_RATE = 16000


@dataclass
class SpeakerIDConfig:
    """Configuration for the speaker identification server."""

    # Where to persist speaker profiles (JSON files)
    profiles_dir: Path = Path("./speaker-profiles")

    # Cosine similarity threshold for a positive identification
    default_threshold: float = 0.75

    # Device for Resemblyzer ("cpu" or "cuda")
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Voice encoder — lazy-loaded singleton (shared across requests)
# ---------------------------------------------------------------------------

_encoder: Any = None  # resemblyzer.VoiceEncoder


def _get_encoder(device: str = "cpu") -> Any:
    """Return a lazy-loaded Resemblyzer VoiceEncoder."""
    global _encoder
    if _encoder is None:
        log.info("Loading Resemblyzer voice encoder (device=%s)...", device)
        from resemblyzer import VoiceEncoder  # type: ignore[import]

        _encoder = VoiceEncoder(device)
        log.info("Voice encoder ready")
    return _encoder


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------


def load_profiles(profiles_dir: Path) -> dict[str, dict]:
    """Load all speaker profiles from disk into a dict keyed by speaker name."""
    profiles: dict[str, dict] = {}
    for path in profiles_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            data["embedding"] = np.array(data["embedding"], dtype=np.float32)
            profiles[data["name"]] = data
        except Exception as exc:
            log.warning("Failed to load profile %s: %s", path, exc)
    return profiles


def save_profile(
    profiles_dir: Path,
    name: str,
    embedding: np.ndarray,
    metadata: dict | None = None,
) -> None:
    """Write a speaker profile JSON to disk."""
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile: dict[str, Any] = {
        "name": name,
        "embedding": embedding.tolist(),
        "enrolled_at": time.time(),
        "metadata": metadata or {},
    }
    path = profiles_dir / f"{name.lower().replace(' ', '_')}.json"
    path.write_text(json.dumps(profile, indent=2))
    log.info("Saved speaker profile '%s' → %s", name, path)


def audio_to_array(content: bytes) -> np.ndarray:
    """Decode uploaded audio bytes to a 16kHz mono float32 numpy array."""
    buf = io.BytesIO(content)
    try:
        audio, sr = sf.read(buf)
    except Exception:
        try:
            import librosa  # type: ignore[import]

            buf.seek(0)
            audio, sr = librosa.load(buf, sr=TARGET_SAMPLE_RATE, mono=True)
            return audio.astype(np.float32)
        except ImportError:
            raise ValueError(
                "Could not decode audio — install 'librosa' for broader format support"
            )

    # Mono conversion
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != TARGET_SAMPLE_RATE:
        try:
            import librosa  # type: ignore[import]

            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        except ImportError:
            raise ValueError(
                f"Audio sample rate is {sr} Hz but {TARGET_SAMPLE_RATE} Hz required — "
                "install 'librosa' to enable automatic resampling"
            )

    return audio.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_app(config: SpeakerIDConfig) -> FastAPI:
    """Create and return the Speaker ID FastAPI application."""

    app = FastAPI(
        title="Speaker ID (openclaw-voice)",
        description="Speaker identification via Resemblyzer voice embeddings.",
        version="0.1.0",
    )

    # Ensure profiles directory exists at startup
    config.profiles_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Routes
    # ---------------------------------------------------------------------------

    @app.post("/identify")
    async def identify(
        file: UploadFile = File(...),
        threshold: float = Form(config.default_threshold),
    ) -> JSONResponse:
        """Identify the speaker in an audio file.

        Returns the best match if cosine similarity ≥ threshold.
        Access levels are taken from the enrolled profile metadata.
        """
        start = time.time()
        content = await file.read()
        if not content:
            raise HTTPException(400, "Empty file")

        encoder = _get_encoder(config.device)

        try:
            audio = audio_to_array(content)
        except Exception as exc:
            raise HTTPException(400, f"Could not decode audio: {exc}") from exc

        from resemblyzer import preprocess_wav  # type: ignore[import]

        processed = preprocess_wav(audio, source_sr=TARGET_SAMPLE_RATE)
        if len(processed) < MIN_IDENTIFY_SAMPLES:
            raise HTTPException(400, "Audio too short for speaker identification (need ≥0.1s)")

        embedding: np.ndarray = encoder.embed_utterance(processed)

        profiles = load_profiles(config.profiles_dir)
        elapsed = round(time.time() - start, 3)

        if not profiles:
            return JSONResponse(
                {
                    "speaker": None,
                    "confidence": 0.0,
                    "access_level": "basic",
                    "message": "No speakers enrolled",
                    "elapsed": elapsed,
                }
            )

        best_name: str | None = None
        best_score = -1.0
        for name, profile in profiles.items():
            score = cosine_similarity(embedding, profile["embedding"])
            if score > best_score:
                best_score = score
                best_name = name

        elapsed = round(time.time() - start, 3)

        if best_score >= threshold and best_name:
            access_level = (
                profiles[best_name].get("metadata", {}).get("access_level", "standard")
            )
            return JSONResponse(
                {
                    "speaker": best_name,
                    "confidence": round(best_score, 4),
                    "access_level": access_level,
                    "elapsed": elapsed,
                }
            )
        else:
            return JSONResponse(
                {
                    "speaker": None,
                    "confidence": round(best_score, 4),
                    "best_guess": best_name,
                    "access_level": "basic",
                    "elapsed": elapsed,
                }
            )

    @app.post("/enroll")
    async def enroll(
        name: str = Form(...),
        access_level: str = Form("standard"),
        file: UploadFile = File(...),
    ) -> JSONResponse:
        """Enroll or update a speaker.

        Provide name, access_level (full|standard|basic), and an audio sample.
        Multiple calls average embeddings for better accuracy.

        access_level options:
          - full     — trusted household member (e.g. can run automations)
          - standard — recognised guest
          - basic    — fallback / unknown
        """
        content = await file.read()
        if not content:
            raise HTTPException(400, "Empty file")

        encoder = _get_encoder(config.device)

        try:
            audio = audio_to_array(content)
        except Exception as exc:
            raise HTTPException(400, f"Could not decode audio: {exc}") from exc

        from resemblyzer import preprocess_wav  # type: ignore[import]

        processed = preprocess_wav(audio, source_sr=TARGET_SAMPLE_RATE)
        if len(processed) < MIN_ENROLL_SAMPLES:
            raise HTTPException(400, "Audio too short — need at least 0.5 seconds")

        new_embedding: np.ndarray = encoder.embed_utterance(processed)

        profiles = load_profiles(config.profiles_dir)
        if name in profiles:
            existing_emb: np.ndarray = profiles[name]["embedding"]
            sample_count: int = profiles[name].get("metadata", {}).get("sample_count", 1)
            # Weighted average then re-normalise
            averaged = (existing_emb * sample_count + new_embedding) / (sample_count + 1)
            averaged = (averaged / np.linalg.norm(averaged)).astype(np.float32)
            save_profile(
                config.profiles_dir,
                name,
                averaged,
                {"access_level": access_level, "sample_count": sample_count + 1},
            )
            return JSONResponse(
                {
                    "status": "updated",
                    "speaker": name,
                    "samples": sample_count + 1,
                    "message": f"Averaged {sample_count + 1} samples for '{name}'",
                }
            )
        else:
            save_profile(
                config.profiles_dir,
                name,
                new_embedding,
                {"access_level": access_level, "sample_count": 1},
            )
            return JSONResponse(
                {
                    "status": "enrolled",
                    "speaker": name,
                    "samples": 1,
                    "message": (
                        f"Enrolled '{name}' with 1 sample. "
                        "Add more samples for better accuracy."
                    ),
                }
            )

    @app.get("/speakers")
    async def list_speakers() -> JSONResponse:
        """List all enrolled speakers."""
        profiles = load_profiles(config.profiles_dir)
        speakers = [
            {
                "name": name,
                "access_level": p.get("metadata", {}).get("access_level", "standard"),
                "samples": p.get("metadata", {}).get("sample_count", 1),
                "enrolled_at": p.get("enrolled_at"),
            }
            for name, p in profiles.items()
        ]
        return JSONResponse({"speakers": speakers})

    @app.delete("/speakers/{name}")
    async def delete_speaker(name: str) -> JSONResponse:
        """Remove a speaker profile by name."""
        path = config.profiles_dir / f"{name.lower().replace(' ', '_')}.json"
        if path.exists():
            path.unlink()
            log.info("Deleted speaker profile '%s'", name)
            return JSONResponse({"status": "deleted", "speaker": name})
        raise HTTPException(404, f"Speaker '{name}' not found")

    @app.get("/health")
    async def health() -> dict:
        """Health check — returns model status and enrolled speaker count."""
        profiles = load_profiles(config.profiles_dir)
        return {
            "status": "ok",
            "model": "resemblyzer",
            "device": config.device,
            "enrolled_speakers": len(profiles),
            "profiles_dir": str(config.profiles_dir),
        }

    return app


# ---------------------------------------------------------------------------
# Standalone entry point (used by CLI)
# ---------------------------------------------------------------------------


def run_speaker_id(config: SpeakerIDConfig, host: str = "0.0.0.0", port: int = 8003) -> None:
    """Start the speaker ID server (blocking)."""
    log.info("Starting Speaker ID server on %s:%d", host, port)
    log.info("  Profiles dir: %s", config.profiles_dir)
    log.info("  Threshold:    %.2f", config.default_threshold)
    log.info("  Device:       %s", config.device)

    # Pre-load model at startup
    _get_encoder(config.device)

    app = create_app(config)
    uvicorn.run(app, host=host, port=port, log_level="info")
