"""Test endpoints for the HTTP control server.

Provides /test/* routes that exercise the voice pipeline (STT, LLM, TTS)
without requiring a real Discord connection. Designed to be mounted
alongside the existing /health and /control/* routes in the aiohttp server.

Public API:
  - ``build_test_routes(pipeline)`` — returns ``web.RouteTableDef`` to add
    to the existing aiohttp ``Application``.

Endpoints:
    GET  /test/health    — checks reachability of LLM, STT, TTS services
    POST /test/pipeline  — sends text through LLM (+ optional TTS), returns response
    POST /test/stt       — sends base64 PCM audio through STT, returns transcript
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time

from aiohttp import web

log = logging.getLogger("openclaw_voice.test_endpoints")

DEFAULT_LLM_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_STT_URL = "http://localhost:8001/inference"
DEFAULT_TTS_URL = "http://localhost:8002/v1/audio/speech"


def build_test_routes(
    pipeline: object | None = None,
) -> web.RouteTableDef:
    """Build aiohttp routes for pipeline test endpoints.

    Args:
        pipeline: A ``VoicePipeline`` instance (or None if not connected).
                  When None, /test/pipeline and /test/stt return 503.

    Returns:
        ``web.RouteTableDef`` that can be added to an aiohttp ``Application``.
    """
    routes = web.RouteTableDef()

    @routes.get("/test/health")
    async def test_health(_request: web.Request) -> web.Response:
        """Check reachability of LLM, STT, and TTS backend services.

        Returns JSON::

            {
                "services": {
                    "llm": {"status": "ok"|"error", "url": "...", "latency_ms": N},
                    "stt": {"status": "ok"|"error", "url": "...", "latency_ms": N},
                    "tts": {"status": "ok"|"error", "url": "...", "latency_ms": N}
                },
                "all_healthy": true|false
            }
        """
        import httpx

        services_config = {
            "llm": DEFAULT_LLM_URL,
            "stt": DEFAULT_STT_URL,
            "tts": DEFAULT_TTS_URL,
        }

        if pipeline is not None:
            from openclaw_voice.voice_pipeline import VoicePipeline

            if isinstance(pipeline, VoicePipeline):
                cfg = pipeline.config
                services_config["llm"] = cfg.llm_url
                services_config["stt"] = cfg.whisper_url
                services_config["tts"] = cfg.kokoro_url

        results: dict[str, dict] = {}

        async def _check_service(name: str, url: str) -> None:
            t_start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    if name == "llm":
                        check_url = url.rsplit("/", 1)[0] + "/models"
                        resp = await client.get(check_url)
                    elif name == "stt":
                        resp = await client.get(url.rsplit("/", 1)[0] + "/")
                    else:
                        resp = await client.get(url.rsplit("/", 3)[0] + "/")
                latency_ms = int((time.monotonic() - t_start) * 1000)
                results[name] = {
                    "status": "ok" if resp.status_code < 500 else "error",
                    "url": url,
                    "latency_ms": latency_ms,
                    "http_status": resp.status_code,
                }
            except Exception as exc:
                latency_ms = int((time.monotonic() - t_start) * 1000)
                results[name] = {
                    "status": "error",
                    "url": url,
                    "latency_ms": latency_ms,
                    "error": str(exc),
                }

        await asyncio.gather(*[_check_service(name, url) for name, url in services_config.items()])

        all_healthy = all(r["status"] == "ok" for r in results.values())
        return web.json_response({"services": results, "all_healthy": all_healthy})

    @routes.post("/test/pipeline")
    async def test_pipeline(request: web.Request) -> web.Response:
        """Run text through LLM (+ optional TTS), skipping STT.

        Request JSON::

            {
                "text": "hello, what is 2+2?",
                "skip_tts": false  // optional, default false
            }

        Response JSON::

            {
                "status": "ok",
                "llm_response": "...",
                "escalation": null | "...",
                "tts_bytes": 12345,
                "latency_ms": 456
            }
        """
        try:
            data = await request.json()
        except Exception as exc:
            return web.json_response({"error": f"Invalid JSON: {exc}"}, status=400)

        text = data.get("text", "").strip()
        if not text:
            return web.json_response({"error": "text is required"}, status=400)

        if not pipeline:
            return web.json_response(
                {"error": "No pipeline available (bot not connected to voice)"},
                status=503,
            )

        skip_tts = data.get("skip_tts", False)

        t_start = time.monotonic()

        messages = [
            {"role": "system", "content": pipeline.build_system_prompt()},
            {"role": "user", "content": text},
        ]

        loop = asyncio.get_event_loop()
        llm_response, escalation = await loop.run_in_executor(
            None, pipeline.call_llm_with_tools, messages
        )

        tts_bytes = 0
        if llm_response and not skip_tts:
            wav = await loop.run_in_executor(None, pipeline.synthesize_response, llm_response)
            tts_bytes = len(wav)

        latency_ms = int((time.monotonic() - t_start) * 1000)
        log.info(
            "Test pipeline complete",
            extra={
                "text": text[:60],
                "llm_len": len(llm_response),
                "escalation": escalation is not None,
                "tts_bytes": tts_bytes,
                "latency_ms": latency_ms,
            },
        )

        return web.json_response(
            {
                "status": "ok",
                "llm_response": llm_response,
                "escalation": escalation,
                "tts_bytes": tts_bytes,
                "latency_ms": latency_ms,
            }
        )

    @routes.post("/test/stt")
    async def test_stt(request: web.Request) -> web.Response:
        """Run base64-encoded PCM audio through STT.

        Request JSON::

            {
                "audio_b64": "<base64-encoded 16kHz mono int16 PCM>"
            }

        Response JSON::

            {
                "status": "ok",
                "transcript": "...",
                "audio_bytes": 12800,
                "latency_ms": 234
            }
        """
        try:
            data = await request.json()
        except Exception as exc:
            return web.json_response({"error": f"Invalid JSON: {exc}"}, status=400)

        audio_b64 = data.get("audio_b64", "")
        if not audio_b64:
            return web.json_response({"error": "audio_b64 is required"}, status=400)

        try:
            pcm_bytes = base64.b64decode(audio_b64, validate=True)
        except Exception as exc:
            return web.json_response({"error": f"Invalid base64: {exc}"}, status=400)

        if not pipeline:
            return web.json_response(
                {"error": "No pipeline available (bot not connected to voice)"},
                status=503,
            )

        t_start = time.monotonic()
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(None, pipeline.run_stt, pcm_bytes)
        latency_ms = int((time.monotonic() - t_start) * 1000)

        log.info(
            "Test STT complete",
            extra={
                "audio_bytes": len(pcm_bytes),
                "transcript_len": len(transcript),
                "latency_ms": latency_ms,
            },
        )

        return web.json_response(
            {
                "status": "ok",
                "transcript": transcript,
                "audio_bytes": len(pcm_bytes),
                "latency_ms": latency_ms,
            }
        )

    return routes
