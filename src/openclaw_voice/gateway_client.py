"""
OpenClaw Gateway WebSocket client for voice bot → main agent escalation.

Connects to the local OpenClaw gateway, sends a message to the main agent
session, and collects the streamed response.

Device authentication uses Ed25519 keypair signing.  A persistent keypair
is generated on first use and stored alongside the config.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import ssl
import time
import uuid
from pathlib import Path

log = logging.getLogger("openclaw_voice.gateway_client")

# ---------------------------------------------------------------------------
# Device identity (Ed25519 keypair — persistent)
# ---------------------------------------------------------------------------

_IDENTITY_PATH = Path.home() / ".openclaw" / "voice-device-identity.json"


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _load_or_create_identity(path: Path = _IDENTITY_PATH) -> dict:
    """Load or generate a persistent Ed25519 device identity.

    Returns dict with keys: deviceId, publicKeyPem, privateKeyPem, publicKeyRaw.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    if path.is_file():
        try:
            stored = json.loads(path.read_text())
            if (
                stored.get("version") == 1
                and stored.get("publicKeyPem")
                and stored.get("privateKeyPem")
            ):
                # Derive raw public key bytes for base64url encoding
                from cryptography.hazmat.primitives.serialization import load_pem_public_key

                pub = load_pem_public_key(stored["publicKeyPem"].encode())
                raw = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
                stored["publicKeyRaw"] = raw
                return stored
        except Exception as exc:
            log.warning("Could not load device identity from %s: %s", path, exc)

    # Generate new Ed25519 keypair
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    pub_pem = public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()
    priv_pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode()
    raw = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    device_id = hashlib.sha256(raw).hexdigest()

    identity = {
        "version": 1,
        "deviceId": device_id,
        "publicKeyPem": pub_pem,
        "privateKeyPem": priv_pem,
        "createdAtMs": int(time.time() * 1000),
        "publicKeyRaw": raw,  # transient, not persisted
    }

    # Persist (without raw bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    persist = {k: v for k, v in identity.items() if k != "publicKeyRaw"}
    path.write_text(json.dumps(persist, indent=2) + "\n")
    with contextlib.suppress(OSError):
        path.chmod(0o600)

    log.info("Generated new device identity: %s", device_id[:16])
    return identity


def _sign_payload(private_key_pem: str, payload: str) -> str:
    """Sign a payload string with the Ed25519 private key, return base64url."""
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    key = load_pem_private_key(private_key_pem.encode(), password=None)
    sig = key.sign(payload.encode("utf-8"))
    return _base64url_encode(sig)


def _build_auth_payload(
    device_id: str,
    client_id: str,
    client_mode: str,
    role: str,
    scopes: list[str],
    signed_at_ms: int,
    token: str,
    nonce: str,
) -> str:
    """Build the v2 device auth payload string matching gateway expectations."""
    return "|".join(
        [
            "v2",
            device_id,
            client_id,
            client_mode,
            role,
            ",".join(scopes),
            str(signed_at_ms),
            token,
            nonce,
        ]
    )


# ---------------------------------------------------------------------------
# Gateway WebSocket client
# ---------------------------------------------------------------------------

CLIENT_ID = "gateway-client"
CLIENT_MODE = "backend"


async def send_to_bel(
    message: str,
    *,
    timeout_s: float = 90.0,
    gateway_url: str | None = None,
    gateway_token: str | None = None,
) -> str | None:
    """Send a message to the main agent via the OpenClaw gateway WebSocket.

    Connects, authenticates with device signing, sends an ``agent`` request,
    collects streamed text from ``event:agent`` frames, and waits for the
    final response.

    Returns:
        The main agent's response text, or None on failure/timeout.
    """
    try:
        import websockets  # type: ignore[import-untyped]
    except ImportError:
        log.error("websockets package not installed — pip install websockets")
        return None

    url = gateway_url or os.environ.get("OPENCLAW_GATEWAY_URL", "wss://127.0.0.1:18789")
    token = gateway_token or os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")

    # Allow self-signed certs for local gateway
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    # Load device identity
    try:
        identity = _load_or_create_identity()
    except Exception as exc:
        log.error("Failed to load/create device identity: %s", exc)
        return None

    req_id = str(uuid.uuid4())[:8]
    idem_key = str(uuid.uuid4())
    collected_text: list[str] = []
    run_id: str | None = None
    role = "operator"
    scopes = ["operator.read", "operator.write"]

    try:
        async with websockets.connect(
            url,
            ssl=ssl_ctx,
            open_timeout=10,
            close_timeout=5,
        ) as ws:
            # Step 1: Wait for connect.challenge
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            challenge = json.loads(raw)
            if challenge.get("event") != "connect.challenge":
                log.warning("Expected connect.challenge, got: %s", challenge)
                return None

            nonce = challenge.get("payload", {}).get("nonce", "")

            # Step 2: Build device signature
            signed_at_ms = int(time.time() * 1000)
            auth_payload = _build_auth_payload(
                device_id=identity["deviceId"],
                client_id=CLIENT_ID,
                client_mode=CLIENT_MODE,
                role=role,
                scopes=scopes,
                signed_at_ms=signed_at_ms,
                token=token,
                nonce=nonce,
            )
            signature = _sign_payload(identity["privateKeyPem"], auth_payload)
            public_key_b64 = _base64url_encode(identity["publicKeyRaw"])

            # Step 3: Send connect handshake
            connect_msg = {
                "type": "req",
                "id": "connect-1",
                "method": "connect",
                "params": {
                    "minProtocol": 3,
                    "maxProtocol": 3,
                    "client": {
                        "id": CLIENT_ID,
                        "version": "0.1.0",
                        "platform": "linux",
                        "mode": CLIENT_MODE,
                    },
                    "role": role,
                    "scopes": scopes,
                    "caps": [],
                    "commands": [],
                    "permissions": {},
                    "auth": {"token": token},
                    "locale": "en-US",
                    "userAgent": "openclaw-voice/0.1.0",
                    "device": {
                        "id": identity["deviceId"],
                        "publicKey": public_key_b64,
                        "signature": signature,
                        "signedAt": signed_at_ms,
                        "nonce": nonce,
                    },
                },
            }
            await ws.send(json.dumps(connect_msg))

            # Step 4: Wait for connect response
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            resp = json.loads(raw)
            if resp.get("type") == "res" and resp.get("ok"):
                log.info("Connected to gateway (device %s)", identity["deviceId"][:16])
            else:
                log.warning("Gateway connect failed: %s", resp)
                return None

            # Step 5: Send agent request
            agent_msg = {
                "type": "req",
                "id": req_id,
                "method": "agent",
                "params": {
                    "message": message,
                    "agentId": "main",
                    "idempotencyKey": idem_key,
                },
            }
            await ws.send(json.dumps(agent_msg))
            log.info("Sent escalation: %.100s", message)

            # Step 6: Collect streaming events and wait for final response
            deadline = asyncio.get_event_loop().time() + timeout_s

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    log.warning("Escalation timed out after %.0fs", timeout_s)
                    return None

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    log.warning("Escalation timed out")
                    return None

                frame = json.loads(raw)

                # Track streaming assistant text
                # NOTE: event:agent text may be cumulative (full text so far)
                # or incremental (just the delta). We keep only the latest
                # value and use it as fallback if the final res has no text.
                if frame.get("type") == "event" and frame.get("event") == "agent":
                    payload_data = frame.get("payload", {})
                    data = payload_data.get("data", {})
                    stream = payload_data.get("stream", "")
                    if stream == "assistant" and "text" in data:
                        # Replace — keep only the latest (cumulative) value
                        collected_text.clear()
                        collected_text.append(data["text"])
                    if not run_id:
                        run_id = payload_data.get("runId")

                # Check for final response
                elif frame.get("type") == "res" and frame.get("id") == req_id:
                    payload = frame.get("payload", {})
                    status = payload.get("status", "")

                    if status == "accepted":
                        run_id = payload.get("runId")
                        log.debug("Agent run accepted: %s", run_id)
                        continue

                    if status == "ok":
                        result = payload.get("result", {})
                        if isinstance(result, dict):
                            text = (
                                result.get("text")
                                or result.get("reply")
                                or result.get("output", "")
                            )
                        elif isinstance(result, str):
                            text = result
                        else:
                            text = ""

                        if not text and collected_text:
                            text = "".join(collected_text)

                        log.info("Main agent responded (run %s): %.100s", run_id, text)
                        return text.strip() if text else None

                    else:
                        log.warning("Agent run %s: %s", status, payload)
                        return None

                # Ignore other events (presence, tick, etc.)

    except Exception as exc:
        log.error("Gateway escalation error: %s", exc)
        return None
