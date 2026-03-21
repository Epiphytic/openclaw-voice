#!/usr/bin/env bash
#
# End-to-end test script for the voice pipeline.
#
# Exercises the /test/* HTTP endpoints on the bot's control server
# (default: localhost:18790) to validate the full STT -> LLM -> TTS pipeline
# without requiring a real Discord connection.
#
# Usage:
#   ./scripts/test_pipeline.sh              # defaults to localhost:18790
#   ./scripts/test_pipeline.sh 18790        # explicit port
#   CONTROL_PORT=18790 ./scripts/test_pipeline.sh
#
# Exit code: 0 if all tests pass, 1 otherwise.

set -euo pipefail

PORT="${1:-${CONTROL_PORT:-18790}}"
BASE="http://localhost:${PORT}"
PASS=0
FAIL=0
SKIP=0

# Colors (disable if not a terminal)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    NC='\033[0m'
else
    GREEN=''
    RED=''
    YELLOW=''
    NC=''
fi

pass() { echo -e "  ${GREEN}PASS${NC}: $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}FAIL${NC}: $1 — $2"; FAIL=$((FAIL + 1)); }
skip() { echo -e "  ${YELLOW}SKIP${NC}: $1 — $2"; SKIP=$((SKIP + 1)); }

echo "========================================="
echo "Pipeline E2E Test Suite"
echo "Target: ${BASE}"
echo "========================================="
echo ""

# -------------------------------------------------------------------
# 0. Connectivity check — is the control server up?
# -------------------------------------------------------------------
echo "[0] Control server connectivity"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/health" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "000" ]; then
    echo -e "  ${RED}FATAL${NC}: Cannot reach control server at ${BASE}/health"
    echo "  Is the bot running? Start with: openclaw-voice discord-bot ..."
    exit 1
fi
pass "Control server reachable (HTTP ${HTTP_CODE})"
echo ""

# -------------------------------------------------------------------
# 1. GET /test/health — service health checks
# -------------------------------------------------------------------
echo "[1] Service health check (GET /test/health)"
HEALTH_RESP=$(curl -s "${BASE}/test/health")
ALL_HEALTHY=$(echo "$HEALTH_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('all_healthy', False))" 2>/dev/null || echo "error")

if [ "$ALL_HEALTHY" = "True" ]; then
    pass "All services healthy"
else
    # Report individual service status
    for SVC in llm stt tts; do
        STATUS=$(echo "$HEALTH_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['services']['${SVC}']['status'])" 2>/dev/null || echo "error")
        if [ "$STATUS" = "ok" ]; then
            pass "${SVC} service: ok"
        else
            ERROR=$(echo "$HEALTH_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['services']['${SVC}'].get('error','unknown'))" 2>/dev/null || echo "unknown")
            fail "${SVC} service" "${ERROR}"
        fi
    done
fi
echo ""

# -------------------------------------------------------------------
# 2. POST /test/pipeline — LLM + TTS pipeline test
# -------------------------------------------------------------------
echo "[2] Pipeline test (POST /test/pipeline)"
PIPELINE_RESP=$(curl -s -X POST "${BASE}/test/pipeline" \
    -H "Content-Type: application/json" \
    -d '{"text": "hello, what is 2+2?"}' 2>/dev/null || echo '{"error":"curl_failed"}')

PIPELINE_STATUS=$(echo "$PIPELINE_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','error'))" 2>/dev/null || echo "error")

if [ "$PIPELINE_STATUS" = "ok" ]; then
    LLM_RESP=$(echo "$PIPELINE_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('llm_response','') or d.get('escalation',''))" 2>/dev/null || echo "")
    TTS_BYTES=$(echo "$PIPELINE_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tts_bytes',0))" 2>/dev/null || echo "0")
    LATENCY=$(echo "$PIPELINE_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('latency_ms',0))" 2>/dev/null || echo "?")

    if [ -n "$LLM_RESP" ]; then
        pass "LLM response: '${LLM_RESP:0:80}...' (${LATENCY}ms)"
    else
        fail "LLM response" "empty response"
    fi

    if [ "$TTS_BYTES" -gt 0 ] 2>/dev/null; then
        pass "TTS synthesis: ${TTS_BYTES} bytes"
    else
        skip "TTS synthesis" "0 bytes (LLM may have escalated)"
    fi
else
    ERROR=$(echo "$PIPELINE_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error','unknown'))" 2>/dev/null || echo "unknown")
    fail "Pipeline test" "${ERROR}"
fi
echo ""

# -------------------------------------------------------------------
# 3. POST /test/pipeline (text-only, skip TTS)
# -------------------------------------------------------------------
echo "[3] Pipeline text-only test (POST /test/pipeline, skip_tts=true)"
PIPELINE2_RESP=$(curl -s -X POST "${BASE}/test/pipeline" \
    -H "Content-Type: application/json" \
    -d '{"text": "hello", "skip_tts": true}' 2>/dev/null || echo '{"error":"curl_failed"}')

PIPELINE2_STATUS=$(echo "$PIPELINE2_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','error'))" 2>/dev/null || echo "error")
if [ "$PIPELINE2_STATUS" = "ok" ]; then
    pass "Text-only pipeline works"
else
    ERROR=$(echo "$PIPELINE2_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error','unknown'))" 2>/dev/null || echo "unknown")
    fail "Text-only pipeline" "${ERROR}"
fi
echo ""

# -------------------------------------------------------------------
# 4. POST /test/stt — STT test with minimal audio
# -------------------------------------------------------------------
echo "[4] STT test (POST /test/stt)"

# Generate a minimal sine wave PCM buffer (200ms, 440Hz, 16kHz mono int16)
AUDIO_B64=$(python3 -c "
import base64, struct, math
sr = 16000
dur_ms = 200
n = sr * dur_ms // 1000
samples = [int(16000 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n)]
pcm = struct.pack(f'<{n}h', *samples)
print(base64.b64encode(pcm).decode())
" 2>/dev/null || echo "")

if [ -z "$AUDIO_B64" ]; then
    skip "STT test" "Could not generate test audio"
else
    STT_RESP=$(curl -s -X POST "${BASE}/test/stt" \
        -H "Content-Type: application/json" \
        -d "{\"audio_b64\": \"${AUDIO_B64}\"}" 2>/dev/null || echo '{"error":"curl_failed"}')

    STT_STATUS=$(echo "$STT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','error'))" 2>/dev/null || echo "error")

    if [ "$STT_STATUS" = "ok" ]; then
        TRANSCRIPT=$(echo "$STT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('transcript',''))" 2>/dev/null || echo "")
        LATENCY=$(echo "$STT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('latency_ms',0))" 2>/dev/null || echo "?")
        if [ -n "$TRANSCRIPT" ]; then
            pass "STT transcript: '${TRANSCRIPT:0:80}' (${LATENCY}ms)"
        else
            pass "STT returned empty transcript (expected for synthetic audio, ${LATENCY}ms)"
        fi
    else
        ERROR=$(echo "$STT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error','unknown'))" 2>/dev/null || echo "unknown")
        fail "STT test" "${ERROR}"
    fi
fi
echo ""

# -------------------------------------------------------------------
# 5. Validation tests (error handling)
# -------------------------------------------------------------------
echo "[5] Error handling validation"

# Missing text field
ERR_RESP=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE}/test/pipeline" \
    -H "Content-Type: application/json" \
    -d '{}' 2>/dev/null || echo "000")
if [ "$ERR_RESP" = "400" ]; then
    pass "Missing text returns 400"
else
    fail "Missing text validation" "Expected 400, got ${ERR_RESP}"
fi

# Missing audio field
ERR_RESP2=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE}/test/stt" \
    -H "Content-Type: application/json" \
    -d '{}' 2>/dev/null || echo "000")
if [ "$ERR_RESP2" = "400" ]; then
    pass "Missing audio_b64 returns 400"
else
    fail "Missing audio validation" "Expected 400, got ${ERR_RESP2}"
fi

# Invalid base64
ERR_RESP3=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE}/test/stt" \
    -H "Content-Type: application/json" \
    -d '{"audio_b64": "not-valid!!!"}' 2>/dev/null || echo "000")
if [ "$ERR_RESP3" = "400" ]; then
    pass "Invalid base64 returns 400"
else
    fail "Invalid base64 validation" "Expected 400, got ${ERR_RESP3}"
fi
echo ""

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo "========================================="
TOTAL=$((PASS + FAIL + SKIP))
echo "Results: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped (${TOTAL} total)"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
