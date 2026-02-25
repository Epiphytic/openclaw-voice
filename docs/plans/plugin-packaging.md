# Plan: Package Discord Voice Bot as OpenClaw Plugin

**Status:** Draft — awaiting V2 architecture refactor completion  
**Date:** 2026-02-25  
**Author:** Bel  
**Depends on:** V2 ConversationLog architecture (feat/v2-conversation-log)

## Context

The Discord voice bot ("Chip") currently runs as a standalone Python process, manually started via CLI. Escalation to the main agent (Bel) uses a hand-rolled WebSocket client (`gateway_client.py`) that implements the full gateway auth protocol (Ed25519 device signing, challenge-response, streaming).

OpenClaw plugins are TypeScript modules that run in-process with the Gateway. They can register background services, agent tools, RPC handlers, and config schemas. This plan packages the voice bot as `@openclaw/discord-voice` — a TS plugin that manages the Python process and bridges it cleanly into the OpenClaw ecosystem.

## Goals

1. **Install & configure via OpenClaw** — `openclaw plugins install @openclaw/discord-voice`, configure in `openclaw.json`
2. **Process lifecycle** — Gateway manages the Python process (start, stop, restart on crash, health checks)
3. **Native escalation** — Replace raw WebSocket gateway client with plugin-mediated RPC
4. **Agent tools** — Let Bel push voice actions (join channel, speak, leave)
5. **Config consolidation** — Single config surface in `plugins.entries.discord-voice.config`
6. **Zero hardcoded secrets** — Token flows through OpenClaw's auth system

## Architecture

```
┌─────────────────────────────────────────────────┐
│  OpenClaw Gateway (Node.js)                     │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  @openclaw/discord-voice plugin (TS)      │  │
│  │                                           │  │
│  │  • Background service (process manager)   │  │
│  │  • Agent tools (voice_join, voice_leave,  │  │
│  │    voice_speak, voice_status)             │  │
│  │  • Escalation RPC handler                 │  │
│  │  • Config schema + validation             │  │
│  └──────────┬───────────────┬────────────────┘  │
│             │ spawn/manage  │ HTTP/JSON-RPC      │
└─────────────┼───────────────┼────────────────────┘
              │               │
┌─────────────▼───────────────▼────────────────────┐
│  Voice Bot Process (Python)                       │
│                                                   │
│  • Discord voice gateway (py-cord)                │
│  • VoiceSink + debounce → STT Worker              │
│  • ConversationLog → LLM Worker → TTS Worker      │
│  • Local tools (weather, time, search)            │
│  • Escalation: HTTP POST to plugin endpoint       │
│  • Health endpoint: GET /health                   │
└───────────────────────────────────────────────────┘
```

## Components

### 1. TypeScript Plugin (`@openclaw/discord-voice`)

#### Manifest (`openclaw.plugin.json`)
```json
{
  "id": "discord-voice",
  "name": "Discord Voice",
  "description": "Real-time Discord voice assistant with local LLM, STT, and TTS",
  "version": "0.1.0",
  "channels": [],
  "configSchema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "pythonPath": {
        "type": "string",
        "description": "Path to Python venv binary",
        "default": ""
      },
      "botToken": {
        "type": "string",
        "description": "Discord bot token (or path to token file)"
      },
      "guildIds": {
        "type": "array",
        "items": { "type": "number" },
        "description": "Discord guild IDs to operate in"
      },
      "transcriptChannelId": {
        "type": "string",
        "description": "Channel ID for posting voice transcripts"
      },
      "llmModel": {
        "type": "string",
        "default": "Qwen/Qwen3-30B-A3B-Instruct-2507"
      },
      "llmUrl": {
        "type": "string",
        "default": "http://localhost:8000/v1/chat/completions"
      },
      "whisperUrl": {
        "type": "string",
        "default": "http://localhost:8001/inference"
      },
      "kokoroUrl": {
        "type": "string",
        "default": "http://localhost:8002/v1/audio/speech"
      },
      "ttsVoice": {
        "type": "string",
        "default": "af_heart"
      },
      "speakerIdUrl": {
        "type": "string",
        "default": ""
      },
      "botName": {
        "type": "string",
        "default": "Assistant"
      },
      "mainAgentName": {
        "type": "string",
        "default": "main agent"
      },
      "defaultLocation": {
        "type": "string",
        "default": ""
      },
      "defaultTimezone": {
        "type": "string",
        "default": "UTC"
      },
      "extraContext": {
        "type": "string",
        "default": ""
      },
      "whisperPrompt": {
        "type": "string",
        "default": ""
      },
      "vadSilenceMs": {
        "type": "number",
        "default": 1500
      },
      "vadMinSpeechMs": {
        "type": "number",
        "default": 500
      },
      "escalationPort": {
        "type": "number",
        "default": 18790,
        "description": "Local HTTP port for plugin ↔ Python bot communication"
      }
    },
    "required": ["botToken", "guildIds"]
  }
}
```

#### Plugin entry (`index.ts`)

```typescript
export default function (api) {
  const cfg = api.config;  // validated against configSchema

  // --- Background Service: Python process manager ---
  api.registerService({
    name: "discord-voice-bot",
    async start() {
      // Build env/args from plugin config
      // Spawn Python process with health monitoring
      // Restart on crash (max 3 retries, then alert)
    },
    async stop() {
      // Graceful SIGTERM → wait → SIGKILL
    },
    async health() {
      // GET http://localhost:{escalationPort}/health
    },
  });

  // --- Agent Tools ---
  api.registerTool({
    name: "voice_join",
    description: "Join a Discord voice channel",
    parameters: { /* guild_id, channel_id */ },
    async execute(_id, params) {
      // POST to Python bot's control endpoint
    },
  });

  api.registerTool({
    name: "voice_leave",
    description: "Leave the current voice channel",
    parameters: { /* guild_id */ },
    async execute(_id, params) { /* ... */ },
  });

  api.registerTool({
    name: "voice_speak",
    description: "Synthesize and play text in voice channel",
    parameters: { /* guild_id, text */ },
    async execute(_id, params) { /* ... */ },
  });

  api.registerTool({
    name: "voice_status",
    description: "Get voice bot status (connected guilds, active sessions)",
    parameters: {},
    async execute() { /* GET /health */ },
  });

  // --- Escalation RPC Handler ---
  api.registerRpc("discord-voice.escalate", async (params) => {
    // params: { message, guildId, channelId, userId }
    // Use api.runtime to send agent request (native, no WS hand-rolling)
    // Return agent response text
  });
}
```

### 2. Python Bot Changes

#### Replace `gateway_client.py` with `escalation_client.py`
```python
async def escalate(message: str, guild_id: int, channel_id: int, 
                   user_id: int, port: int = 18790) -> str:
    """Send escalation to OpenClaw plugin via local HTTP."""
    async with aiohttp.ClientSession() as session:
        resp = await session.post(
            f"http://127.0.0.1:{port}/rpc/discord-voice.escalate",
            json={
                "message": message,
                "guildId": guild_id,
                "channelId": channel_id,
                "userId": user_id,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        )
        data = await resp.json()
        return data["text"]
```

This replaces ~340 lines of WebSocket protocol, Ed25519 signing, challenge-response, and streaming assembly with ~15 lines of HTTP POST.

#### Add control + health endpoints
```python
# Lightweight aiohttp server alongside Discord bot
routes = web.RouteTableDef()

@routes.get("/health")
async def health(request):
    return web.json_response({
        "status": "ok",
        "guilds": list(bot.voice_sessions.keys()),
        "uptime": time.monotonic() - bot.start_time,
    })

@routes.post("/control/join")
async def join(request):
    data = await request.json()
    # Trigger /join on specified guild/channel
    
@routes.post("/control/leave")  
async def leave(request):
    data = await request.json()
    # Trigger /leave

@routes.post("/control/speak")
async def speak(request):
    data = await request.json()
    # Synthesize + queue TTS for text
```

#### Config input changes
- Accept config via `--config-json` (stdin JSON) in addition to `--config` (TOML file)
- Plugin passes config as JSON when spawning the process
- TOML file still works for standalone/development use

### 3. Config Migration

**Before (standalone):**
```
/home/models/voice-bot.toml          # instance config
OPENCLAW_VOICE_DISCORD_TOKEN=...     # env var
manual process start                 # no lifecycle management
```

**After (plugin):**
```json
{
  "plugins": {
    "entries": {
      "discord-voice": {
        "enabled": true,
        "config": {
          "botToken": "file:///home/models/discord-voice-token.txt",
          "guildIds": [1473159530316566551],
          "transcriptChannelId": "1476027839391469718",
          "llmModel": "Qwen/Qwen3-30B-A3B-Instruct-2507",
          "botName": "Chip",
          "mainAgentName": "Bel",
          "defaultLocation": "Cassidy, BC, Canada",
          "defaultTimezone": "America/Vancouver",
          "extraContext": "You're on a farm on Vancouver Island. The primary user is Liam.",
          "whisperPrompt": "Chip, Bel, Liam, Cassidy, Ladysmith, Vancouver Island"
        }
      }
    }
  }
}
```

## What Gets Deleted

- `gateway_client.py` (~344 lines) — replaced by ~15-line HTTP escalation client
- `~/.openclaw/voice-device-identity.json` — Ed25519 keypair no longer needed
- Manual process management scripts/cron — plugin handles lifecycle
- Raw WebSocket auth protocol — plugin uses native `api.runtime` agent calls

## What Stays

- All Python voice processing (STT, LLM, TTS, VAD, ConversationLog, tools)
- Discord bot connection (py-cord voice gateway)
- `voice_pipeline.py`, `tools.py`, facades/, bridges
- `cli.py` (gains `--config-json` but keeps TOML support)

## Implementation Phases

### Phase 1: HTTP Control Layer (Python side)
1. Add aiohttp health + control server to discord_bot.py
2. Add `escalation_client.py` (HTTP POST replacement for gateway WS)
3. Add `--config-json` flag to CLI
4. Test: standalone mode still works with TOML, HTTP endpoints respond

### Phase 2: TypeScript Plugin Scaffold
5. Create plugin directory structure + manifest
6. Implement background service (process spawn/monitor/restart)
7. Implement config → env/args mapping
8. Test: `openclaw plugins install` from local path, bot starts on gateway start

### Phase 3: Escalation Bridge
9. Implement `discord-voice.escalate` RPC handler using `api.runtime`
10. Wire Python escalation client to plugin endpoint
11. Test: full escalation round-trip (Chip → plugin → Bel → plugin → Chip → TTS)

### Phase 4: Agent Tools
12. Register voice_join, voice_leave, voice_speak, voice_status tools
13. Test: Bel can proactively join voice and speak

### Phase 5: Cleanup
14. Remove `gateway_client.py`
15. Remove device identity generation
16. Update README, MANIFEST.md
17. Publish to npm as `@openclaw/discord-voice` (or `@epiphytic/openclaw-discord-voice`)

## Open Questions

1. **npm scope** — `@openclaw/discord-voice` (official) or `@epiphytic/openclaw-discord-voice` (community)?
2. **Python path discovery** — should the plugin auto-detect venv, or require explicit `pythonPath`?
3. **Token handling** — `file://` URI for token files, or OpenClaw's auth profile system?
4. **Streaming escalation** — HTTP POST is request/response. For keepalive messages during long escalation, should the plugin push SSE events back to the Python process, or should the Python process poll?
5. **Multi-guild** — one Python process per guild, or one process serving all guilds? (Current: one process, all guilds)
