/**
 * @epiphytic/openclaw-discord-voice
 *
 * OpenClaw plugin that manages the Discord voice bot (Chip) as a child process.
 *
 * Responsibilities:
 *   - Background service: spawn, monitor, and restart the Python voice bot
 *   - Escalation RPC HTTP handler: receive requests from Chip, route to the main agent, return text
 *   - Agent tools (optional): voice_join, voice_leave, voice_speak, voice_status
 *
 * Config is passed to the Python process as JSON written to its stdin.
 */

import { Type } from "@sinclair/typebox";
import { exec as execCb, spawn, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import { existsSync } from "node:fs";
import * as http from "node:http";
import * as path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

const execAsync = promisify(execCb);

// ---------------------------------------------------------------------------
// Config shape (mirrors openclaw.plugin.json configSchema)
// ---------------------------------------------------------------------------

interface DiscordVoiceConfig {
  enabled?: boolean;
  pythonPath?: string;
  botToken: string;
  guildIds: number[];
  transcriptChannelId?: string;
  llmModel?: string;
  llmUrl?: string;
  whisperUrl?: string;
  kokoroUrl?: string;
  ttsVoice?: string;
  botName?: string;
  mainAgentName?: string;
  defaultLocation?: string;
  defaultTimezone?: string;
  extraContext?: string;
  whisperPrompt?: string;
  vadMinSpeechMs?: number;
  speechEndDelayMs?: number;
  channelContextMessages?: number;
  ttsReadChannel?: boolean;
  correctionsFile?: string;
  /** Port for the Python bot's HTTP control/health server (default 18790) */
  controlPort?: number;
}

// ---------------------------------------------------------------------------
// Python discovery
// ---------------------------------------------------------------------------

const VENV_CANDIDATES = [
  path.join(process.env.HOME ?? "/root", ".openclaw", "workspace", "openclaw-voice", ".venv"),
  path.join(process.env.HOME ?? "/root", ".venv"),
  path.join(process.env.HOME ?? "/root", "venv"),
  "/opt/openclaw-voice/.venv",
];

async function findPython(override?: string): Promise<string> {
  if (override && existsSync(override)) {
    return override;
  }

  for (const base of VENV_CANDIDATES) {
    const candidate = path.join(base, "bin", "python");
    if (existsSync(candidate)) {
      return candidate;
    }
  }

  // Fall back to PATH
  try {
    const { stdout } = await execAsync("which python3 || which python");
    const found = stdout.trim();
    if (found) {
      return found;
    }
  } catch {
    // ignore
  }

  return "python3";
}

// ---------------------------------------------------------------------------
// Core agent deps (mirrors voice-call's core-bridge.ts pattern)
// ---------------------------------------------------------------------------

type CoreAgentDeps = {
  resolveAgentDir: (cfg: unknown, agentId: string) => string;
  resolveAgentWorkspaceDir: (cfg: unknown, agentId: string) => string;
  resolveAgentIdentity: (cfg: unknown, agentId: string) => { name?: string | null } | null;
  resolveThinkingDefault: (params: { cfg: unknown; provider?: string; model?: string }) => string;
  runEmbeddedPiAgent: (params: {
    sessionId: string;
    sessionKey?: string;
    messageProvider?: string;
    sessionFile: string;
    workspaceDir: string;
    config?: unknown;
    prompt: string;
    provider?: string;
    model?: string;
    thinkLevel?: string;
    verboseLevel?: string;
    timeoutMs: number;
    runId: string;
    lane?: string;
    extraSystemPrompt?: string;
    agentDir?: string;
  }) => Promise<{ payloads?: Array<{ text?: string; isError?: boolean }>; meta?: { aborted?: boolean } }>;
  resolveAgentTimeoutMs: (opts: { cfg: unknown }) => number;
  ensureAgentWorkspace: (params?: { dir: string }) => Promise<void>;
  resolveStorePath: (store?: string, opts?: { agentId?: string }) => string;
  loadSessionStore: (storePath: string) => Record<string, unknown>;
  saveSessionStore: (storePath: string, store: Record<string, unknown>) => Promise<void>;
  resolveSessionFilePath: (sessionId: string, entry: unknown, opts?: { agentId?: string }) => string;
  DEFAULT_MODEL: string;
  DEFAULT_PROVIDER: string;
};

let _coreDepsPromise: Promise<CoreAgentDeps> | null = null;

function resolveOpenClawRoot(): string {
  const override = process.env.OPENCLAW_ROOT?.trim();
  if (override) return override;

  const candidates: string[] = [];
  if (process.argv[1]) candidates.push(path.dirname(process.argv[1]));
  candidates.push(process.cwd());
  try {
    candidates.push(path.dirname(fileURLToPath(import.meta.url)));
  } catch { /* ignore */ }

  for (const start of candidates) {
    let dir = start;
    for (;;) {
      const pkgPath = path.join(dir, "package.json");
      try {
        if (fs.existsSync(pkgPath)) {
          const raw = fs.readFileSync(pkgPath, "utf8");
          const pkg = JSON.parse(raw) as { name?: string };
          if (pkg.name === "openclaw") return dir;
        }
      } catch { /* ignore */ }
      const parent = path.dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  }

  throw new Error("Cannot resolve OpenClaw root. Set OPENCLAW_ROOT.");
}

async function loadCoreAgentDeps(): Promise<CoreAgentDeps> {
  if (_coreDepsPromise) return _coreDepsPromise;
  _coreDepsPromise = (async () => {
    const distPath = path.join(resolveOpenClawRoot(), "dist", "extensionAPI.js");
    if (!fs.existsSync(distPath)) {
      throw new Error(
        `Missing core module at ${distPath}. Run \`pnpm build\` or install the official package.`,
      );
    }
    return await import(pathToFileURL(distPath).href) as CoreAgentDeps;
  })();
  return _coreDepsPromise;
}

// ---------------------------------------------------------------------------
// Agent invocation
// ---------------------------------------------------------------------------

async function invokeMainAgent(params: {
  message: string;
  guildId: string;
  channelId: string;
  userId: string;
  cfg: unknown;
}): Promise<string> {
  const { message, guildId, channelId, userId, cfg } = params;

  const deps = await loadCoreAgentDeps();

  const agentId = "main";
  const sessionKey = `discord-voice:${guildId}:${channelId}`;
  const storePath = deps.resolveStorePath((cfg as any)?.session?.store, { agentId });
  const agentDir = deps.resolveAgentDir(cfg, agentId);
  const workspaceDir = deps.resolveAgentWorkspaceDir(cfg, agentId);
  await deps.ensureAgentWorkspace({ dir: workspaceDir });

  const sessionStore = deps.loadSessionStore(storePath);
  const now = Date.now();
  type SessionEntry = { sessionId: string; updatedAt: number };
  let sessionEntry = sessionStore[sessionKey] as SessionEntry | undefined;
  if (!sessionEntry) {
    const { randomUUID } = await import("node:crypto");
    sessionEntry = { sessionId: randomUUID(), updatedAt: now };
    sessionStore[sessionKey] = sessionEntry;
    await deps.saveSessionStore(storePath, sessionStore);
  }

  const sessionId = sessionEntry.sessionId;
  const sessionFile = deps.resolveSessionFilePath(sessionId, sessionEntry, { agentId });

  const modelRef = (cfg as any)?.agents?.list?.[0]?.model ?? deps.DEFAULT_MODEL;
  const slashIdx = modelRef.indexOf("/");
  const provider = slashIdx === -1 ? deps.DEFAULT_PROVIDER : modelRef.slice(0, slashIdx);
  const model = slashIdx === -1 ? modelRef : modelRef.slice(slashIdx + 1);

  const thinkLevel = deps.resolveThinkingDefault({ cfg, provider, model });
  const timeoutMs = deps.resolveAgentTimeoutMs({ cfg });
  const runId = `discord-voice:${guildId}:${Date.now()}`;

  const result = await deps.runEmbeddedPiAgent({
    sessionId,
    sessionKey,
    messageProvider: "discord-voice",
    sessionFile,
    workspaceDir,
    config: cfg,
    prompt: message,
    provider,
    model,
    thinkLevel,
    verboseLevel: "off",
    timeoutMs,
    runId,
    lane: "discord-voice",
    agentDir,
  });

  const texts = (result.payloads ?? [])
    .filter((p) => p.text && !p.isError)
    .map((p) => p.text?.trim())
    .filter(Boolean);

  return texts.join(" ") || "I couldn't generate a response right now.";
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

function readJsonBody(req: http.IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk: Buffer) => { body += chunk.toString(); });
    req.on("end", () => {
      try { resolve(JSON.parse(body)); }
      catch (e) { reject(new Error("Invalid JSON body")); }
    });
    req.on("error", reject);
  });
}

function sendJson(res: http.ServerResponse, status: number, data: unknown): void {
  const body = JSON.stringify(data);
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(body);
}

// ---------------------------------------------------------------------------
// Process manager
// ---------------------------------------------------------------------------

const MAX_RESTARTS = 3;
const RESTART_DELAY_MS = 5_000;
const STOP_GRACE_MS = 5_000;

class VoiceBotProcess {
  private proc: ChildProcess | null = null;
  private restarts = 0;
  private stopping = false;
  private config: DiscordVoiceConfig;
  private pythonPath: string;
  private logger: OpenClawPluginApi["logger"];
  private gatewayPort: number;

  constructor(opts: {
    config: DiscordVoiceConfig;
    pythonPath: string;
    logger: OpenClawPluginApi["logger"];
    gatewayPort: number;
  }) {
    this.config = opts.config;
    this.pythonPath = opts.pythonPath;
    this.logger = opts.logger;
    this.gatewayPort = opts.gatewayPort;
  }

  async start(): Promise<void> {
    this.stopping = false;
    this.restarts = 0;
    await this._spawn();
  }

  private buildConfigJson(): string {
    const controlPort = this.config.controlPort ?? 18790;
    const payload = {
      token: this.config.botToken,
      guild_ids: this.config.guildIds,
      transcript_channel_id: this.config.transcriptChannelId ?? null,
      llm_model: this.config.llmModel ?? "Qwen/Qwen3-30B-A3B-Instruct-2507",
      llm_url: this.config.llmUrl ?? "http://localhost:8000/v1/chat/completions",
      whisper_url: this.config.whisperUrl ?? "http://localhost:8001/inference",
      kokoro_url: this.config.kokoroUrl ?? "http://localhost:8002/v1/audio/speech",
      tts_voice: this.config.ttsVoice ?? "af_heart",
      bot_name: this.config.botName ?? "Assistant",
      main_agent_name: this.config.mainAgentName ?? "main agent",
      default_location: this.config.defaultLocation ?? "",
      default_timezone: this.config.defaultTimezone ?? "UTC",
      extra_context: this.config.extraContext ?? "",
      whisper_prompt: this.config.whisperPrompt ?? "",
      vad_min_speech_ms: this.config.vadMinSpeechMs ?? 500,
      speech_end_delay_ms: this.config.speechEndDelayMs ?? 1000,
      channel_context_messages: this.config.channelContextMessages ?? 10,
      tts_read_channel: this.config.ttsReadChannel ?? true,
      corrections_file: this.config.correctionsFile ?? "",
      control_port: controlPort,
      gateway_port: this.gatewayPort,
    };
    return JSON.stringify(payload);
  }

  private async _spawn(): Promise<void> {
    if (this.stopping) return;

    const configJson = this.buildConfigJson();

    this.logger.info(`[discord-voice] Spawning Python bot (python: ${this.pythonPath})`);

    this.proc = spawn(
      this.pythonPath,
      ["-m", "openclaw_voice.cli", "discord-bot", "--config-json", "-"],
      {
        stdio: ["pipe", "inherit", "inherit"],
        env: { ...process.env },
      },
    );

    // Write config JSON to stdin, then close it
    if (this.proc.stdin) {
      this.proc.stdin.write(configJson + "\n");
      this.proc.stdin.end();
    }

    this.proc.on("exit", (code, signal) => {
      if (this.stopping) {
        this.logger.info(`[discord-voice] Bot process exited (stopping): code=${code} signal=${signal}`);
        return;
      }

      this.logger.warn(`[discord-voice] Bot process exited unexpectedly: code=${code} signal=${signal}`);

      if (this.restarts < MAX_RESTARTS) {
        this.restarts++;
        this.logger.info(`[discord-voice] Restart ${this.restarts}/${MAX_RESTARTS} in ${RESTART_DELAY_MS}ms…`);
        setTimeout(() => { this._spawn().catch(() => {}); }, RESTART_DELAY_MS);
      } else {
        this.logger.error(`[discord-voice] Bot process crashed ${MAX_RESTARTS} times — giving up.`);
      }
    });

    this.proc.on("error", (err) => {
      this.logger.error(`[discord-voice] Bot process error: ${err.message}`);
    });
  }

  async stop(): Promise<void> {
    this.stopping = true;
    const proc = this.proc;
    if (!proc || proc.exitCode !== null) return;

    proc.kill("SIGTERM");

    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        this.logger.warn("[discord-voice] Graceful stop timed out — SIGKILL");
        proc.kill("SIGKILL");
        resolve();
      }, STOP_GRACE_MS);

      proc.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
    });

    this.proc = null;
  }

  get controlPort(): number {
    return this.config.controlPort ?? 18790;
  }

  async httpPost(path: string, body: unknown): Promise<unknown> {
    const port = this.controlPort;
    return new Promise((resolve, reject) => {
      const data = JSON.stringify(body);
      const req = http.request(
        { hostname: "127.0.0.1", port, path, method: "POST",
          headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(data) } },
        (res) => {
          let raw = "";
          res.on("data", (c: Buffer) => { raw += c; });
          res.on("end", () => {
            try { resolve(JSON.parse(raw)); }
            catch { resolve({ raw }); }
          });
        },
      );
      req.on("error", reject);
      req.setTimeout(30_000, () => { req.destroy(new Error("Request timed out")); });
      req.write(data);
      req.end();
    });
  }

  async httpGet(path: string): Promise<unknown> {
    const port = this.controlPort;
    return new Promise((resolve, reject) => {
      const req = http.request(
        { hostname: "127.0.0.1", port, path, method: "GET" },
        (res) => {
          let raw = "";
          res.on("data", (c: Buffer) => { raw += c; });
          res.on("end", () => {
            try { resolve(JSON.parse(raw)); }
            catch { resolve({ raw }); }
          });
        },
      );
      req.on("error", reject);
      req.setTimeout(10_000, () => { req.destroy(new Error("Request timed out")); });
      req.end();
    });
  }
}

// ---------------------------------------------------------------------------
// Plugin registration
// ---------------------------------------------------------------------------

export default function register(api: OpenClawPluginApi): void {
  const raw = (api.pluginConfig ?? {}) as Record<string, unknown>;

  if (raw.enabled === false) {
    api.logger.info("[discord-voice] Plugin disabled in config");
    return;
  }

  const cfg: DiscordVoiceConfig = {
    botToken: (raw.botToken as string) ?? "",
    guildIds: (raw.guildIds as number[]) ?? [],
    transcriptChannelId: raw.transcriptChannelId as string | undefined,
    llmModel: raw.llmModel as string | undefined,
    llmUrl: raw.llmUrl as string | undefined,
    whisperUrl: raw.whisperUrl as string | undefined,
    kokoroUrl: raw.kokoroUrl as string | undefined,
    ttsVoice: raw.ttsVoice as string | undefined,
    botName: raw.botName as string | undefined,
    mainAgentName: raw.mainAgentName as string | undefined,
    defaultLocation: raw.defaultLocation as string | undefined,
    defaultTimezone: raw.defaultTimezone as string | undefined,
    extraContext: raw.extraContext as string | undefined,
    whisperPrompt: raw.whisperPrompt as string | undefined,
    vadMinSpeechMs: raw.vadMinSpeechMs as number | undefined,
    speechEndDelayMs: raw.speechEndDelayMs as number | undefined,
    channelContextMessages: raw.channelContextMessages as number | undefined,
    ttsReadChannel: raw.ttsReadChannel as boolean | undefined,
    correctionsFile: raw.correctionsFile as string | undefined,
    controlPort: (raw.controlPort as number | undefined) ?? 18790,
    pythonPath: raw.pythonPath as string | undefined,
  };

  if (!cfg.botToken) {
    api.logger.error("[discord-voice] botToken is required but not configured");
    return;
  }
  if (!cfg.guildIds.length) {
    api.logger.warn("[discord-voice] guildIds is empty — bot will not join any guilds");
  }

  // Derive gateway HTTP port from OpenClaw config
  const gatewayPort: number = (api.config as any)?.gateway?.port ?? 18789;

  let botManager: VoiceBotProcess | null = null;

  // -------------------------------------------------------------------------
  // Escalation HTTP route — registered on OpenClaw gateway HTTP server
  // -------------------------------------------------------------------------
  api.registerHttpRoute({
    path: "/rpc/discord-voice.escalate",
    handler: async (req, res) => {
      if (req.method !== "POST") {
        sendJson(res, 405, { error: "Method Not Allowed" });
        return;
      }
      let body: unknown;
      try {
        body = await readJsonBody(req);
      } catch {
        sendJson(res, 400, { error: "Invalid JSON body" });
        return;
      }

      const params = body as Record<string, unknown>;
      const message = typeof params.message === "string" ? params.message : "";
      const guildId = String(params.guildId ?? "");
      const channelId = String(params.channelId ?? "");
      const userId = String(params.userId ?? "");

      if (!message) {
        sendJson(res, 400, { error: "message is required" });
        return;
      }

      try {
        const text = await invokeMainAgent({
          message,
          guildId,
          channelId,
          userId,
          cfg: api.config,
        });
        sendJson(res, 200, { text });
      } catch (err) {
        api.logger.error(`[discord-voice] Escalation error: ${err instanceof Error ? err.message : String(err)}`);
        sendJson(res, 500, { error: "Agent invocation failed", text: "I'm having trouble connecting right now." });
      }
    },
  });

  // -------------------------------------------------------------------------
  // Background service
  // -------------------------------------------------------------------------
  api.registerService({
    id: "discord-voice-bot",
    start: async () => {
      const pythonPath = await findPython(cfg.pythonPath);
      api.logger.info(`[discord-voice] Using Python: ${pythonPath}`);

      botManager = new VoiceBotProcess({
        config: cfg,
        pythonPath,
        logger: api.logger,
        gatewayPort,
      });

      await botManager.start();
    },
    stop: async () => {
      if (botManager) {
        await botManager.stop();
        botManager = null;
      }
    },
  });

  // -------------------------------------------------------------------------
  // Agent tools (optional — must be allow-listed by the agent config)
  // -------------------------------------------------------------------------

  api.registerTool(
    {
      name: "voice_join",
      label: "Voice Join",
      description: "Join a Discord voice channel. The voice bot will connect and start listening.",
      parameters: Type.Object({
        guild_id: Type.Number({ description: "Discord guild (server) ID" }),
        channel_id: Type.Number({ description: "Discord voice channel ID" }),
      }),
      async execute(_id, params) {
        if (!botManager) {
          return { content: [{ type: "text" as const, text: "Voice bot is not running." }] };
        }
        try {
          const result = await botManager.httpPost("/control/join", {
            guild_id: params.guild_id,
            channel_id: params.channel_id,
          });
          return { content: [{ type: "text" as const, text: JSON.stringify(result) }] };
        } catch (err) {
          return { content: [{ type: "text" as const, text: `Failed to join: ${err instanceof Error ? err.message : String(err)}` }] };
        }
      },
    },
    { optional: true },
  );

  api.registerTool(
    {
      name: "voice_leave",
      label: "Voice Leave",
      description: "Disconnect the voice bot from a Discord voice channel.",
      parameters: Type.Object({
        guild_id: Type.Number({ description: "Discord guild (server) ID" }),
      }),
      async execute(_id, params) {
        if (!botManager) {
          return { content: [{ type: "text" as const, text: "Voice bot is not running." }] };
        }
        try {
          const result = await botManager.httpPost("/control/leave", { guild_id: params.guild_id });
          return { content: [{ type: "text" as const, text: JSON.stringify(result) }] };
        } catch (err) {
          return { content: [{ type: "text" as const, text: `Failed to leave: ${err instanceof Error ? err.message : String(err)}` }] };
        }
      },
    },
    { optional: true },
  );

  api.registerTool(
    {
      name: "voice_speak",
      label: "Voice Speak",
      description: "Synthesize text and play it in a Discord voice channel.",
      parameters: Type.Object({
        guild_id: Type.Number({ description: "Discord guild (server) ID" }),
        text: Type.String({ description: "Text to synthesize and play" }),
      }),
      async execute(_id, params) {
        if (!botManager) {
          return { content: [{ type: "text" as const, text: "Voice bot is not running." }] };
        }
        try {
          const result = await botManager.httpPost("/control/speak", {
            guild_id: params.guild_id,
            text: params.text,
          });
          return { content: [{ type: "text" as const, text: JSON.stringify(result) }] };
        } catch (err) {
          return { content: [{ type: "text" as const, text: `Failed to speak: ${err instanceof Error ? err.message : String(err)}` }] };
        }
      },
    },
    { optional: true },
  );

  api.registerTool(
    {
      name: "voice_status",
      label: "Voice Status",
      description: "Get the current status of the Discord voice bot (connected guilds, uptime).",
      parameters: Type.Object({}),
      async execute(_id, _params) {
        if (!botManager) {
          return { content: [{ type: "text" as const, text: "Voice bot is not running." }] };
        }
        try {
          const status = await botManager.httpGet("/health");
          return { content: [{ type: "text" as const, text: JSON.stringify(status, null, 2) }] };
        } catch (err) {
          return { content: [{ type: "text" as const, text: `Failed to get status: ${err instanceof Error ? err.message : String(err)}` }] };
        }
      },
    },
    { optional: true },
  );
}
