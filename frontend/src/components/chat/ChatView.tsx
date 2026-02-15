"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ChatInput, type ChatPayload } from "./ChatInput";
import { MessageBubble } from "./MessageBubble";
import { TerminalBlock, type TerminalLine } from "./TerminalBlock";
import { FlaskConical, Sparkles, ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";

// ─── Types ──────────────────────────────────────────────────────────────────

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  plotImage?: string; // base64 data URI from matplotlib
  error?: string;
}

interface ChatViewProps {
  sessionId: string | null;
  onSessionId: (id: string) => void;
  onSessionLabel: (label: string) => void;
}

// ─── Tool label mapping ─────────────────────────────────────────────────────

const TOOL_LABELS: Record<string, string> = {
  run_intake: "Analyzing problem description & PDFs",
  code_algorithm: "Generating algorithm implementation",
  use_generators: "Configuring instance generators",
  run_benchmark: "Executing benchmark",
  modify_generators: "Modifying generators",
  modify_execution_config: "Updating execution config",
  load_suite: "Loading benchmark suite",
  load_custom_instances: "Loading custom instances",
  analyze_results: "Generating visualization",
  show_status: "Checking pipeline status",
  set_execution_mode: "Setting execution mode",
  remove_algorithm: "Removing algorithm",
  go_back: "Resetting pipeline state",
  export_results: "Exporting results",
};

// ─── Quick-start suggestions ────────────────────────────────────────────────

const SUGGESTIONS = [
  "Benchmark greedy vs random vertex cover on Erdos-Renyi graphs",
  "Compare max-cut algorithms from my paper",
  "Run minimum vertex cover with Biq Mac suite instances",
];

// ─── Component ──────────────────────────────────────────────────────────────

export function ChatView({
  sessionId,
  onSessionId,
  onSessionLabel,
}: ChatViewProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [started, setStarted] = useState(false);
  const [terminalLines, setTerminalLines] = useState<TerminalLine[]>([]);
  const [terminalStatus, setTerminalStatus] = useState<
    "running" | "done" | "error"
  >("done");
  const bottomRef = useRef<HTMLDivElement>(null);
  const sessionRef = useRef<string | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading, terminalLines]);

  // ── SSE stream reader ───────────────────────────────────────────────────

  const processStream = useCallback(
    async (response: Response): Promise<{
      reply: string;
      plotImage?: string;
      session_id?: string;
      error?: string;
    }> => {
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let result: {
        reply: string;
        plotImage?: string;
        session_id?: string;
        error?: string;
      } = { reply: "" };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by double newlines
        const parts = buffer.split("\n\n");
        buffer = parts.pop()!; // keep incomplete part

        for (const part of parts) {
          for (const line of part.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            let event: Record<string, unknown>;
            try {
              event = JSON.parse(line.slice(6));
            } catch {
              continue;
            }

            switch (event.type) {
              case "thinking":
                setTerminalLines((prev) => {
                  // Avoid duplicate "Querying LLM..." lines
                  const last = prev[prev.length - 1];
                  if (last?.text === "> Querying LLM...") return prev;
                  return [
                    ...prev,
                    { text: "> Querying LLM...", type: "dim" },
                  ];
                });
                break;

              case "tool_start": {
                const tool = event.tool as string;
                const label = TOOL_LABELS[tool] || tool;
                setTerminalLines((prev) => [
                  ...prev,
                  { text: `$ ${label}`, type: "info" },
                ]);
                break;
              }

              case "tool_end": {
                const toolResult = (event.result as string) || "";
                // Show the first non-empty line of the result
                const firstLine =
                  toolResult.split("\n").find((l) => l.trim()) ||
                  "Done";
                setTerminalLines((prev) => [
                  ...prev,
                  { text: `  ${firstLine}`, type: "success" },
                ]);
                break;
              }

              case "done": {
                const reply = (event.reply as string) || "";
                const sid = event.session_id as string | undefined;
                const plotImg = (event.plot_image as string) || undefined;

                result = {
                  reply,
                  session_id: sid,
                  plotImage: plotImg,
                };

                setTerminalLines((prev) => [
                  ...prev,
                  { text: "✓ Turn complete", type: "success" },
                ]);
                setTerminalStatus("done");
                break;
              }

              case "error": {
                const errMsg = (event.error as string) || "Unknown error";
                // If session expired, clear it so user can start fresh
                if (errMsg.toLowerCase().includes("session expired")) {
                  sessionRef.current = null;
                }
                result = {
                  reply: "",
                  error: errMsg,
                };
                setTerminalLines((prev) => [
                  ...prev,
                  { text: `✗ Error: ${errMsg}`, type: "error" },
                ]);
                setTerminalStatus("error");
                break;
              }

              case "heartbeat":
                // Keep-alive, ignore
                break;
            }
          }
        }
      }

      return result;
    },
    [],
  );

  // ── Send message ────────────────────────────────────────────────────────

  const sendMessage = useCallback(
    async (payload: ChatPayload) => {
      setStarted(true);

      // Build visible summary for user bubble
      const parts = [payload.message];
      if (payload.execution_mode)
        parts.push(`[mode: ${payload.execution_mode}]`);
      if (payload.llm_backend)
        parts.push(`[llm: ${payload.llm_backend}]`);
      if (payload.pdfs?.length) parts.push(`[${payload.pdfs.length} PDF(s)]`);
      if (payload.custom_algorithm_file)
        parts.push(`[${payload.custom_algorithm_file.filename}]`);

      const userMsg: Message = {
        id: `u-${Date.now()}`,
        role: "user",
        content: parts.join("  "),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      // Reset terminal for this turn
      setTerminalLines([
        { text: "$ benchwarmer orchestrator", type: "dim" },
        {
          text: `> "${payload.message.slice(0, 80)}${payload.message.length > 80 ? "..." : ""}"`,
          type: "info",
        },
      ]);
      setTerminalStatus("running");

      // Session label from first message
      if (!sessionRef.current) {
        onSessionLabel(payload.message.slice(0, 40));
      }

      try {
        // Build request body
        const body: Record<string, unknown> = {
          message: payload.message,
        };
        if (sessionRef.current) {
          body.session_id = sessionRef.current;
        } else {
          if (payload.execution_mode) body.execution_mode = payload.execution_mode;
          if (payload.llm_backend) body.llm_backend = payload.llm_backend;
          if (payload.pdfs?.length) body.pdfs = payload.pdfs;
          if (payload.custom_algorithm_file)
            body.custom_algorithm_file = payload.custom_algorithm_file;
        }

        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });

        if (!res.ok) {
          // Non-stream error (setup failure)
          const data = await res.json().catch(() => ({}));
          const errMsg =
            data.error ?? data.detail ?? `HTTP ${res.status}`;
          setTerminalLines((prev) => [
            ...prev,
            { text: `✗ ${errMsg}`, type: "error" },
          ]);
          setTerminalStatus("error");
          setMessages((prev) => [
            ...prev,
            {
              id: `a-${Date.now()}`,
              role: "assistant",
              content: "",
              error: errMsg,
            },
          ]);
          return;
        }

        // Parse SSE stream
        const result = await processStream(res);

        // Store session id
        if (result.session_id) {
          sessionRef.current = result.session_id;
          onSessionId(result.session_id);
        }

        const assistantMsg: Message = {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: result.reply,
          plotImage: result.plotImage,
          error: result.error,
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (e) {
        const errMsg =
          e instanceof Error ? e.message : "Network error";
        setTerminalLines((prev) => [
          ...prev,
          { text: `✗ ${errMsg}`, type: "error" },
        ]);
        setTerminalStatus("error");
        setMessages((prev) => [
          ...prev,
          {
            id: `a-${Date.now()}`,
            role: "assistant",
            content: "",
            error: errMsg,
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [onSessionId, onSessionLabel, processStream],
  );

  const showSetup = !started && messages.length === 0;

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* ── Messages area ────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 pt-6 pb-32 space-y-5">
          {/* Empty state */}
          {showSetup && !isLoading && (
            <div className="flex flex-col items-center justify-center pt-20 text-center animate-in">
              <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-[var(--accent)]/10 mb-6">
                <FlaskConical className="w-7 h-7 text-[var(--accent)]" />
              </div>
              <h2 className="text-2xl font-semibold tracking-tight text-foreground mb-2">
                What would you like to benchmark?
              </h2>
              <p className="text-muted-foreground text-sm max-w-md mb-8 leading-relaxed">
                Upload your PDFs and custom algorithm, choose execution mode,
                and describe the problem. Then follow the steps — just like the
                CLI.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => sendMessage({ message: s })}
                    className={cn(
                      "group flex items-center gap-2 rounded-xl border border-border px-4 py-2 text-sm",
                      "text-muted-foreground hover:text-foreground hover:bg-secondary hover:border-border-strong",
                      "transition-all duration-150",
                    )}
                  >
                    <Sparkles className="w-3.5 h-3.5 text-[var(--accent)]/60 group-hover:text-[var(--accent)]" />
                    <span>{s}</span>
                    <ArrowRight className="w-3 h-3 opacity-0 -translate-x-1 group-hover:opacity-60 group-hover:translate-x-0 transition-all" />
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((m) => (
            <MessageBubble
              key={m.id}
              role={m.role}
              content={m.content}
              plotImage={m.plotImage}
              error={m.error}
            />
          ))}

          {/* Loading indicator */}
          {isLoading && (
            <div className="flex gap-3 animate-in">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--accent)]/10 text-[var(--accent)]">
                <div className="w-4 h-4 rounded-full border-2 border-[var(--accent)]/30 border-t-[var(--accent)] animate-spin" />
              </div>
              <div className="rounded-2xl px-4 py-2.5 text-sm text-muted-foreground">
                Thinking...
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* ── Terminal strip (shows during/after activity) ──────── */}
      {terminalLines.length > 0 && (
        <div className="border-t border-border bg-[var(--terminal-bg)]">
          <div className="max-w-3xl mx-auto px-4">
            <TerminalBlock
              lines={terminalLines}
              status={terminalStatus}
              className="border-0 rounded-none shadow-none"
            />
          </div>
        </div>
      )}

      {/* ── Input ────────────────────────────────────────────── */}
      <div className="border-t border-border bg-background/90 backdrop-blur-sm py-3 px-4">
        <div className="max-w-3xl mx-auto">
          <ChatInput
            onSubmit={sendMessage}
            isLoading={isLoading}
            isFirstMessage={showSetup}
          />
        </div>
      </div>
    </div>
  );
}
