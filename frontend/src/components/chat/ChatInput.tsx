"use client";

import { useState, useRef, useCallback } from "react";
import { Send, Loader2, FileText, FileCode, X } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ChatPayload {
  message: string;
  execution_mode?: "local" | "modal";
  llm_backend?: "claude" | "nemotron";
  pdfs?: { filename: string; content_base64: string }[];
  custom_algorithm_file?: { filename: string; content_base64: string };
}

interface ChatInputProps {
  onSubmit: (payload: ChatPayload) => void;
  isLoading: boolean;
  /** True for the very first message (show attachment hint) */
  isFirstMessage?: boolean;
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result.includes(",") ? result.split(",")[1] : result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export function ChatInput({
  onSubmit,
  isLoading,
  isFirstMessage = false,
}: ChatInputProps) {
  const [message, setMessage] = useState("");
  const [executionMode, setExecutionMode] = useState<"local" | "modal">("local");
  const [llmBackend, setLlmBackend] = useState<"claude" | "nemotron">("claude");
  const [pdfFiles, setPdfFiles] = useState<File[]>([]);
  const [pyFile, setPyFile] = useState<File | null>(null);
  const pdfRef = useRef<HTMLInputElement>(null);
  const pyRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = useCallback(
    async (e?: React.FormEvent) => {
      e?.preventDefault();
      const trimmed = message.trim();
      if (!trimmed || isLoading) return;

      const pdfs =
        pdfFiles.length > 0
          ? await Promise.all(
              pdfFiles.map(async (f) => ({
                filename: f.name,
                content_base64: await fileToBase64(f),
              })),
            )
          : undefined;

      let custom_algorithm_file: ChatPayload["custom_algorithm_file"];
      if (pyFile) {
        custom_algorithm_file = {
          filename: pyFile.name,
          content_base64: await fileToBase64(pyFile),
        };
      }

      onSubmit({
        message: trimmed,
        execution_mode: isFirstMessage ? executionMode : undefined,
        llm_backend: isFirstMessage ? llmBackend : undefined,
        pdfs: isFirstMessage ? pdfs : undefined,
        custom_algorithm_file: isFirstMessage ? custom_algorithm_file : undefined,
      });
      setMessage("");
      if (isFirstMessage) {
        setPdfFiles([]);
        setPyFile(null);
      }
    },
    [message, executionMode, llmBackend, pdfFiles, pyFile, isLoading, onSubmit, isFirstMessage],
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    for (const f of files) {
      if (f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf")) {
        setPdfFiles((prev) => [...prev, f]);
      } else if (f.name.endsWith(".py")) {
        setPyFile(f);
      }
    }
  }, []);

  const hasAttachments = pdfFiles.length > 0 || pyFile !== null;

  return (
    <form
      onSubmit={handleSubmit}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      className="w-full"
    >
      <div
        className={cn(
          "rounded-2xl border bg-card/60 backdrop-blur-sm overflow-hidden transition-colors",
          "focus-within:border-[var(--accent)]/40 focus-within:shadow-[0_0_0_1px_var(--terminal-glow)]",
          "border-border-strong",
        )}
      >
        {/* First-turn setup controls */}
        {isFirstMessage && (
          <div className="px-4 pt-3 pb-1 border-b border-border/60 space-y-3">
            {/* Row of toggles */}
            <div className="flex flex-wrap items-start gap-x-6 gap-y-3">
              {/* Execution mode toggle */}
              <div>
                <p className="text-[11px] uppercase tracking-wider text-muted-foreground mb-1.5">
                  Execution mode
                </p>
                <div className="inline-flex rounded-lg border border-border overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setExecutionMode("local")}
                    className={cn(
                      "px-3 py-1.5 text-xs font-medium transition-colors",
                      executionMode === "local"
                        ? "bg-[var(--accent)] text-[var(--accent-foreground)]"
                        : "text-muted-foreground hover:text-foreground hover:bg-secondary",
                    )}
                  >
                    local
                  </button>
                  <button
                    type="button"
                    onClick={() => setExecutionMode("modal")}
                    className={cn(
                      "px-3 py-1.5 text-xs font-medium border-l border-border transition-colors",
                      executionMode === "modal"
                        ? "bg-[var(--accent)] text-[var(--accent-foreground)]"
                        : "text-muted-foreground hover:text-foreground hover:bg-secondary",
                    )}
                  >
                    modal
                  </button>
                </div>
              </div>

              {/* LLM backend toggle */}
              <div>
                <p className="text-[11px] uppercase tracking-wider text-muted-foreground mb-1.5">
                  LLM Backend
                </p>
                <div className="inline-flex rounded-lg border border-border overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setLlmBackend("claude")}
                    className={cn(
                      "px-3 py-1.5 text-xs font-medium transition-colors",
                      llmBackend === "claude"
                        ? "bg-[var(--accent)] text-[var(--accent-foreground)]"
                        : "text-muted-foreground hover:text-foreground hover:bg-secondary",
                    )}
                  >
                    Claude Sonnet
                  </button>
                  <button
                    type="button"
                    onClick={() => setLlmBackend("nemotron")}
                    className={cn(
                      "px-3 py-1.5 text-xs font-medium border-l border-border transition-colors",
                      llmBackend === "nemotron"
                        ? "bg-[var(--accent)] text-[var(--accent-foreground)]"
                        : "text-muted-foreground hover:text-foreground hover:bg-secondary",
                    )}
                  >
                    Nemotron
                  </button>
                </div>
              </div>
            </div>

            {/* File upload buttons */}
            <div className="flex flex-wrap items-center gap-2">
              <input
                ref={pdfRef}
                type="file"
                accept="*/*"
                multiple
                onChange={(e) => {
                  const files = e.target.files;
                  if (files) {
                    const valid = Array.from(files).filter(
                      (f) => f.name.toLowerCase().endsWith(".pdf"),
                    );
                    if (valid.length) setPdfFiles((prev) => [...prev, ...valid]);
                  }
                  e.target.value = "";
                }}
                className="hidden"
              />
              <button
                type="button"
                onClick={() => pdfRef.current?.click()}
                className="rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-secondary"
              >
                Upload PDF(s)
              </button>

              <input
                ref={pyRef}
                type="file"
                accept=".py"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f && f.name.toLowerCase().endsWith(".py")) setPyFile(f);
                  e.target.value = "";
                }}
                className="hidden"
              />
              <button
                type="button"
                onClick={() => pyRef.current?.click()}
                className="rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-secondary"
              >
                Upload custom .py
              </button>
            </div>
          </div>
        )}

        {/* Attachment chips */}
        {isFirstMessage && hasAttachments && (
          <div className="flex flex-wrap gap-1.5 px-4 pt-3">
            {pdfFiles.map((f, i) => (
              <span
                key={`pdf-${i}`}
                className="inline-flex items-center gap-1.5 rounded-md bg-secondary px-2 py-1 text-xs text-muted-foreground"
              >
                <FileText className="w-3 h-3" />
                <span className="max-w-[120px] truncate">{f.name}</span>
                <button
                  type="button"
                  onClick={() =>
                    setPdfFiles((prev) => prev.filter((_, j) => j !== i))
                  }
                  className="p-0.5 rounded hover:bg-zinc-700"
                >
                  <X className="w-2.5 h-2.5" />
                </button>
              </span>
            ))}
            {pyFile && (
              <span className="inline-flex items-center gap-1.5 rounded-md bg-[var(--accent)]/10 border border-[var(--accent)]/20 px-2 py-1 text-xs text-[var(--accent)]">
                <FileCode className="w-3 h-3" />
                <span className="max-w-[120px] truncate">{pyFile.name}</span>
                <button
                  type="button"
                  onClick={() => setPyFile(null)}
                  className="p-0.5 rounded hover:bg-[var(--accent)]/20"
                >
                  <X className="w-2.5 h-2.5" />
                </button>
              </span>
            )}
          </div>
        )}

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            isFirstMessage
              ? "Describe your problem, attach PDFs and a .py algorithm file..."
              : "Type a message..."
          }
          rows={isFirstMessage ? 3 : 1}
          disabled={isLoading}
          className={cn(
            "w-full bg-transparent px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground/60",
            "resize-none focus:outline-none disabled:opacity-50",
          )}
        />

        {/* Bottom bar */}
        <div className="flex items-center justify-between px-3 pb-2">
          <div className="text-[11px] text-muted-foreground/60 px-1">
            {isFirstMessage
              ? "First input: mode + LLM + PDFs + custom .py + problem description"
              : "Follow-up step"}
          </div>

          <button
            type="submit"
            disabled={!message.trim() || isLoading}
            className={cn(
              "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-all",
              "bg-[var(--accent)] text-[var(--accent-foreground)]",
              "hover:opacity-90 active:scale-[0.97]",
              "disabled:opacity-30 disabled:pointer-events-none",
            )}
          >
            {isLoading ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Send className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      </div>
    </form>
  );
}
