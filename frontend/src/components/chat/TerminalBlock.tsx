"use client";

import { useEffect, useRef } from "react";
import { Terminal } from "lucide-react";
import { cn } from "@/lib/utils";

export interface TerminalLine {
  text: string;
  type?: "info" | "success" | "error" | "dim";
}

interface TerminalBlockProps {
  title?: string;
  lines: TerminalLine[];
  status?: "running" | "done" | "error";
  className?: string;
}

export function TerminalBlock({
  title = "Modal Sandbox",
  lines,
  status = "done",
  className,
}: TerminalBlockProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);

  return (
    <div
      className={cn(
        "rounded-xl overflow-hidden border",
        "bg-[var(--terminal-bg)] border-[var(--terminal-border)]",
        "shadow-[0_0_24px_var(--terminal-glow)]",
        className,
      )}
    >
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-[var(--terminal-header)] border-b border-[var(--terminal-border)]">
        <div className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-[var(--terminal-text)]" />
          <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--terminal-text)]/80">
            {title}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {status === "running" && (
            <span className="flex items-center gap-1.5 text-[10px] text-[var(--terminal-text)] font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--terminal-text)] pulse-dot" />
              running
            </span>
          )}
          {status === "done" && (
            <span className="flex items-center gap-1.5 text-[10px] text-[var(--terminal-text)]/60 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--terminal-text)]/40" />
              done
            </span>
          )}
          {status === "error" && (
            <span className="flex items-center gap-1.5 text-[10px] text-red-400 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
              error
            </span>
          )}
          {/* Window dots */}
          <div className="flex gap-1 ml-3">
            <span className="w-2 h-2 rounded-full bg-zinc-700" />
            <span className="w-2 h-2 rounded-full bg-zinc-700" />
            <span className="w-2 h-2 rounded-full bg-[var(--terminal-text)]/30" />
          </div>
        </div>
      </div>

      {/* Terminal body */}
      <div className="px-4 py-3 max-h-[280px] overflow-y-auto font-mono text-[13px] leading-[1.6] space-y-0.5">
        {lines.map((line, i) => (
          <div
            key={i}
            className={cn(
              "animate-in",
              line.type === "success" && "text-[var(--terminal-text)]",
              line.type === "error" && "text-red-400",
              line.type === "dim" && "text-zinc-600",
              (!line.type || line.type === "info") && "text-[var(--terminal-text)]/80",
            )}
            style={{ animationDelay: `${i * 30}ms` }}
          >
            {line.text}
          </div>
        ))}
        {status === "running" && (
          <span className="inline-block w-2 h-4 bg-[var(--terminal-text)] terminal-cursor" />
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}
