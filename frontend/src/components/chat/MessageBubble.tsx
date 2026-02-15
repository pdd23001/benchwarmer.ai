"use client";

import { useMemo } from "react";
import { User, Bot, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  plotImage?: string; // base64 data URI from matplotlib
  error?: string;
}

// ─── Markdown table parser ──────────────────────────────────────────────────

function renderTable(headerLine: string, rows: string[]): string {
  const parseRow = (line: string) =>
    line
      .split("|")
      .slice(1, -1) // trim leading/trailing empty splits
      .map((c) => c.trim());

  const headers = parseRow(headerLine);
  const bodyRows = rows.map(parseRow);

  let html =
    '<div class="overflow-x-auto my-3"><table class="w-full text-xs font-mono border-collapse">';

  // Header
  html += "<thead><tr>";
  for (const h of headers) {
    html += `<th class="text-left px-3 py-2 border-b border-border text-muted-foreground font-semibold whitespace-nowrap">${h}</th>`;
  }
  html += "</tr></thead>";

  // Body
  html += "<tbody>";
  for (const row of bodyRows) {
    html += '<tr class="border-b border-border/40 hover:bg-secondary/30">';
    for (let i = 0; i < headers.length; i++) {
      const cell = row[i] ?? "";
      html += `<td class="px-3 py-1.5 whitespace-nowrap">${cell}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table></div>";

  return html;
}

// ─── Markdown → HTML ────────────────────────────────────────────────────────

function renderMarkdown(text: string): string {
  const lines = text.split("\n");
  const output: string[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Detect markdown table: current line has |, next line is separator (|---|...)
    if (
      line.trim().startsWith("|") &&
      i + 1 < lines.length &&
      /^\|[\s\-:|]+\|$/.test(lines[i + 1].trim())
    ) {
      const headerLine = line;
      i += 2; // skip header + separator
      const bodyRows: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        bodyRows.push(lines[i]);
        i++;
      }
      output.push(renderTable(headerLine, bodyRows));
      continue;
    }

    // Regular line — apply inline markdown
    let escaped = line
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    escaped = escaped
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      .replace(
        /`(.+?)`/g,
        '<code class="rounded bg-secondary px-1 py-0.5 text-xs font-mono">$1</code>',
      );

    output.push(escaped);
    i++;
  }

  return output.join("<br />");
}

// ─── Component ──────────────────────────────────────────────────────────────

export function MessageBubble({
  role,
  content,
  plotImage,
  error,
}: MessageBubbleProps) {
  const isUser = role === "user";
  const html = useMemo(
    () => (content ? renderMarkdown(content) : ""),
    [content],
  );

  return (
    <div
      className={cn(
        "flex gap-3 w-full animate-in",
        isUser ? "flex-row-reverse" : "flex-row",
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg",
          isUser
            ? "bg-secondary text-muted-foreground"
            : "bg-[var(--accent)]/10 text-[var(--accent)]",
        )}
      >
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      {/* Bubble */}
      <div
        className={cn(
          "max-w-[85%] min-w-0",
          isUser ? "text-right" : "text-left",
        )}
      >
        {/* Error */}
        {error && (
          <div className="flex items-start gap-2 rounded-xl bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-400">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {/* Text content */}
        {html && (
          <div
            className={cn(
              "rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
              isUser
                ? "bg-secondary text-foreground"
                : "bg-transparent text-foreground/90",
            )}
            dangerouslySetInnerHTML={{ __html: html }}
          />
        )}

        {/* Inline plot image (matplotlib) */}
        {plotImage && (
          <div className="mt-3 rounded-xl overflow-hidden border border-border/60 bg-white">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={plotImage}
              alt="Benchmark visualization"
              className="w-full h-auto"
            />
          </div>
        )}
      </div>
    </div>
  );
}
