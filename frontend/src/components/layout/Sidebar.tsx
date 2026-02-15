"use client";

import { useState } from "react";
import {
  FlaskConical,
  Plus,
  MessageSquare,
  PanelLeftClose,
  PanelLeft,
  Github,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface Session {
  id: string;
  label: string;
  active?: boolean;
}

interface SidebarProps {
  sessions: Session[];
  onNewSession: () => void;
  onSelectSession: (id: string) => void;
  activeSessionId: string | null;
}

export function Sidebar({
  sessions,
  onNewSession,
  onSelectSession,
  activeSessionId,
}: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "h-screen sticky top-0 flex flex-col border-r border-border bg-[#0c0c0f] transition-all duration-200 ease-out",
        collapsed ? "w-[52px]" : "w-[260px]",
      )}
    >
      {/* ── Header ──────────────────────────────────── */}
      <div className="flex items-center justify-between h-14 px-3 border-b border-border shrink-0">
        {!collapsed && (
          <div className="flex items-center gap-2 min-w-0">
            <FlaskConical className="w-4 h-4 text-[var(--accent)] shrink-0" />
            <span className="text-sm font-semibold tracking-tight truncate">
              Benchwarmer
            </span>
          </div>
        )}
        <button
          onClick={() => setCollapsed((c) => !c)}
          className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? (
            <PanelLeft className="w-4 h-4" />
          ) : (
            <PanelLeftClose className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* ── New session button ──────────────────────── */}
      <div className="px-2 pt-2">
        <button
          onClick={onNewSession}
          className={cn(
            "w-full flex items-center gap-2 rounded-lg text-sm font-medium transition-colors",
            "border border-border-strong bg-secondary/60 hover:bg-secondary text-foreground",
            collapsed ? "justify-center p-2" : "px-3 py-2",
          )}
        >
          <Plus className="w-4 h-4 shrink-0" />
          {!collapsed && <span>New experiment</span>}
        </button>
      </div>

      {/* ── Session list ────────────────────────────── */}
      <div className="flex-1 overflow-y-auto py-3 px-2 space-y-0.5">
        {!collapsed && sessions.length > 0 && (
          <p className="px-2 pb-1.5 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
            Recent
          </p>
        )}
        {sessions.map((s) => (
          <button
            key={s.id}
            onClick={() => onSelectSession(s.id)}
            className={cn(
              "w-full flex items-center gap-2 rounded-lg text-sm transition-colors",
              collapsed ? "justify-center p-2" : "px-3 py-1.5",
              s.id === activeSessionId
                ? "bg-secondary text-foreground"
                : "text-muted-foreground hover:text-foreground hover:bg-secondary/50",
            )}
          >
            <MessageSquare className="w-3.5 h-3.5 shrink-0" />
            {!collapsed && (
              <span className="truncate text-left">{s.label}</span>
            )}
          </button>
        ))}
      </div>

      {/* ── Footer ──────────────────────────────────── */}
      <div className="border-t border-border px-2 py-2">
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className={cn(
            "w-full flex items-center gap-2 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-colors",
            collapsed ? "justify-center p-2" : "px-3 py-1.5",
          )}
        >
          <Github className="w-3.5 h-3.5 shrink-0" />
          {!collapsed && <span>GitHub</span>}
        </a>
      </div>
    </aside>
  );
}
