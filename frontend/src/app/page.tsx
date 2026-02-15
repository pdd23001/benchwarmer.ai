"use client";

import { useState, useCallback, useRef } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { ChatView } from "@/components/chat/ChatView";

interface Session {
  id: string;
  label: string;
}

export default function Home() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  /** Key that forces ChatView to re-mount ONLY when user clicks New/Select */
  const [viewKey, setViewKey] = useState(0);

  /** Pending label set before session id is known */
  const pendingLabel = useRef<string>("New experiment");

  const handleNewSession = useCallback(() => {
    setActiveId(null);
    pendingLabel.current = "New experiment";
    setViewKey((k) => k + 1);
  }, []);

  const handleSelectSession = useCallback((id: string) => {
    setActiveId(id);
    setViewKey((k) => k + 1);
  }, []);

  /** Called by ChatView when backend returns a session_id for the first time */
  const handleSessionId = useCallback((id: string) => {
    setActiveId(id);
    setSessions((prev) => {
      if (prev.find((s) => s.id === id)) return prev;
      return [{ id, label: pendingLabel.current }, ...prev];
    });
  }, []);

  /** Called by ChatView with the first message text, for sidebar label */
  const handleSessionLabel = useCallback((label: string) => {
    const trimmed = label.slice(0, 50);
    pendingLabel.current = trimmed;
    // If session is already in the list, update it
    setSessions((prev) =>
      prev.map((s) =>
        s.id === activeId ? { ...s, label: trimmed } : s,
      ),
    );
  }, [activeId]);

  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Sidebar
        sessions={sessions}
        onNewSession={handleNewSession}
        onSelectSession={handleSelectSession}
        activeSessionId={activeId}
      />
      <main className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
        <ChatView
          key={viewKey}
          sessionId={activeId}
          onSessionId={handleSessionId}
          onSessionLabel={handleSessionLabel}
        />
      </main>
    </div>
  );
}
