"use client";

import { useEffect, useRef } from "react";
import { Terminal as TerminalIcon, Maximize2, Minimize2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface LogEntry {
    timestamp: string;
    message: string;
    type: "info" | "error" | "success";
}

interface LiveTerminalProps {
    logs: LogEntry[];
    isVisible: boolean;
}

export function LiveTerminal({ logs, isVisible }: LiveTerminalProps) {
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (isVisible && bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [logs, isVisible]);

    if (!isVisible && logs.length === 0) return null;

    return (
        <div className="w-full rounded-xl overflow-hidden border border-border bg-[#0c0c0c] text-green-400 font-mono text-sm shadow-xl">
            <div className="flex items-center justify-between px-4 py-2 bg-[#1a1a1a] border-b border-border/20">
                <div className="flex items-center gap-2 text-muted-foreground">
                    <TerminalIcon className="w-4 h-4" />
                    <span className="text-xs font-semibold uppercase tracking-wider">Live Logs</span>
                </div>
                <div className="flex gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500/20"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-green-500/20"></div>
                </div>
            </div>

            <div className="p-4 h-[300px] overflow-y-auto space-y-1 scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent">
                {logs.map((log, i) => (
                    <div key={i} className={cn("flex gap-3", log.type === "error" && "text-red-400", log.type === "success" && "text-blue-400")}>
                        <span className="text-zinc-600 shrink-0 select-none">[{log.timestamp}]</span>
                        <span className="break-all">{log.message}</span>
                    </div>
                ))}
                <div ref={bottomRef} />

                {logs.length === 0 && (
                    <div className="text-zinc-600 italic">Waiting for experiment to start...</div>
                )}
            </div>
        </div>
    );
}
