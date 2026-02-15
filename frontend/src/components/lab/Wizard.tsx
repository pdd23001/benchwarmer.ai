"use client";

import { useState, useRef, useEffect, ReactNode } from "react";
import {
    Upload, FileText, Play, Loader2, ArrowRight, User, Cpu, Terminal,
    X, Paperclip, Send, CheckCircle2, AlertCircle, BarChart3, Cloud, Layout
} from "lucide-react";
import { cn } from "@/lib/utils";
import { ResultsChart } from "./ResultsChart";

// ─── Types ───────────────────────────────────────────────────────────────────

interface SessionState {
    id: string | null;
    status: "idle" | "analyzing" | "intake" | "configuring" | "running" | "completed";
    problemClass: string | null;
    userAlgo: string | null;
    config: any | null;
    executionMode: "local" | "modal";
}

interface Challenger {
    id: string;
    type: "pdf" | "baseline";
    name: string;
    status: "pending" | "analyzing" | "ready" | "error";
    message?: string;
    implementation?: any;
}

interface Message {
    id: string;
    role: "system" | "user" | "agent";
    content: ReactNode | string;
    timestamp: number;
    actions?: ReactNode; // Interactive elements embedded in message
}

// ─── Main Wizard Component ───────────────────────────────────────────────────

export function Wizard() {
    // Core State
    const [session, setSession] = useState<SessionState>({
        id: null,
        status: "idle",
        problemClass: null,
        userAlgo: null,
        config: null,
        executionMode: "local"
    });

    const [challengers, setChallengers] = useState<Challenger[]>([]);

    // Chat State
    const [messages, setMessages] = useState<Message[]>([
        {
            id: "welcome",
            role: "agent",
            content: "Welcome to Benchwarmer.AI. To get started, please **upload your algorithm** (Python file) that you'd like to benchmark.",
            timestamp: Date.now()
        }
    ]);

    // Helpers
    const addMessage = (role: Message["role"], content: ReactNode | string, actions?: ReactNode) => {
        setMessages(prev => [...prev, {
            id: Math.random().toString(36).substring(7),
            role,
            content,
            timestamp: Date.now(),
            actions
        }]);
    };

    return (
        <div className="flex-1 flex flex-col h-full bg-background/50 backdrop-blur-sm overflow-hidden relative">
            {/* Header */}
            <header className="h-14 border-b bg-background/80 backdrop-blur sticky top-0 z-10 flex items-center justify-between px-6">
                <div className="flex items-center gap-3">
                    <div className="md:hidden font-bold">Benchwarmer.AI</div>
                    {session.problemClass && (
                        <div className="flex items-center gap-2 text-sm text-muted-foreground animate-in fade-in slide-in-from-left-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-primary" />
                            <span>{session.problemClass} Protocol</span>
                        </div>
                    )}
                </div>
                <div className="flex items-center gap-3">
                    {session.status !== "idle" && (
                        <div className="flex items-center gap-1.5 bg-secondary/50 px-3 py-1.5 rounded-md border text-xs font-medium">
                            <Cloud className={cn("w-3.5 h-3.5", session.executionMode === "modal" ? "text-blue-500" : "text-muted-foreground")} />
                            <span>{session.executionMode === "modal" ? "Cloud Execution" : "Local Execution"}</span>
                        </div>
                    )}
                </div>
            </header>

            {/* Main Content Area */}
            <div className="flex-1 overflow-hidden flex flex-col relative">
                {session.status === "running" || session.status === "completed" ? (
                    <LiveDashboard
                        session={session}
                        challengers={challengers}
                        onBack={() => setSession(s => ({ ...s, status: "intake" }))}
                    />
                ) : (
                    <ChatInterface
                        session={session}
                        setSession={setSession}
                        messages={messages}
                        addMessage={addMessage}
                        challengers={challengers}
                        setChallengers={setChallengers}
                    />
                )}
            </div>
        </div>
    );
}

// ─── Chat Interface ──────────────────────────────────────────────────────────

function ChatInterface({
    session, setSession, messages, addMessage, challengers, setChallengers
}: {
    session: SessionState,
    setSession: React.Dispatch<React.SetStateAction<SessionState>>,
    messages: Message[],
    addMessage: (role: Message["role"], content: ReactNode | string, actions?: ReactNode) => void,
    challengers: Challenger[],
    setChallengers: React.Dispatch<React.SetStateAction<Challenger[]>>
}) {
    const [inputValue, setInputValue] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isTyping]);

    // ─── Handlers ───

    // 1. Initial Algorithm Upload
    const handleAlgoUpload = async (file: File) => {
        setIsTyping(true);
        addMessage("user", <span className="flex items-center gap-2"><Upload className="w-4 h-4" /> Uploaded Algorithm: {file.name}</span>);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch("/api/python/session/start", { method: "POST", body: formData });
            if (!res.ok) throw new Error("Upload failed");

            const data = await res.json();

            setTimeout(() => {
                setSession(s => ({
                    ...s,
                    id: data.session_id,
                    problemClass: data.detected_class,
                    userAlgo: data.filename,
                    status: "intake"
                }));

                addMessage("agent", `I've analyzed **${data.filename}**. It appears to be a **${data.detected_class}** solver.`);
                setTimeout(() => {
                    addMessage("agent", "Now, please **upload any research papers (PDFs)** you'd like to benchmark against. I will generate competitor algorithms from them automatically.");
                    setIsTyping(false);
                }, 800);
            }, 1000);

        } catch (e) {
            addMessage("system", "Error uploading algorithm. Please try again.");
            setIsTyping(false);
        }
    };

    // 2. Challenger PDF Upload
    const handleChallengerUpload = async (file: File) => {
        if (!session.id) return;

        setIsTyping(true);
        addMessage("user", <span className="flex items-center gap-2"><FileText className="w-4 h-4" /> Uploaded Paper: {file.name}</span>);

        const formData = new FormData();
        formData.append("type", "pdf");
        formData.append("file", file);

        try {
            const res = await fetch(`/api/python/session/${session.id}/challenger`, { method: "POST", body: formData });
            const data = await res.json();

            setChallengers(prev => [...prev, {
                id: data.challenger_id,
                type: "pdf",
                name: file.name,
                status: "pending"
            }]);

            addMessage("agent", `Received **${file.name}**. I've added it to the challenger queue.`);
        } catch (e) {
            addMessage("system", "Failed to add challenger.");
        } finally {
            setIsTyping(false);
        }
    };

    // 3. User Text Input / Commands
    const handleSend = async () => {
        if (!inputValue.trim()) return;
        const text = inputValue;
        setInputValue("");

        addMessage("user", text);
        setIsTyping(true);

        // Simple Command Parser
        const lower = text.toLowerCase();

        if (session.status === "idle") {
            // Expecting upload, but got text
            setTimeout(() => {
                addMessage("agent", "Please upload your Python algorithm file to start the session.");
                setIsTyping(false);
            }, 500);
            return;
        }

        if (lower.includes("run") || lower.includes("start")) {
            // Transition to Configuration/Run
            addMessage("agent", "Preparing the benchmark environment...", (
                <div className="mt-3 bg-card border rounded-lg p-4 w-full max-w-sm animate-in fade-in zoom-in-95">
                    <h3 className="font-semibold text-sm mb-3">Execution Configuration</h3>
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                <Cloud className="w-4 h-4" /> Cloud Execution (Modal)
                            </div>
                            {/* Toggle Switch */}
                            <button
                                onClick={() => setSession(s => ({ ...s, executionMode: s.executionMode === "local" ? "modal" : "local" }))}
                                className={cn(
                                    "relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2",
                                    session.executionMode === "modal" ? "bg-primary" : "bg-input"
                                )}
                            >
                                <span className={cn(
                                    "inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform",
                                    session.executionMode === "modal" ? "translate-x-5" : "translate-x-1"
                                )} />
                            </button>
                        </div>
                        <button
                            onClick={() => {
                                setSession(s => ({ ...s, status: "running" }));
                            }}
                            className="w-full flex items-center justify-center gap-2 bg-primary text-primary-foreground py-2 rounded-md text-sm font-medium hover:bg-primary/90 transition-colors"
                        >
                            <Play className="w-4 h-4" /> Start Benchmark
                        </button>
                    </div>
                </div>
            ));
            setIsTyping(false);
            return;
        }

        // Generic Chat Response
        setTimeout(() => {
            addMessage("agent", "I'm listening. You can upload PDFs or type 'Run' to start.");
            setIsTyping(false);
        }, 600);
    };

    return (
        <div className="flex flex-col h-full">
            {/* Chat History */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6">
                {messages.map((m) => (
                    <div key={m.id} className={cn("flex w-full animate-in slide-in-from-bottom-2 duration-500", m.role === "user" ? "justify-end" : "justify-start")}>
                        <div className={cn("flex flex-col max-w-[85%] md:max-w-[70%]", m.role === "user" ? "items-end" : "items-start")}>

                            {/* Avatar & Bubble Container */}
                            <div className={cn("flex gap-3", m.role === "user" ? "flex-row-reverse" : "flex-row")}>
                                <div className={cn(
                                    "w-8 h-8 rounded-full flex items-center justify-center shrink-0 border shadow-sm",
                                    m.role === "agent" ? "bg-primary/10 border-primary/20 text-primary" :
                                        m.role === "user" ? "bg-secondary border-muted-foreground/20 text-secondary-foreground" : "bg-destructive/10 text-destructive"
                                )}>
                                    {m.role === "agent" && <Cpu className="w-4 h-4" />}
                                    {m.role === "user" && <User className="w-4 h-4" />}
                                    {m.role === "system" && <AlertCircle className="w-4 h-4" />}
                                </div>

                                <div className={cn(
                                    "px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm",
                                    m.role === "agent" ? "bg-card border rounded-tl-none" :
                                        m.role === "user" ? "bg-primary text-primary-foreground rounded-tr-none" : "bg-destructive/10 border-destructive/20 rounded-tl-none"
                                )}>
                                    <div dangerouslySetInnerHTML={{ __html: typeof m.content === 'string' ? m.content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') : "" }} />
                                    {typeof m.content !== 'string' && m.content}
                                </div>
                            </div>

                            {/* Interactive Actions rendered below the message */}
                            {m.actions && (
                                <div className="mt-2 ml-11 w-full">
                                    {m.actions}
                                </div>
                            )}
                        </div>
                    </div>
                ))}

                {isTyping && (
                    <div className="flex gap-3 items-center ml-0 animate-pulse">
                        <div className="w-8 h-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center">
                            <Cpu className="w-4 h-4 text-primary" />
                        </div>
                        <div className="text-xs text-muted-foreground font-medium">Thinking...</div>
                    </div>
                )}
            </div>

            {/* Input Zone */}
            <div className="p-4 md:p-6 bg-background/50 border-t backdrop-blur">
                <div className="max-w-4xl mx-auto flex flex-col gap-3">

                    {/* Active Challengers List (Mini View) */}
                    {challengers.length > 0 && (
                        <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
                            {challengers.map(c => (
                                <div key={c.id} className="flex items-center gap-2 bg-secondary/80 text-secondary-foreground px-3 py-1.5 rounded-full text-xs font-medium border text-nowrap">
                                    <FileText className="w-3 h-3" />
                                    {c.name}
                                    {c.status === "pending" && <div className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-pulse" />}
                                    {c.status === "ready" && <div className="w-1.5 h-1.5 rounded-full bg-green-500" />}
                                </div>
                            ))}
                        </div>
                    )}

                    <div className="relative flex items-center gap-2 bg-muted/40 p-2 rounded-xl border focus-within:ring-2 focus-within:ring-primary/20 transition-all shadow-sm">
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            className="p-3 text-muted-foreground hover:bg-background rounded-lg transition-colors"
                        >
                            <Paperclip className="w-5 h-5" />
                        </button>
                        <input
                            ref={fileInputRef}
                            type="file"
                            className="hidden"
                            accept={session.status === "idle" ? ".py" : ".pdf"}
                            onChange={(e) => {
                                const file = e.target.files?.[0];
                                if (!file) return;
                                if (session.status === "idle" && file.name.endsWith(".py")) {
                                    handleAlgoUpload(file);
                                } else if (session.status !== "idle" && file.name.endsWith(".pdf")) {
                                    handleChallengerUpload(file);
                                } else {
                                    addMessage("system", session.status === "idle" ? "Please upload a .py file first." : "Please upload .pdf files for challengers.");
                                }
                            }}
                        />

                        <input
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && handleSend()}
                            placeholder={session.status === "idle" ? "Upload your algorithm to begin..." : "Type 'Run' to start, or upload more PDFs..."}
                            className="flex-1 bg-transparent border-none focus:outline-none text-sm px-2 h-10"
                        />

                        <button
                            onClick={handleSend}
                            disabled={!inputValue.trim()}
                            className={cn(
                                "p-3 rounded-lg transition-all",
                                inputValue.trim() ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground opacity-50"
                            )}
                        >
                            <ArrowRight className="w-5 h-5" />
                        </button>
                    </div>
                    <div className="text-center text-[10px] text-muted-foreground/60">
                        Benchwarmer.AI may produce inaccurate benchmarks. Verify results independently.
                    </div>
                </div>
            </div>
        </div>
    );
}

// ─── Configuration View Component ────────────────────────────────────────────
function ConfigurationView({ session, setSession, onStart }: { session: SessionState, setSession: React.Dispatch<React.SetStateAction<SessionState>>, onStart: () => void }) {
    return (
        <div className="mt-3 bg-card border rounded-lg p-4 w-full max-w-sm animate-in fade-in zoom-in-95">
            <h3 className="font-semibold text-sm mb-3">Execution Configuration</h3>
            <div className="space-y-3">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Cloud className="w-4 h-4" /> Cloud Execution (Modal)
                    </div>
                    {/* Toggle Switch */}
                    <button
                        onClick={() => setSession(s => ({ ...s, executionMode: s.executionMode === "local" ? "modal" : "local" }))}
                        className={cn(
                            "relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2",
                            session.executionMode === "modal" ? "bg-primary" : "bg-input"
                        )}
                    >
                        <span className={cn(
                            "inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform",
                            session.executionMode === "modal" ? "translate-x-5" : "translate-x-1"
                        )} />
                    </button>
                </div>
                <button
                    onClick={onStart}
                    className="w-full flex items-center justify-center gap-2 bg-primary text-primary-foreground py-2 rounded-md text-sm font-medium hover:bg-primary/90 transition-colors"
                >
                    <Play className="w-4 h-4" /> Start Benchmark
                </button>
            </div>
        </div>
    );
}


// ─── Live Dashboard ──────────────────────────────────────────────────────────

function LiveDashboard({ session, challengers, onBack }: { session: SessionState, challengers: Challenger[], onBack: () => void }) {
    // Map challenger_id -> logs
    const [challengerLogs, setChallengerLogs] = useState<Record<string, string[]>>({});
    const [systemLogs, setSystemLogs] = useState<string[]>([]);

    // Results
    const [results, setResults] = useState<any>(null);
    const [status, setStatus] = useState("initializing");

    // WebSocket
    useEffect(() => {
        if (!session.id) return;

        // Initiate Run
        fetch(`/api/python/session/${session.id}/run`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ execution_mode: session.executionMode })
        }).catch(err => console.error("Run failed init", err));

        // Connect WS
        const ws = new WebSocket(`ws://localhost:8000/api/session/${session.id}/live`);

        ws.onopen = () => setStatus("connected");
        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);

            if (msg.type === "log") {
                const text = `[${msg.source}] ${msg.message}`;

                if (msg.challenger_id) {
                    setChallengerLogs(prev => ({
                        ...prev,
                        [msg.challenger_id]: [...(prev[msg.challenger_id] || []), text]
                    }));
                } else {
                    setSystemLogs(prev => [...prev, text]);
                }
            } else if (msg.type === "status") {
                setStatus(msg.status);
            } else if (msg.type === "challenger_update") {
                // Also log updates to specific challenger
                const text = `[Status] ${msg.status}: ${msg.message}`;
                setChallengerLogs(prev => ({
                    ...prev,
                    [msg.challenger_id]: [...(prev[msg.challenger_id] || []), text]
                }));
            } else if (msg.type === "result") {
                setResults(msg.data);
                setStatus("completed");
            }
        };

        return () => ws.close();
    }, [session.id]);

    const scrollRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [systemLogs]);

    // ─── View 1: Results (Completed) ───
    if (status === "completed" && results) {
        return (
            <div className="flex h-full flex-col p-6 animate-in fade-in">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-2xl font-bold flex items-center gap-2">
                        <BarChart3 className="w-6 h-6 text-primary" /> Benchmark Results
                    </h2>
                    <button
                        onClick={onBack}
                        className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-2 px-3 py-1.5 border rounded-md"
                    >
                        <ArrowRight className="w-4 h-4 rotate-180" /> Back to Setup
                    </button>
                </div>

                <div className="flex-1 overflow-hidden flex flex-col gap-6">
                    <div className="flex-1 bg-card border rounded-xl shadow-sm p-4 relative">
                        <ResultsChart
                            title={results.title}
                            xLabel={results.xLabel}
                            yLabel={results.yLabel}
                            data={results.data}
                            series={results.series}
                        />
                    </div>
                    {/* Collapsible System Logs could go here */}
                </div>
            </div>
        );
    }

    // ─── View 2: Grid Execution (Running) ───
    return (
        <div className="flex h-full flex-col p-6 animate-in mb-8 overflow-hidden">
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-4">
                    <h2 className="text-xl font-bold flex items-center gap-2">
                        <Terminal className="w-5 h-5 text-primary" /> Live Benchmark
                    </h2>
                    <span className={cn(
                        "text-xs px-2.5 py-1 rounded-full font-medium border uppercase tracking-wider animate-pulse",
                        status === "completed" ? "bg-green-100 text-green-700 border-green-200" : "bg-blue-50 text-blue-700 border-blue-200"
                    )}>
                        {status}
                    </span>
                </div>
                {/* Duplicate Cloud Badge Removed */}
            </div>

            {/* Grid */}
            <div className="flex-1 overflow-y-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pb-20">
                {/* 1. User Algo Card */}
                <div className="bg-card border rounded-xl shadow-sm overflow-hidden flex flex-col h-80">
                    <div className="px-4 py-3 border-b bg-muted/30 flex items-center justify-between">
                        <div className="flex items-center gap-2 font-medium">
                            <User className="w-4 h-4 text-primary" />
                            <span className="truncate max-w-[150px]">{session.userAlgo}</span>
                        </div>
                        <div className="text-xs text-muted-foreground">User Algorithm</div>
                    </div>
                    <div className="flex-1 bg-black/95 p-3 text-xs font-mono text-green-400 overflow-y-auto">
                        {/* Filter system logs for user algo related stuff or just show system logs here? */}
                        {/* For now, let's show General System Logs here as the 'Main' terminal */}
                        {systemLogs.map((log, i) => (
                            <div key={i} className="mb-1 break-all border-b border-white/5 pb-0.5 opacity-80">{log}</div>
                        ))}
                        <div className="animate-pulse">_</div>
                    </div>
                </div>

                {/* 2. Challenger Cards */}
                {challengers.map(c => (
                    <div key={c.id} className="bg-card border rounded-xl shadow-sm overflow-hidden flex flex-col h-80">
                        <div className="px-4 py-3 border-b bg-muted/30 flex items-center justify-between">
                            <div className="flex items-center gap-2 font-medium">
                                <FileText className="w-4 h-4 text-orange-500" />
                                <span className="truncate max-w-[150px]">{c.name}</span>
                            </div>
                            <div className={cn(
                                "text-xs px-2 py-0.5 rounded-full border capitalize",
                                c.status === "ready" ? "bg-green-100 text-green-700 border-green-200" :
                                    c.status === "error" ? "bg-red-100 text-red-700 border-red-200" : "bg-yellow-50 text-yellow-700 border-yellow-200"
                            )}>
                                {c.status}
                            </div>
                        </div>
                        <div className="flex-1 bg-black/95 p-3 text-xs font-mono text-blue-300 overflow-y-auto relative">
                            {(challengerLogs[c.id] || []).map((log, i) => (
                                <div key={i} className="mb-1 break-all border-b border-white/5 pb-0.5 opacity-80">{log}</div>
                            ))}
                            {c.status === "analyzing" && (
                                <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-[1px]">
                                    <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
