"use client";

import { useState, useRef, useEffect, ReactNode } from "react";
import { Upload, FileText, Play, Loader2, ArrowRight, User, Cpu, Terminal, X, Paperclip, Send } from "lucide-react";
import { cn } from "@/lib/utils";
import { ResultsChart } from "./ResultsChart";

// ─── Types ───────────────────────────────────────────────────────────────────

interface SessionState {
    id: string | null;
    problemClass: string | null;
    userAlgo: string | null;
    config: any | null;
}

interface Challenger {
    id: string;
    type: "pdf" | "baseline";
    name: string;
    status: "pending" | "analyzing" | "ready" | "error";
    message?: string;
}

interface Message {
    id: string;
    role: "system" | "user" | "agent";
    content: ReactNode | string;
    timestamp: number;
}

// ─── Main Wizard Component ───────────────────────────────────────────────────

export function Wizard() {
    const [view, setView] = useState<"initial" | "chat" | "running">("initial");
    const [session, setSession] = useState<SessionState>({
        id: null,
        problemClass: null,
        userAlgo: null,
        config: null
    });
    const [challengers, setChallengers] = useState<Challenger[]>([]);

    // Chat state
    const [messages, setMessages] = useState<Message[]>([]);
    const [isTyping, setIsTyping] = useState(false);

    // ─── Actions ───

    const addMessage = (role: Message["role"], content: ReactNode | string) => {
        setMessages(prev => [...prev, {
            id: Math.random().toString(36).substring(7),
            role,
            content,
            timestamp: Date.now()
        }]);
    };

    const handleInitialUpload = async (file: File) => {
        setIsTyping(true);
        // Transition to chat immediately for visuals
        setView("chat");

        // Add fake user message for the upload
        addMessage("user", `Uploaded algorithm: ${file.name}`);
        addMessage("system", "Analyzing your code...");

        const formData = new FormData();
        formData.append("file", file);

        try {
            // Use proxy path
            const res = await fetch("/api/python/session/start", {
                method: "POST",
                body: formData
            });

            if (!res.ok) throw new Error("Upload failed");

            const data = await res.json();

            // Artificial delay for "Agent Thinking" effect
            setTimeout(() => {
                setSession({
                    id: data.session_id,
                    problemClass: data.detected_class,
                    userAlgo: data.filename,
                    config: null
                });

                setIsTyping(false);
                addMessage("agent", `I've analyzed **${data.filename}**. It looks like a solver for **${data.detected_class}**.`);

                setTimeout(() => {
                    addMessage("agent", "You can now add challengers by uploading research papers (PDF), or simply describe how you want to configure the benchmark environment.");
                }, 800);
            }, 1000);

        } catch (err) {
            console.error(err);
            setIsTyping(false);
            addMessage("system", "Error uploading file. Please reload and try again.");
        }
    };

    return (
        <div className="w-full h-full flex flex-col bg-card overflow-hidden relative transition-all duration-500">

            {/* Header / Status Bar */}
            <div className="h-14 border-b px-6 flex items-center justify-between bg-muted/20">
                <div className="flex items-center gap-3">
                    <span className="text-lg font-bold tracking-tight">Benchwarmer<span className="text-primary">.AI</span></span>
                    {session.problemClass && (
                        <>
                            <div className="w-1 h-1 rounded-full bg-muted-foreground/30" />
                            <span className="text-sm text-muted-foreground">
                                {session.problemClass}
                            </span>
                        </>
                    )}
                </div>
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    {challengers.length > 0 && (
                        <span>{challengers.length} Challenger{challengers.length !== 1 ? 's' : ''} Active</span>
                    )}
                </div>
            </div>

            {/* Views */}
            <div className="flex-1 relative overflow-hidden">
                {view === "initial" && (
                    <InitialUploadView onUpload={handleInitialUpload} />
                )}

                {view === "chat" && (
                    <ChatView
                        messages={messages}
                        isTyping={isTyping}
                        session={session}
                        challengers={challengers}
                        setChallengers={setChallengers}
                        onAddMessage={addMessage}
                        setTyping={setIsTyping}
                        onRun={() => setView("running")}
                        setSession={setSession}
                    />
                )}

                {view === "running" && session.id && (
                    <LiveExecutionView
                        sessionId={session.id}
                        userAlgo={session.userAlgo!}
                        challengers={challengers}
                    />
                )}
            </div>
        </div>
    );
}

// ─── View 1: Initial Centered Upload ─────────────────────────────────────────

function InitialUploadView({ onUpload }: { onUpload: (f: File) => void }) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [dragging, setDragging] = useState(false);

    const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) onUpload(file);
    };

    return (
        <div className="absolute inset-0 flex flex-col items-center justify-center p-8 transition-opacity duration-300 animate-in fade-in zoom-in-95">
            <div className="text-center space-y-3 mb-6">
                <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center mx-auto mb-3">
                    <User className="w-6 h-6 text-primary" />
                </div>
                <h1 className="text-2xl font-bold tracking-tight">Upload Your Algorithm</h1>
                <p className="text-sm text-muted-foreground max-w-md mx-auto">
                    Drop a Python file to benchmark against research papers
                </p>
            </div>

            <div
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                    e.preventDefault();
                    setDragging(false);
                    const file = e.dataTransfer.files?.[0];
                    if (file && file.name.endsWith('.py')) onUpload(file);
                }}
                className={cn(
                    "w-full max-w-lg h-40 border-2 border-dashed rounded-xl flex flex-col items-center justify-center gap-2 cursor-pointer transition-all",
                    dragging ? "border-primary bg-primary/5 scale-[1.02]" : "border-muted-foreground/20 hover:border-primary/50 hover:bg-muted/30"
                )}
            >
                <Upload className="w-7 h-7 text-muted-foreground" />
                <p className="font-medium text-sm">Drop your .py file here</p>
            </div>
            <input ref={inputRef} type="file" accept=".py" className="hidden" onChange={handleFile} />
        </div>
    );
}

// ─── View 2: Chat Interface ──────────────────────────────────────────────────

function ChatView({
    messages, isTyping, session, challengers, setChallengers, onAddMessage, setTyping, onRun, setSession
}: {
    messages: Message[],
    isTyping: boolean,
    session: SessionState,
    challengers: Challenger[],
    setChallengers: React.Dispatch<React.SetStateAction<Challenger[]>>,
    onAddMessage: (role: Message["role"], content: ReactNode | string) => void,
    setTyping: (t: boolean) => void,
    onRun: () => void,
    setSession: React.Dispatch<React.SetStateAction<SessionState>>
}) {
    const [inputValue, setInputValue] = useState("");
    const inputRef = useRef<HTMLInputElement>(null);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isTyping]);

    const handleSend = async (file?: File) => {
        if (!inputValue.trim() && !file) return;

        const text = inputValue;
        setInputValue("");

        // 1. Handle File Upload (Assume Challenger PDF)
        if (file) {
            onAddMessage("user", <span className="flex items-center gap-2"><FileText className="w-4 h-4" /> Uploaded {file.name}</span>);
            setTyping(true);

            const formData = new FormData();
            formData.append("type", "pdf");
            formData.append("file", file);

            try {
                const res = await fetch(`/api/python/session/${session.id}/challenger`, {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();

                setChallengers(prev => [...prev, {
                    id: data.challenger_id,
                    type: "pdf",
                    name: file.name,
                    status: data.status || "pending"
                }]);

                onAddMessage("agent", `Received **${file.name}**. I'll implement this during the benchmark run.`);
            } catch (e) {
                onAddMessage("system", "Failed to upload challenger.");
            } finally {
                setTyping(false);
            }
            return;
        }

        // 2. Handle Text (Assume Config/Chat)
        onAddMessage("user", text);
        setTyping(true);

        // Simple Keyword logic for v0
        const lower = text.toLowerCase();

        if (lower.includes("run") || lower.includes("start") || lower.includes("go")) {
            // Trigger configuration -> run
            onAddMessage("agent", "Configuring the environment based on your preferences...");

            try {
                const formData = new FormData();
                formData.append("preferences", text); // Pass the whole text as prefs

                const res = await fetch(`/api/python/session/${session.id}/configure`, {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                setSession(s => ({ ...s, config: data.config }));

                onAddMessage("agent", "Configuration ready. Starting the benchmark race now!");
                setTimeout(onRun, 1500);
            } catch (e) {
                onAddMessage("system", "Failed to configure benchmark.");
            } finally {
                setTyping(false);
            }
        } else {
            // General chat / gathering prefs
            setTimeout(() => {
                onAddMessage("agent", "Got it. Anything else? You can upload more PDFs or type 'Run' to start.");
                setTyping(false);
            }, 600);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="flex flex-col h-full bg-muted/5">
            {/* Messages */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-6">
                {messages.map((m) => (
                    <div key={m.id} className={cn("flex w-full", m.role === "user" ? "justify-end" : "justify-start")}>
                        <div className={cn(
                            "max-w-[80%]",
                            m.role === "user" ? "bg-primary text-primary-foreground rounded-2xl rounded-tr-sm px-4 py-3" : "flex gap-3"
                        )}>
                            {m.role !== "user" && (
                                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 border border-primary/20">
                                    <Cpu className="w-4 h-4 text-primary" />
                                </div>
                            )}
                            <div className={cn(
                                "text-sm leading-relaxed",
                                m.role !== "user" && "bg-muted/50 border rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm text-foreground"
                            )}>
                                {typeof m.content === 'string' ? (
                                    <div dangerouslySetInnerHTML={{
                                        __html: m.content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                    }} />
                                ) : m.content}
                            </div>
                        </div>
                    </div>
                ))}

                {isTyping && (
                    <div className="flex w-full justify-start">
                        <div className="flex gap-3 max-w-[80%]">
                            <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 border border-primary/20">
                                <Cpu className="w-4 h-4 text-primary" />
                            </div>
                            <div className="bg-muted/50 border rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm flex items-center gap-2 text-foreground">
                                <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                                <span className="text-xs text-muted-foreground">Thinking...</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div className="p-4 bg-background border-t">

                {/* Active Challengers Chips */}
                {challengers.length > 0 && (
                    <div className="flex gap-2 mb-3 overflow-x-auto pb-2">
                        {challengers.map(c => (
                            <div key={c.id} className="flex items-center gap-2 bg-secondary text-secondary-foreground px-3 py-1.5 rounded-full text-xs font-medium border animate-in slide-in-from-bottom-2">
                                <FileText className="w-3 h-3" />
                                <span className="max-w-[100px] truncate">{c.name}</span>
                                {c.status === "pending" && <div className="w-2 h-2 rounded-full bg-yellow-500 ml-1" title="Queued for implementation" />}
                                {c.status === "analyzing" && <Loader2 className="w-3 h-3 animate-spin ml-1" />}
                                {c.status === "ready" && <div className="w-2 h-2 rounded-full bg-green-500 ml-1" />}
                            </div>
                        ))}
                    </div>
                )}

                <div className="relative flex items-end gap-2 bg-muted/30 p-2 rounded-xl border focus-within:ring-2 focus-within:ring-primary/20 transition-all">
                    <button
                        onClick={() => inputRef.current?.click()}
                        className="p-2.5 text-muted-foreground hover:bg-muted rounded-lg transition-colors"
                        title="Upload PDF Paper"
                    >
                        <Paperclip className="w-5 h-5" />
                    </button>
                    <input ref={inputRef} type="file" accept=".pdf" className="hidden" onChange={(e) => {
                        if (e.target.files?.[0]) handleSend(e.target.files[0]);
                    }} />

                    <textarea
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={handleKeyPress}
                        placeholder={challengers.length === 0 ? "Ask to add a challenger or configure..." : "Type 'Run' to start or add more papers..."}
                        className="flex-1 bg-transparent border-none focus:outline-none resize-none max-h-32 py-2.5 text-sm"
                        style={{ height: '44px' }}
                    />

                    <button
                        onClick={() => handleSend()}
                        disabled={!inputValue.trim() || isTyping}
                        className={cn(
                            "p-2.5 rounded-lg transition-all mb-0.5",
                            inputValue.trim() ? "bg-primary text-primary-foreground shadow-sm hover:opacity-90" : "text-muted-foreground bg-transparent"
                        )}
                    >
                        <ArrowRight className="w-5 h-5" />
                    </button>
                </div>
            </div>
        </div>
    );
}

// ─── View 3: Live Execution ──────────────────────────────────────────────────

function LiveExecutionView({ sessionId }: { sessionId: string, userAlgo: string, challengers: Challenger[] }) {
    const [status, setStatus] = useState("connecting");
    const [logs, setLogs] = useState<string[]>([]);
    const [results, setResults] = useState<any>(null);
    const logRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        let ws: WebSocket | null = null;

        const initializeRun = async () => {
            try {
                // Start the run and wait for it to begin
                await fetch(`/api/python/session/${sessionId}/run`, { method: "POST" });

                // Small delay to ensure backend is ready for WebSocket
                await new Promise(resolve => setTimeout(resolve, 500));

                // Now connect WebSocket
                ws = new WebSocket(`ws://localhost:8000/api/session/${sessionId}/live`);

                ws.onopen = () => {
                    console.log("WebSocket connected successfully");
                    setStatus("running");
                    setLogs(prev => [...prev, "System: Connected to live environment..."]);
                };

                ws.onmessage = (event) => {
                    const msg = JSON.parse(event.data);
                    if (msg.type === "log") {
                        setLogs(prev => [...prev, `${msg.source}: ${msg.message}`]);
                    } else if (msg.type === "status") {
                        setStatus(msg.status);
                        if (msg.message) {
                            setLogs(prev => [...prev, `System: ${msg.message}`]);
                        }
                    } else if (msg.type === "result") {
                        setResults(msg.data);
                    } else if (msg.type === "challenger_update") {
                        setLogs(prev => [...prev, `Agent: ${msg.message}`]);
                    }
                };

                ws.onerror = (error) => {
                    console.error("WebSocket error:", error);
                    setLogs(prev => [...prev, "System: WebSocket connection error"]);
                    setStatus("error");
                };

                ws.onclose = (event) => {
                    console.log("WebSocket closed:", event.code, event.reason);
                    if (event.code !== 1000) { // 1000 = normal closure
                        setLogs(prev => [...prev, `System: Connection closed (${event.reason || "Unknown reason"})`]);
                        setStatus("error");
                    }
                };
            } catch (error) {
                console.error("Failed to start run:", error);
                setLogs(prev => [...prev, "System: Failed to start benchmark"]);
                setStatus("error");
            }
        };

        initializeRun();

        return () => {
            if (ws) ws.close();
        };
    }, [sessionId]);

    // Auto-scroll logs
    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="h-full flex flex-col p-6 animate-in fade-in">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold flex items-center gap-2">
                    <Terminal className="w-5 h-5" /> Live Benchmark
                </h2>
                <span className={cn(
                    "text-xs px-2 py-1 rounded-full font-medium uppercase",
                    status === "running" ? "bg-amber-100 text-amber-700" : "bg-green-100 text-green-700"
                )}>
                    {status}
                </span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-0">
                {/* Live Terminal */}
                <div ref={logRef} className="bg-black/95 rounded-lg p-4 font-mono text-xs text-green-400 overflow-y-auto border border-gray-800 shadow-inner flex flex-col gap-1">
                    {logs.map((log, i) => (
                        <div key={i} className="whitespace-pre-wrap break-all border-b border-white/5 pb-0.5">{log}</div>
                    ))}
                    {status === "running" && (
                        <div className="animate-pulse">_</div>
                    )}
                </div>

                {/* Results Chart */}
                <div className="bg-muted/10 rounded-lg p-4 border flex items-center justify-center relative overflow-hidden">
                    {results ? (
                        <ResultsChart {...results} />
                    ) : (
                        <div className="text-center text-muted-foreground z-10">
                            <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2 opacity-50" />
                            <p className="text-sm">Waiting for results...</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}


