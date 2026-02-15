import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import {
    MessageSquare,
    PanelLeftClose,
    PanelLeft,
    FileCode,
    Clock,
    Trash2,
    Pencil,
} from "lucide-react"

interface ChatSession {
    id: string
    created_at: string
    title: string
}

interface SidebarProps {
    className?: string
    hasChat?: boolean
    currentSessionId?: string | null
    codedAlgorithms?: string[]
    onAlgorithmClick?: (name: string) => void
    activeAlgorithm?: string | null
    onSessionClick?: (sessionId: string) => void
    onNewChat?: () => void
}

export function Sidebar({
    className,
    hasChat = false,
    currentSessionId,
    codedAlgorithms = [],
    onAlgorithmClick,
    activeAlgorithm,
    onSessionClick,
    onNewChat,
}: SidebarProps) {
    const [isCollapsed, setIsCollapsed] = useState(false)
    const [sessions, setSessions] = useState<ChatSession[]>([])
    const [editingId, setEditingId] = useState<string | null>(null)
    const [editingTitle, setEditingTitle] = useState("")

    // Fetch chat history
    useEffect(() => {
        const fetchSessions = async () => {
            try {
                const res = await fetch("/api/sessions")
                if (res.ok) {
                    const data = await res.json()
                    setSessions(data)
                }
            } catch (error) {
                console.error("Failed to fetch sessions", error)
            }
        }
        fetchSessions()
        const interval = setInterval(fetchSessions, 10000)
        return () => clearInterval(interval)
    }, [])

    // History sessions = all sessions except the current one
    const historySessions = sessions.filter((s) => s.id !== currentSessionId)

    const handleDeleteSession = async (sessionId: string) => {
        try {
            const res = await fetch(`/api/sessions/${sessionId}`, { method: "DELETE" })
            if (res.ok) {
                setSessions((prev) => prev.filter((s) => s.id !== sessionId))
            }
        } catch (error) {
            console.error("Failed to delete session", error)
        }
    }

    const startRename = (session: ChatSession) => {
        setEditingId(session.id)
        setEditingTitle(session.title)
    }

    const handleRename = async (sessionId: string) => {
        const trimmed = editingTitle.trim()
        if (!trimmed) {
            setEditingId(null)
            return
        }
        try {
            const res = await fetch(`/api/sessions/${sessionId}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title: trimmed }),
            })
            if (res.ok) {
                setSessions((prev) =>
                    prev.map((s) => (s.id === sessionId ? { ...s, title: trimmed } : s))
                )
            }
        } catch (error) {
            console.error("Failed to rename session", error)
        } finally {
            setEditingId(null)
        }
    }

    return (
        <div
            className={cn(
                "relative flex flex-col h-screen border-r border-sidebar-border transition-all duration-200 ease-in-out",
                isCollapsed ? "w-[52px]" : "w-[260px]",
                className
            )}
            style={{ backgroundColor: "hsl(var(--sidebar))" }}
        >
            {/* Header */}
            <div className={cn("flex items-center h-14 px-3 gap-2 flex-shrink-0", isCollapsed ? "justify-center" : "justify-between")}>
                {!isCollapsed && (
                    <div className="flex items-center gap-2 pl-1">
                        <img src="/bench-logo.png" alt="Benchwarmer.AI" className="h-6 w-6 rounded-md object-cover" />
                        <span className="font-semibold text-sm text-sidebar-foreground tracking-tight">
                            benchwarmer
                        </span>
                    </div>
                )}
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-foreground hover:bg-muted rounded-lg flex-shrink-0"
                    onClick={() => setIsCollapsed(!isCollapsed)}
                >
                    {isCollapsed ? (
                        <PanelLeft className="h-4 w-4" />
                    ) : (
                        <PanelLeftClose className="h-4 w-4" />
                    )}
                </Button>
            </div>

            {/* Current Chat */}
            <nav className="flex flex-col gap-0.5 px-2 py-1">
                <Button
                    variant="ghost"
                    className={cn(
                        "w-full justify-start gap-2.5 h-9 text-sm font-normal rounded-lg",
                        isCollapsed && "justify-center px-0",
                        "bg-muted text-foreground"
                    )}
                    onClick={onNewChat}
                >
                    <MessageSquare className="h-4 w-4 flex-shrink-0" />
                    {!isCollapsed && <span>Current Chat</span>}
                </Button>
            </nav>

            {/* Coded Algorithms - only show during active chat with coded algos */}
            {hasChat && codedAlgorithms.length > 0 && (
                <>
                    <div className="mx-3 my-1.5 border-t border-sidebar-border" />
                    <div className="flex flex-col overflow-hidden">
                        {!isCollapsed && (
                            <div className="px-4 pb-1.5 text-[11px] font-medium text-muted-foreground/60 tracking-wider uppercase">
                                Implemented Algorithms
                            </div>
                        )}
                        <div className="px-2 space-y-0.5">
                            {codedAlgorithms.map((name) => (
                                <Button
                                    key={name}
                                    variant="ghost"
                                    className={cn(
                                        "w-full justify-start gap-2.5 h-8 text-xs font-normal truncate rounded-lg",
                                        isCollapsed && "justify-center px-0",
                                        activeAlgorithm === name
                                            ? "bg-violet-500/10 text-violet-300 border border-violet-500/20"
                                            : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                                    )}
                                    onClick={() => onAlgorithmClick?.(name)}
                                    title={`${name}.py — click to view code`}
                                >
                                    <FileCode className="h-3.5 w-3.5 shrink-0 text-violet-400" />
                                    {!isCollapsed && (
                                        <span className="truncate">{name}.py</span>
                                    )}
                                </Button>
                            ))}
                        </div>
                    </div>
                </>
            )}

            {/* History */}
            {historySessions.length > 0 && (
                <>
                    <div className="mx-3 my-2 border-t border-sidebar-border" />
                    <div className="flex-1 flex flex-col overflow-hidden min-h-0">
                        {!isCollapsed && (
                            <div className="px-4 pb-1.5 text-[11px] font-medium text-muted-foreground/60 tracking-wider uppercase flex items-center gap-1.5">
                                <Clock className="h-3 w-3" />
                                History
                            </div>
                        )}
                        <ScrollArea className="flex-1">
                            <div className="px-2 space-y-0.5">
                                {historySessions.map((session) => (
                                    <div
                                        key={session.id}
                                        className="group relative flex items-center"
                                    >
                                        {editingId === session.id ? (
                                            <div className="flex items-center w-full px-2 h-8">
                                                <MessageSquare className="h-3.5 w-3.5 shrink-0 text-muted-foreground/60 mr-2" />
                                                <input
                                                    autoFocus
                                                    className="flex-1 min-w-0 bg-transparent text-xs text-foreground outline-none border-b border-violet-500/50 py-0.5"
                                                    value={editingTitle}
                                                    onChange={(e) => setEditingTitle(e.target.value)}
                                                    onKeyDown={(e) => {
                                                        if (e.key === "Enter") handleRename(session.id)
                                                        if (e.key === "Escape") setEditingId(null)
                                                    }}
                                                    onBlur={() => handleRename(session.id)}
                                                />
                                            </div>
                                        ) : (
                                            <>
                                                <Button
                                                    variant="ghost"
                                                    className={cn(
                                                        "w-full justify-start gap-2.5 h-8 text-xs font-normal truncate rounded-lg pr-14",
                                                        isCollapsed && "justify-center px-0",
                                                        "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                                                    )}
                                                    onClick={() => onSessionClick?.(session.id)}
                                                    title={session.title}
                                                >
                                                    <MessageSquare className="h-3.5 w-3.5 shrink-0 text-muted-foreground/60" />
                                                    {!isCollapsed && (
                                                        <span className="truncate">{session.title}</span>
                                                    )}
                                                </Button>
                                                {!isCollapsed && (
                                                    <div className="absolute right-1 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                                                        <button
                                                            className="p-1 rounded hover:bg-muted text-muted-foreground/40 hover:text-foreground"
                                                            onClick={(e) => {
                                                                e.stopPropagation()
                                                                startRename(session)
                                                            }}
                                                            title="Rename chat"
                                                        >
                                                            <Pencil className="h-3 w-3" />
                                                        </button>
                                                        <button
                                                            className="p-1 rounded hover:bg-destructive/20 hover:text-destructive text-muted-foreground/40"
                                                            onClick={(e) => {
                                                                e.stopPropagation()
                                                                handleDeleteSession(session.id)
                                                            }}
                                                            title="Delete chat"
                                                        >
                                                            <Trash2 className="h-3 w-3" />
                                                        </button>
                                                    </div>
                                                )}
                                            </>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </ScrollArea>
                    </div>
                </>
            )}
        </div>
    )
}
