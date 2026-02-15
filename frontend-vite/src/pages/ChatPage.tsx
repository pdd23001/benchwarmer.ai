import { useState, useEffect } from "react"
import { Layout } from "@/components/Layout"
import { MessageList } from "@/components/chat/MessageList"
import { ChatInput } from "@/components/chat/ChatInput"
import { UploadZone } from "@/components/chat/UploadZone"
import { CodeViewer } from "@/components/chat/CodeViewer"
import { SandboxPanel } from "@/components/chat/SandboxPanel"
import { useChat } from "@/hooks/use-chat"
import { Button } from "@/components/ui/button"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
import { Cpu, Cloud, PlusCircle, FileText, FileCode } from "lucide-react"
import { BenchwarmerLogo } from "@/components/BenchwarmerLogo"

type Phase = "welcome" | "transitioning" | "chat"

export default function ChatPage() {
    const { messages, isLoading, sendMessage, clearChat, loadSession, codedAlgorithms, sessionId, benchmarkProgress, setBenchmarkProgress } = useChat()
    const [executionMode, setExecutionMode] = useState("local")
    const [llmBackend, setLlmBackend] = useState("claude")
    const [viewingAlgo, setViewingAlgo] = useState<string | null>(null)

    // Lifted file state shared between UploadZones and ChatInput
    const [pdfs, setPdfs] = useState<File[]>([])
    const [algo, setAlgo] = useState<File | undefined>(undefined)

    // Phase management
    const [phase, setPhase] = useState<Phase>(messages.length > 0 ? "chat" : "welcome")

    // Sync phase with messages when messages come in from polling/restore
    useEffect(() => {
        if (messages.length > 0 && phase === "welcome") {
            setPhase("chat")
        }
    }, [messages.length, phase])

    const handleSend = (
        text: string,
        files: { pdfs?: File[]; algorithm?: File }
    ) => {
        // In welcome phase, ChatInput's controlled state already includes upload zone files
        sendMessage(text, files, { executionMode, llmBackend })

        if (phase === "welcome") {
            setPhase("transitioning")
            setTimeout(() => setPhase("chat"), 550)
            setPdfs([])
            setAlgo(undefined)
        }
    }

    const handleAlgorithmSelect = (indices: number[]) => {
        const text = indices.join(" ")
        sendMessage(text, {}, { executionMode, llmBackend }, { hidden: true })
    }

    const handleChoiceSelect = (_choiceId: string, values: string[]) => {
        const text = values.join(", ")
        sendMessage(text, {}, { executionMode, llmBackend }, { hidden: true })
    }

    const handleClearChat = () => {
        clearChat()
        setPdfs([])
        setAlgo(undefined)
        setViewingAlgo(null)
        setPhase("welcome")
    }

    const isEmpty = messages.length === 0
    const isWelcome = phase === "welcome" || phase === "transitioning"

    // Show sandbox panel when benchmark is running or recently completed
    const showSandbox = benchmarkProgress.status === "running" || benchmarkProgress.status === "complete"

    // Auto-dismiss sandbox panel 3 seconds after benchmark completes
    useEffect(() => {
        if (benchmarkProgress.status === "complete") {
            const timer = setTimeout(() => {
                setBenchmarkProgress({ algorithms: [], status: null })
            }, 3000)
            return () => clearTimeout(timer)
        }
    }, [benchmarkProgress.status, setBenchmarkProgress])

    return (
        <Layout
            hasChat={!isEmpty}
            currentSessionId={sessionId}
            codedAlgorithms={codedAlgorithms}
            onAlgorithmClick={(name) => setViewingAlgo(name)}
            activeAlgorithm={viewingAlgo}
            onSessionClick={(sid) => {
                setViewingAlgo(null)
                loadSession(sid)
                setPhase("chat")
            }}
            onNewChat={handleClearChat}
        >
            <div className="flex flex-col h-screen relative overflow-hidden">
                {/* ── WELCOME / TRANSITIONING STATE ── */}
                {isWelcome && (
                    <div
                        className={`flex flex-col h-full transition-all duration-500 ease-out ${
                            phase === "transitioning" ? "opacity-0 scale-95" : "opacity-100 scale-100"
                        }`}
                    >
                        {/* Logo pinned to top center */}
                        <div className="flex justify-center pt-14 pb-0">
                            <BenchwarmerLogo className="w-[520px] sm:w-[620px]" />
                        </div>

                        {/* Triangle layout filling remaining space */}
                        <div className="flex-1 flex items-center justify-center px-10">
                            <div className="relative w-full max-w-[1000px] h-[480px]">
                                {/* Top-left: Research Papers */}
                                <div className="absolute -left-4 top-0 w-[260px]">
                                    <UploadZone
                                        accept=".pdf"
                                        multiple
                                        icon={<FileText className="h-10 w-10" />}
                                        title="Research Papers"
                                        subtitle="Drop PDFs here"
                                        files={pdfs}
                                        onFilesChange={setPdfs}
                                        className="py-10"
                                    />
                                </div>

                                {/* Bottom-left: Your Algorithm */}
                                <div className="absolute -left-4 bottom-0 w-[260px]">
                                    <UploadZone
                                        accept=".py"
                                        icon={<FileCode className="h-10 w-10" />}
                                        title="Your Algorithm"
                                        subtitle="Drop .py file"
                                        files={algo ? [algo] : []}
                                        onFilesChange={(files) => setAlgo(files[0] || undefined)}
                                        className="py-10"
                                    />
                                </div>

                                {/* Right: Chat input — top and bottom aligned with upload zones */}
                                <div className="absolute right-0 top-0 bottom-0 w-[64%]">
                                    <ChatInput
                                        onSend={handleSend}
                                        disabled={isLoading}
                                        placeholder="Describe your problem..."
                                        hideAttachButtons
                                        className="h-full"
                                        pdfs={pdfs}
                                        onPdfsChange={setPdfs}
                                        algo={algo}
                                        onAlgoChange={setAlgo}
                                        bottomLeftContent={
                                            <div className="flex items-center gap-1.5">
                                                <Select value={llmBackend} onValueChange={setLlmBackend}>
                                                    <SelectTrigger className="w-[100px] h-7 text-[11px] bg-background/50 border-border/50 rounded-lg">
                                                        <SelectValue placeholder="Model" />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="claude">Claude</SelectItem>
                                                        <SelectItem value="nemotron">Nemotron</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                                <Select value={executionMode} onValueChange={setExecutionMode}>
                                                    <SelectTrigger className="w-[130px] h-7 text-[11px] bg-background/50 border-border/50 rounded-lg">
                                                        {executionMode === "local" ? (
                                                            <Cpu className="mr-1 h-3 w-3 text-muted-foreground" />
                                                        ) : (
                                                            <Cloud className="mr-1 h-3 w-3 text-muted-foreground" />
                                                        )}
                                                        <SelectValue placeholder="Mode" />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="local">Sequential</SelectItem>
                                                        <SelectItem value="modal">Modal CPU Sandbox</SelectItem>
                                                        <SelectItem value="ssh_gpu" disabled>SSH GPU Compute (coming soon)</SelectItem>
                                                        <SelectItem value="modal_gpu" disabled>Modal GPU Sandbox (coming soon)</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                        }
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* ── CHAT STATE ── */}
                {phase === "chat" && (
                    <div className="flex flex-col h-full animate-in fade-in slide-in-from-bottom-4 duration-300">
                        {/* Header with controls */}
                        <div className="flex items-center justify-end h-14 px-4 flex-shrink-0 border-b border-border/40">
                            <div className="flex items-center gap-2">
                                <Select value={llmBackend} onValueChange={setLlmBackend} disabled={!isEmpty}>
                                    <SelectTrigger className="w-[120px] h-8 text-xs bg-secondary border-border rounded-lg disabled:opacity-50 disabled:cursor-not-allowed">
                                        <SelectValue placeholder="Model" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="claude">Claude</SelectItem>
                                        <SelectItem value="nemotron">Nemotron</SelectItem>
                                    </SelectContent>
                                </Select>

                                <Select
                                    value={executionMode}
                                    onValueChange={setExecutionMode}
                                    disabled={!isEmpty}
                                >
                                    <SelectTrigger className="w-[160px] h-8 text-xs bg-secondary border-border rounded-lg disabled:opacity-50 disabled:cursor-not-allowed">
                                        {executionMode === "local" ? (
                                            <Cpu className="mr-1.5 h-3.5 w-3.5 text-muted-foreground" />
                                        ) : (
                                            <Cloud className="mr-1.5 h-3.5 w-3.5 text-muted-foreground" />
                                        )}
                                        <SelectValue placeholder="Mode" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="local">Sequential</SelectItem>
                                        <SelectItem value="modal">Modal CPU Sandbox</SelectItem>
                                        <SelectItem value="ssh_gpu" disabled>SSH GPU Compute (coming soon)</SelectItem>
                                        <SelectItem value="modal_gpu" disabled>Modal GPU Sandbox (coming soon)</SelectItem>
                                    </SelectContent>
                                </Select>

                                <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-8 gap-1.5 text-muted-foreground hover:text-foreground rounded-lg"
                                    onClick={handleClearChat}
                                    title="New Chat"
                                >
                                    <PlusCircle className="h-4 w-4" />
                                    <span className="text-xs hidden sm:inline">New Chat</span>
                                </Button>
                            </div>
                        </div>

                        {/* Split view: sandbox panel or code viewer (left) + chat (right) */}
                        <div className="flex-1 min-h-0 flex">
                            {/* Sandbox panel — shown during benchmark execution */}
                            {showSandbox && (
                                <div className="w-[40%] flex-shrink-0">
                                    <SandboxPanel progress={benchmarkProgress} />
                                </div>
                            )}

                            {/* Code viewer panel — hidden when sandbox panel is showing */}
                            {!showSandbox && viewingAlgo && (
                                <div className="w-[45%] flex-shrink-0 border-r border-border">
                                    <CodeViewer
                                        algorithmName={viewingAlgo}
                                        onClose={() => setViewingAlgo(null)}
                                    />
                                </div>
                            )}

                            {/* Chat panel */}
                            <div className="flex-1 min-w-0 flex flex-col">
                                <div className="flex-1 min-h-0 flex flex-col">
                                    <MessageList
                                        messages={messages}
                                        isLoading={isLoading}
                                        onAlgorithmSelect={handleAlgorithmSelect}
                                        onChoiceSelect={handleChoiceSelect}
                                    />
                                </div>
                                <div className="p-4 max-w-3xl w-full mx-auto flex-shrink-0">
                                    <ChatInput onSend={handleSend} disabled={isLoading} />
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </Layout>
    )
}
