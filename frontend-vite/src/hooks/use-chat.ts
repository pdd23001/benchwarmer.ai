import { useState, useRef, useEffect, useCallback } from "react"
import { v4 as uuidv4 } from "uuid"

export type ToolCall = {
    tool: string
    status: "running" | "completed" | "failed"
    result?: any
}

export type AlgorithmChoice = {
    index: number
    name: string
    approach: string
    source: string
}

export type ChoiceOption = {
    value: string
    label: string
    description?: string
}

export type ChoicePrompt = {
    id: string
    title?: string
    options: ChoiceOption[]
    multiSelect: boolean
}

export type Message = {
    id: string
    role: "user" | "assistant"
    content: string
    tools?: ToolCall[]
    plotImage?: string
    algorithmChoices?: AlgorithmChoice[]
    choicePrompt?: ChoicePrompt
    hidden?: boolean
}

export type AlgoProgress = {
    name: string
    completed: number
    total: number
}

export type BenchmarkProgress = {
    algorithms: AlgoProgress[]
    status: "running" | "complete" | null
}

export function useChat() {
    const [messages, setMessages] = useState<Message[]>([])
    const [isLoading, setIsLoading] = useState(false)
    const [sessionId, setSessionId] = useState<string | null>(null)
    const [codedAlgorithms, setCodedAlgorithms] = useState<string[]>([])
    const [benchmarkProgress, setBenchmarkProgress] = useState<BenchmarkProgress>({ algorithms: [], status: null })

    const abortControllerRef = useRef<AbortController | null>(null)
    const pollingRef = useRef(false)

    // API Base URL from environment or empty string (for relative paths/proxy)
    const API_BASE = import.meta.env.VITE_API_BASE_URL || ""

    // ── Poll for completion when backend is still processing ──────────────
    const pollForCompletion = useCallback(
        (sid: string) => {
            if (pollingRef.current) return // already polling
            pollingRef.current = true
            setIsLoading(true)

            const poll = async () => {
                if (!pollingRef.current) return
                try {
                    // Check processing status
                    const statusRes = await fetch(`${API_BASE}/api/chat/${sid}/status`)
                    if (!statusRes.ok) {
                        pollingRef.current = false
                        setIsLoading(false)
                        return
                    }
                    const { processing } = await statusRes.json()

                    // Fetch latest messages from DB
                    const histRes = await fetch(`${API_BASE}/api/chat/${sid}`)
                    if (histRes.ok) {
                        const data = await histRes.json()
                        if (Array.isArray(data) && data.length > 0) {
                            setMessages(data)
                        }
                    }

                    if (processing) {
                        // Keep polling
                        setTimeout(poll, 1500)
                    } else {
                        // Done
                        pollingRef.current = false
                        setIsLoading(false)
                    }
                } catch (error) {
                    console.error("Poll error:", error)
                    pollingRef.current = false
                    setIsLoading(false)
                }
            }

            // Start first poll after a short delay
            setTimeout(poll, 1000)
        },
        []
    )

    // ── On mount: restore session from localStorage if it exists ──────────
    useEffect(() => {
        const savedSessionId = localStorage.getItem("benchwarmer_session_id")
        if (savedSessionId) {
            setSessionId(savedSessionId)
            fetchHistory(savedSessionId)
        }
        return () => {
            pollingRef.current = false
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    // ── Persist session ID to localStorage ───────────────────────────────
    useEffect(() => {
        if (sessionId) {
            localStorage.setItem("benchwarmer_session_id", sessionId)
        } else {
            localStorage.removeItem("benchwarmer_session_id")
        }
    }, [sessionId])

    // ── Fetch history + check if backend is still processing ─────────────
    const fetchHistory = async (sid: string) => {
        try {
            const res = await fetch(`${API_BASE}/api/chat/${sid}`)
            if (!res.ok) {
                if (res.status === 404) {
                    setSessionId(null)
                }
                return
            }
            const data = await res.json()
            if (Array.isArray(data) && data.length > 0) {
                setMessages(data)
            }

            // Check if backend is still processing this session
            const statusRes = await fetch(`${API_BASE}/api/chat/${sid}/status`)
            if (statusRes.ok) {
                const { processing } = await statusRes.json()
                if (processing) {
                    // Backend is still working — poll until it finishes
                    pollForCompletion(sid)
                }
            }
        } catch (error) {
            console.error("Failed to fetch history:", error)
        }
    }

    const clearChat = () => {
        pollingRef.current = false
        setSessionId(null)
        setMessages([])
        setCodedAlgorithms([])
        setBenchmarkProgress({ algorithms: [], status: null })
        setIsLoading(false)
        localStorage.removeItem("benchwarmer_session_id")
    }

    const loadSession = (sid: string) => {
        pollingRef.current = false
        setSessionId(sid)
        setMessages([])
        setCodedAlgorithms([])
        setBenchmarkProgress({ algorithms: [], status: null })
        setIsLoading(false)
        localStorage.setItem("benchwarmer_session_id", sid)
        fetchHistory(sid)
    }

    async function sendMessage(
        text: string,
        attachments: { pdfs?: File[]; algorithm?: File } = {},
        config: { executionMode?: string; llmBackend?: string } = {},
        options: { hidden?: boolean } = {}
    ) {
        // Stop any active polling
        pollingRef.current = false
        setIsLoading(true)

        const userMsgId = uuidv4()
        const newMsg: Message = {
            id: userMsgId,
            role: "user",
            content: text,
            hidden: options.hidden,
        }
        setMessages((prev) => [...prev, newMsg])

        // Prepare request
        const formData = {
            session_id: sessionId,
            message: text,
            execution_mode: config.executionMode || "local",
            llm_backend: config.llmBackend || "claude",
            pdfs: [] as { filename: string; content_base64: string }[],
            custom_algorithm_file: null as {
                filename: string
                content_base64: string
            } | null,
        }

        const readFile = (file: File): Promise<string> => {
            return new Promise((resolve, reject) => {
                const reader = new FileReader()
                reader.onload = () => {
                    if (typeof reader.result === "string") {
                        const base64 = reader.result.split(",")[1]
                        resolve(base64)
                    } else {
                        reject(new Error("Failed to read file"))
                    }
                }
                reader.onerror = reject
                reader.readAsDataURL(file)
            })
        }

        try {
            if (attachments.pdfs) {
                for (const pdf of attachments.pdfs) {
                    const b64 = await readFile(pdf)
                    formData.pdfs.push({
                        filename: pdf.name,
                        content_base64: b64,
                    })
                }
            }
            if (attachments.algorithm) {
                const b64 = await readFile(attachments.algorithm)
                formData.custom_algorithm_file = {
                    filename: attachments.algorithm.name,
                    content_base64: b64,
                }
            }

            abortControllerRef.current = new AbortController()

            const response = await fetch(`${API_BASE}/api/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData),
                signal: abortControllerRef.current.signal,
            })

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`)
            }

            if (!response.body) throw new Error("No response body")

            const reader = response.body.getReader()
            const decoder = new TextDecoder()

            // Create placeholder assistant message
            const assistantMsgId = uuidv4()
            setMessages((prev) => [
                ...prev,
                {
                    id: assistantMsgId,
                    role: "assistant",
                    content: "",
                    tools: [],
                },
            ])

            let buffer = ""

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split("\n\n")
                buffer = lines.pop() || ""

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const jsonStr = line.slice(6)
                        if (jsonStr === "[DONE]") continue

                        try {
                            const event = JSON.parse(jsonStr)

                            setMessages((prev) => {
                                const newMsgs = [...prev]
                                const lastMsg =
                                    newMsgs[newMsgs.length - 1]

                                if (lastMsg.id !== assistantMsgId)
                                    return prev

                                if (event.type === "tool_start") {
                                    lastMsg.tools = [
                                        ...(lastMsg.tools || []),
                                        {
                                            tool: event.tool,
                                            status: "running",
                                        },
                                    ]
                                } else if (event.type === "tool_end") {
                                    if (lastMsg.tools) {
                                        const toolIdx =
                                            lastMsg.tools.findLastIndex(
                                                (t: ToolCall) =>
                                                    t.tool ===
                                                    event.tool &&
                                                    t.status === "running"
                                            )
                                        if (toolIdx !== -1) {
                                            lastMsg.tools[
                                                toolIdx
                                            ].status = "completed"
                                            lastMsg.tools[
                                                toolIdx
                                            ].result = event.result
                                        }
                                    }
                                } else if (event.type === "algorithm_select") {
                                    lastMsg.algorithmChoices = event.algorithms
                                } else if (event.type === "choice_prompt") {
                                    lastMsg.choicePrompt = {
                                        id: event.id,
                                        title: event.title,
                                        options: event.options,
                                        multiSelect: event.multi_select ?? false,
                                    }
                                } else if (event.type === "algorithm_coded") {
                                    setCodedAlgorithms((prev) =>
                                        prev.includes(event.name) ? prev : [...prev, event.name]
                                    )
                                } else if (event.type === "benchmark_start") {
                                    setBenchmarkProgress({
                                        algorithms: event.algorithms.map((a: { name: string; total_runs: number }) => ({
                                            name: a.name,
                                            completed: 0,
                                            total: a.total_runs,
                                        })),
                                        status: "running",
                                    })
                                } else if (event.type === "benchmark_progress") {
                                    setBenchmarkProgress((prev) => ({
                                        ...prev,
                                        algorithms: prev.algorithms.map((a) =>
                                            a.name === event.algorithm
                                                ? { ...a, completed: event.completed, total: event.total }
                                                : a
                                        ),
                                    }))
                                } else if (event.type === "benchmark_complete") {
                                    setBenchmarkProgress((prev) => ({
                                        ...prev,
                                        status: "complete",
                                    }))
                                } else if (event.type === "done") {
                                    setSessionId(event.session_id)
                                    lastMsg.content = event.reply
                                    if (event.plot_image) {
                                        lastMsg.plotImage =
                                            event.plot_image
                                    }
                                } else if (event.type === "error") {
                                    lastMsg.content += `\n\nError: ${event.error}`
                                }

                                return newMsgs
                            })
                        } catch (e) {
                            console.error("Error parsing SSE:", e)
                        }
                    }
                }
            }
        } catch (error: any) {
            if (error.name === "AbortError") return
            console.error("Chat error:", error)
            setMessages((prev) => [
                ...prev,
                {
                    id: uuidv4(),
                    role: "assistant",
                    content: `Error: ${error.message}`,
                },
            ])
        } finally {
            setIsLoading(false)
        }
    }

    return {
        messages,
        isLoading,
        sessionId,
        setSessionId,
        sendMessage,
        clearChat,
        loadSession,
        codedAlgorithms,
        benchmarkProgress,
        setBenchmarkProgress,
    }
}
