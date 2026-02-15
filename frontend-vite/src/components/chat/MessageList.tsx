import { useRef, useEffect } from "react"
import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { type Message } from "@/hooks/use-chat"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Loader2, ChevronDown, BrainCircuit } from "lucide-react"
import { cn } from "@/lib/utils"
import { AlgorithmSelector } from "./AlgorithmSelector"
import { ChoiceSelector } from "./ChoiceSelector"

interface MessageListProps {
    messages: Message[]
    isLoading: boolean
    onAlgorithmSelect?: (indices: number[]) => void
    onChoiceSelect?: (choiceId: string, values: string[]) => void
}

/**
 * Strip the numbered algorithm listing and the "Which of these..."
 * prompt from an assistant message when we have structured data.
 */
function stripAlgorithmListText(content: string): string {
    return content
        // Remove "Which of these paper algorithms..." prompt (including multi-line)
        .replace(/Which of these[\s\S]*?\(Reply by index[^)]*\)\s*/gi, "")
        // Remove any line that looks like a numbered algorithm entry — covers all bold/unbold variants:
        //   "0. **name**: desc"  |  "**0. name**: desc"  |  "0. name: desc"  |  "**0.** **name**: desc"
        .replace(/^\s*\*{0,2}\d+\.?\*{0,2}\s+\*{0,2}[\w_]+\*{0,2}\s*[-–:].+$/gm, "")
        // Remove indented or standalone algorithm-name lines (no number prefix):
        //   "   lp_relaxation_rounding: ..."  |  "   **sdp_goemans_williamson**: ..."
        .replace(/^\s+\*{0,2}[a-z][\w]*\*{0,2}\s*[-–:].+$/gm, "")
        .replace(/^\s*\*{2}[\w]+\*{2}\s*[-–:].+$/gm, "")
        // Remove bullet-style: "- **name**: description" or "• name: desc"
        .replace(/^\s*[-•]\s+\*{0,2}[\w_]+\*{0,2}\s*[-–:].+$/gm, "")
        // Remove sentences introducing the now-removed list
        .replace(/[^.!?\n]*(?:algorithm|implement|extracted|found|loaded)[^.!?\n]*[:]\s*$/gmi, "")
        // Remove dangling "**0." or similar bold-number fragments left inline
        .replace(/\*{0,2}\d+\.\*{0,2}\s*\*{0,2}[\w_]+\*{0,2}\s*[-–:].*$/gm, "")
        // Clean up trailing colons, dots+colons, or lone punctuation left over
        .replace(/\s*[:.]\s*$/gm, "")
        // Collapse multiple blank lines
        .replace(/\n{3,}/g, "\n\n")
        .trim()
}

/**
 * Strip interactive-choice prompt text from a message when we have a selector.
 */
function stripChoicePromptText(content: string, choiceId: string): string {
    if (choiceId === "instance_source") {
        return content
            // Remove "How would you like to provide instances?" and everything after
            .replace(/(?:Now, )?[Hh]ow would you like to provide instances.*$/gms, "")
            // Remove bold option lines like **generators** - description
            .replace(/^\s*\*?\*?(?:generators?|custom\s*JSON?|benchmark\s*suite)\*?\*?\s*[-–:].+$/gmi, "")
            .replace(/\n{3,}/g, "\n\n")
            .trim()
    }
    if (choiceId === "benchmark_suite") {
        return content
            // Remove "Available benchmark suites:" and the bullet list
            .replace(/(?:Available|📦.*?)benchmark suites:[\s\S]*$/gmi, "")
            // Remove "Which suite would you like to use?" prompts
            .replace(/Which suite would you like.*$/gmi, "")
            .replace(/\n{3,}/g, "\n\n")
            .trim()
    }
    if (choiceId === "benchmark_instances") {
        return content
            // Remove "Instances in 'xxx':" and the bullet list + prompt
            .replace(/(?:📋\s*)?Instances in[\s\S]*$/gmi, "")
            // Remove "Which instances would you like to load?" prompts
            .replace(/Which instances would you like.*$/gmi, "")
            .replace(/\n{3,}/g, "\n\n")
            .trim()
    }
    return content
}

function ParsedMessageContent({ content }: { content: string }) {
    const thinkRegex = /<think>([\s\S]*?)<\/think>/g
    const parts: { type: string; content: string }[] = []
    let lastIndex = 0
    let match

    while ((match = thinkRegex.exec(content)) !== null) {
        if (match.index > lastIndex) {
            parts.push({ type: "text", content: content.slice(lastIndex, match.index) })
        }
        parts.push({ type: "think", content: match[1] })
        lastIndex = thinkRegex.lastIndex
    }
    if (lastIndex < content.length) {
        parts.push({ type: "text", content: content.slice(lastIndex) })
    }

    if (parts.length === 0)
        return (
            <div className="prose-chat">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
            </div>
        )

    return (
        <div className="space-y-3">
            {parts.map((part, i) => {
                if (part.type === "think") {
                    return (
                        <Collapsible key={i} className="group">
                            <CollapsibleTrigger className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors py-1">
                                <BrainCircuit className="h-3.5 w-3.5" />
                                <span className="font-medium">Reasoning</span>
                                <ChevronDown className="h-3 w-3 transition-transform group-data-[state=open]:rotate-180" />
                            </CollapsibleTrigger>
                            <CollapsibleContent>
                                <div className="mt-2 pl-3 border-l-2 border-muted text-xs text-muted-foreground font-mono whitespace-pre-wrap leading-relaxed">
                                    {part.content.trim()}
                                </div>
                            </CollapsibleContent>
                        </Collapsible>
                    )
                }
                return (
                    <div key={i} className="prose-chat">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{part.content}</ReactMarkdown>
                    </div>
                )
            })}
        </div>
    )
}

export function MessageList({ messages, isLoading, onAlgorithmSelect, onChoiceSelect }: MessageListProps) {
    // Filter out hidden messages (e.g. auto-sent selections)
    const visibleMessages = messages.filter((m) => !m.hidden)
    const scrollRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({ behavior: "smooth" })
        }
    }, [messages])

    return (
        <div className="flex-1 overflow-y-auto w-full">
            <div className="max-w-3xl mx-auto px-4 py-6 flex flex-col gap-6">
                {visibleMessages.map((message) => (
                    <div key={message.id} className="w-full">
                        {message.role === "user" ? (
                            /* ── User Message ── */
                            <div className="flex justify-end">
                                <div className="max-w-[80%] bg-secondary text-secondary-foreground rounded-2xl rounded-br-md px-4 py-3">
                                    <p className="text-sm leading-relaxed whitespace-pre-wrap">
                                        {message.content}
                                    </p>
                                </div>
                            </div>
                        ) : (
                            /* ── Assistant Message ── */
                            <div className="flex gap-3 w-full">
                                <div className="flex-shrink-0 mt-1">
                                    <img src="/bench-logo.png" alt="Benchwarmer" className="h-7 w-7 rounded-full object-cover" />
                                </div>
                                <div className="flex-1 min-w-0 space-y-2">
                                    <span className="text-xs font-medium text-muted-foreground">
                                        benchwarmer
                                    </span>
                                    <div>
                                        <ParsedMessageContent
                                            content={(() => {
                                                let text = message.content
                                                if (message.algorithmChoices) {
                                                    text = stripAlgorithmListText(text)
                                                }
                                                if (message.choicePrompt) {
                                                    text = stripChoicePromptText(text, message.choicePrompt.id)
                                                }
                                                return text
                                            })()}
                                        />
                                    </div>

                                    {/* Interactive algorithm selector */}
                                    {message.algorithmChoices && message.algorithmChoices.length > 0 && (() => {
                                        // Determine if this selector was already confirmed
                                        // (a subsequent user message exists after this one)
                                        const msgIdx = messages.indexOf(message)
                                        const isConfirmed = msgIdx < messages.length - 1 &&
                                            messages.slice(msgIdx + 1).some(m => m.role === "user")
                                        return (
                                            <div className="pt-2">
                                                <AlgorithmSelector
                                                    algorithms={message.algorithmChoices}
                                                    onConfirm={(indices) => onAlgorithmSelect?.(indices)}
                                                    disabled={isConfirmed}
                                                />
                                            </div>
                                        )
                                    })()}

                                    {/* Interactive choice selector (instance source, etc.) */}
                                    {message.choicePrompt && (() => {
                                        const msgIdx = messages.indexOf(message)
                                        const isConfirmed = msgIdx < messages.length - 1 &&
                                            messages.slice(msgIdx + 1).some(m => m.role === "user")
                                        return (
                                            <div className="pt-2">
                                                <ChoiceSelector
                                                    title={message.choicePrompt.title}
                                                    options={message.choicePrompt.options}
                                                    multiSelect={message.choicePrompt.multiSelect}
                                                    onConfirm={(values) => onChoiceSelect?.(message.choicePrompt!.id, values)}
                                                    disabled={isConfirmed}
                                                />
                                            </div>
                                        )
                                    })()}

                                    {/* Tool calls */}
                                    {message.tools && message.tools.length > 0 && (
                                        <div className="flex flex-wrap gap-1.5 pt-1">
                                            {message.tools.map((tool, idx) => (
                                                <div
                                                    key={idx}
                                                    className={cn(
                                                        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium transition-colors",
                                                        tool.status === "running"
                                                            ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                                                            : tool.status === "completed"
                                                            ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                                            : "bg-red-500/10 text-red-400 border border-red-500/20"
                                                    )}
                                                >
                                                    <div
                                                        className={cn(
                                                            "h-1.5 w-1.5 rounded-full",
                                                            tool.status === "running"
                                                                ? "bg-amber-400 animate-pulse"
                                                                : tool.status === "completed"
                                                                ? "bg-emerald-400"
                                                                : "bg-red-400"
                                                        )}
                                                    />
                                                    <span className="font-mono">{tool.tool}</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    {/* Plot image */}
                                    {message.plotImage && (
                                        <img
                                            src={message.plotImage}
                                            alt="Benchmark Result"
                                            className="rounded-xl border border-border bg-card mt-3 shadow-sm max-w-md"
                                        />
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                ))}

                {/* Loading indicator */}
                {isLoading && (
                    <div className="flex gap-3 w-full">
                        <div className="flex-shrink-0 mt-1">
                            <img src="/bench-logo.png" alt="Benchwarmer" className="h-7 w-7 rounded-full object-cover" />
                        </div>
                        <div className="flex items-center gap-2 pt-1">
                            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                            <span className="text-sm text-muted-foreground">Thinking...</span>
                        </div>
                    </div>
                )}

                <div ref={scrollRef} />
            </div>
        </div>
    )
}
