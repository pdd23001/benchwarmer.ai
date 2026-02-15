import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Paperclip, FileCode, ArrowUp, X } from "lucide-react"

interface ChatInputProps {
    onSend: (text: string, files: { pdfs?: File[]; algorithm?: File }) => void
    disabled?: boolean
    placeholder?: string
    className?: string
    /** If true, hide the PDF/Code attachment buttons (used in welcome layout where separate zones exist) */
    hideAttachButtons?: boolean
    /** Extra content rendered in the bottom-left area (e.g. model selectors) */
    bottomLeftContent?: React.ReactNode
    /** Controlled file state — when provided, ChatInput does not manage its own file state */
    pdfs?: File[]
    onPdfsChange?: (pdfs: File[]) => void
    algo?: File
    onAlgoChange?: (algo: File | undefined) => void
}

export function ChatInput({
    onSend,
    disabled,
    placeholder = "Message Benchwarmer...",
    className,
    hideAttachButtons = false,
    bottomLeftContent,
    pdfs: controlledPdfs,
    onPdfsChange,
    algo: controlledAlgo,
    onAlgoChange,
}: ChatInputProps) {
    const [input, setInput] = useState("")

    // Internal file state (fallback when not controlled)
    const [internalPdfs, setInternalPdfs] = useState<File[]>([])
    const [internalAlgo, setInternalAlgo] = useState<File | undefined>(undefined)

    // Use controlled state if provided, otherwise internal
    const pdfs = controlledPdfs ?? internalPdfs
    const setPdfs = onPdfsChange ?? setInternalPdfs
    const algo = controlledAlgo ?? internalAlgo
    const setAlgo = onAlgoChange ?? setInternalAlgo

    const pdfInputRef = useRef<HTMLInputElement>(null)
    const algoInputRef = useRef<HTMLInputElement>(null)
    const textareaRef = useRef<HTMLTextAreaElement>(null)

    const handleSubmit = (e?: React.FormEvent) => {
        e?.preventDefault()
        if (!input.trim() && !pdfs.length && !algo) return
        onSend(input, { pdfs: pdfs.length ? pdfs : undefined, algorithm: algo })
        setInput("")
        setPdfs([])
        setAlgo(undefined)
        if (textareaRef.current) textareaRef.current.style.height = "auto"
    }

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault()
            handleSubmit()
        }
    }

    const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setInput(e.target.value)
        e.target.style.height = "auto"
        e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`
    }

    const hasContent = input.trim() || pdfs.length > 0 || algo

    return (
        <div className={`relative flex flex-col rounded-2xl border border-border bg-secondary/50 shadow-sm transition-colors focus-within:border-muted-foreground/30 ${className || ""}`}>
            {/* File Previews */}
            {(pdfs.length > 0 || algo) && (
                <div className="flex gap-2 px-4 pt-3 overflow-x-auto">
                    {pdfs.map((f, i) => (
                        <div
                            key={i}
                            className="flex items-center gap-2 bg-muted px-3 py-1.5 rounded-lg text-xs font-medium"
                        >
                            <Paperclip size={12} className="text-muted-foreground" />
                            <span className="truncate max-w-[120px]">{f.name}</span>
                            <button
                                type="button"
                                onClick={() => setPdfs(pdfs.filter((_, idx) => idx !== i))}
                                className="text-muted-foreground hover:text-foreground transition-colors"
                            >
                                <X size={12} />
                            </button>
                        </div>
                    ))}
                    {algo && (
                        <div className="flex items-center gap-2 bg-violet-500/10 px-3 py-1.5 rounded-lg text-xs font-medium border border-violet-500/20">
                            <FileCode size={12} className="text-violet-400" />
                            <span className="truncate max-w-[120px]">{algo.name}</span>
                            <button
                                type="button"
                                onClick={() => setAlgo(undefined)}
                                className="text-muted-foreground hover:text-foreground transition-colors"
                            >
                                <X size={12} />
                            </button>
                        </div>
                    )}
                </div>
            )}

            {/* Text Area */}
            <textarea
                ref={textareaRef}
                value={input}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
                placeholder={placeholder}
                className="w-full flex-1 resize-none bg-transparent placeholder:text-muted-foreground/50 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 min-h-[52px] px-4 pt-3 pb-1 text-[15px] leading-relaxed"
                disabled={disabled}
                rows={1}
            />

            {/* Bottom Bar */}
            <div className="flex items-center justify-between px-3 pb-2.5 pt-0">
                <div className="flex items-center gap-1">
                    {bottomLeftContent}
                    {!hideAttachButtons && (
                        <>
                            <input
                                type="file"
                                multiple
                                accept=".pdf"
                                className="hidden"
                                ref={pdfInputRef}
                                onChange={(e) => {
                                    if (e.target.files)
                                        setPdfs([...pdfs, ...Array.from(e.target.files)])
                                }}
                            />
                            <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                className="h-8 gap-1.5 text-muted-foreground hover:text-foreground rounded-lg px-2.5"
                                onClick={() => pdfInputRef.current?.click()}
                                title="Attach PDFs"
                            >
                                <Paperclip className="h-4 w-4" />
                                <span className="text-xs hidden sm:inline">PDF</span>
                            </Button>

                            <input
                                type="file"
                                accept=".py"
                                className="hidden"
                                ref={algoInputRef}
                                onChange={(e) => {
                                    if (e.target.files?.[0]) setAlgo(e.target.files[0])
                                }}
                            />
                            <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                className="h-8 gap-1.5 text-muted-foreground hover:text-foreground rounded-lg px-2.5"
                                onClick={() => algoInputRef.current?.click()}
                                title="Attach Algorithm (.py)"
                            >
                                <FileCode className="h-4 w-4" />
                                <span className="text-xs hidden sm:inline">Code</span>
                            </Button>
                        </>
                    )}
                </div>

                <Button
                    type="button"
                    size="icon"
                    disabled={disabled || !hasContent}
                    onClick={() => handleSubmit()}
                    className="h-8 w-8 rounded-lg bg-foreground text-background hover:bg-foreground/90 disabled:opacity-30"
                >
                    <ArrowUp className="h-4 w-4" />
                    <span className="sr-only">Send</span>
                </Button>
            </div>
        </div>
    )
}
