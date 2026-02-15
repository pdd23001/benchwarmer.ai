import { useState, useEffect } from "react"
import { X, Copy, Check, FileCode, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"

interface CodeViewerProps {
    algorithmName: string
    onClose: () => void
}

export function CodeViewer({ algorithmName, onClose }: CodeViewerProps) {
    const [code, setCode] = useState<string | null>(null)
    const [loading, setLoading] = useState(true)
    const [copied, setCopied] = useState(false)

    useEffect(() => {
        setLoading(true)
        setCode(null)
        fetch(`/api/algorithms/${algorithmName}`)
            .then((res) => {
                if (!res.ok) throw new Error("Not found")
                return res.json()
            })
            .then((data) => setCode(data.code))
            .catch(() => setCode("# Error: could not load algorithm code"))
            .finally(() => setLoading(false))
    }, [algorithmName])

    const handleCopy = async () => {
        if (!code) return
        await navigator.clipboard.writeText(code)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
    }

    return (
        <div className="flex flex-col h-full bg-background border-r border-border">
            {/* Header */}
            <div className="flex items-center justify-between h-12 px-4 border-b border-border/60 flex-shrink-0">
                <div className="flex items-center gap-2 min-w-0">
                    <FileCode className="h-4 w-4 text-violet-400 flex-shrink-0" />
                    <span className="text-sm font-medium truncate">
                        {algorithmName}.py
                    </span>
                </div>
                <div className="flex items-center gap-1">
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-muted-foreground hover:text-foreground"
                        onClick={handleCopy}
                        title="Copy code"
                    >
                        {copied ? (
                            <Check className="h-3.5 w-3.5 text-emerald-400" />
                        ) : (
                            <Copy className="h-3.5 w-3.5" />
                        )}
                    </Button>
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-muted-foreground hover:text-foreground"
                        onClick={onClose}
                        title="Close"
                    >
                        <X className="h-3.5 w-3.5" />
                    </Button>
                </div>
            </div>

            {/* Code content */}
            <div className="flex-1 overflow-auto">
                {loading ? (
                    <div className="flex items-center justify-center h-full">
                        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                    </div>
                ) : (
                    <pre className="p-4 text-[13px] leading-relaxed font-mono text-foreground/90 whitespace-pre overflow-x-auto">
                        <code>{code}</code>
                    </pre>
                )}
            </div>
        </div>
    )
}
