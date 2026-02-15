import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Check, ArrowRight } from "lucide-react"
import { cn } from "@/lib/utils"

export interface AlgorithmChoice {
    index: number
    name: string
    approach: string
    source: string
}

interface AlgorithmSelectorProps {
    algorithms: AlgorithmChoice[]
    onConfirm: (selectedIndices: number[]) => void
    disabled?: boolean
}

export function AlgorithmSelector({
    algorithms,
    onConfirm,
    disabled = false,
}: AlgorithmSelectorProps) {
    const [selected, setSelected] = useState<Set<number>>(new Set())

    const toggle = (index: number) => {
        if (disabled) return
        setSelected((prev) => {
            const next = new Set(prev)
            if (next.has(index)) {
                next.delete(index)
            } else {
                next.add(index)
            }
            return next
        })
    }

    const selectAll = () => {
        if (disabled) return
        if (selected.size === algorithms.length) {
            setSelected(new Set())
        } else {
            setSelected(new Set(algorithms.map((a) => a.index)))
        }
    }

    const handleConfirm = () => {
        if (selected.size === 0) return
        const indices = Array.from(selected).sort((a, b) => a - b)
        onConfirm(indices)
    }

    // Group algorithms by source
    const grouped = algorithms.reduce<Record<string, AlgorithmChoice[]>>(
        (acc, algo) => {
            const key = algo.source || "Unknown"
            if (!acc[key]) acc[key] = []
            acc[key].push(algo)
            return acc
        },
        {}
    )

    const sources = Object.keys(grouped)

    return (
        <div className="space-y-4 w-full">
            <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">
                    Select algorithms to implement
                </span>
                <button
                    onClick={selectAll}
                    disabled={disabled}
                    className="text-xs text-violet-400 hover:text-violet-300 transition-colors disabled:opacity-50"
                >
                    {selected.size === algorithms.length
                        ? "Deselect all"
                        : "Select all"}
                </button>
            </div>

            <div
                className={cn(
                    "grid gap-4",
                    sources.length === 1
                        ? "grid-cols-1"
                        : sources.length === 2
                        ? "grid-cols-2"
                        : "grid-cols-3"
                )}
            >
                {sources.map((source) => (
                    <div key={source} className="space-y-2">
                        <div className="text-[11px] font-medium text-muted-foreground/60 uppercase tracking-wider truncate px-1" title={source}>
                            {source}
                        </div>
                        <div className="space-y-1.5">
                            {grouped[source].map((algo) => {
                                const isSelected = selected.has(algo.index)
                                return (
                                    <button
                                        key={algo.index}
                                        onClick={() => toggle(algo.index)}
                                        disabled={disabled}
                                        className={cn(
                                            "w-full text-left rounded-xl px-3.5 py-2.5 border transition-all duration-150",
                                            "hover:border-violet-500/40 disabled:opacity-50 disabled:cursor-not-allowed",
                                            isSelected
                                                ? "bg-violet-500/15 border-violet-500/40 text-foreground"
                                                : "bg-secondary/50 border-border text-muted-foreground hover:bg-secondary"
                                        )}
                                    >
                                        <div className="flex items-start gap-2.5">
                                            <div
                                                className={cn(
                                                    "mt-0.5 flex-shrink-0 h-4 w-4 rounded border flex items-center justify-center transition-colors",
                                                    isSelected
                                                        ? "bg-violet-500 border-violet-500"
                                                        : "border-muted-foreground/30"
                                                )}
                                            >
                                                {isSelected && (
                                                    <Check className="h-3 w-3 text-white" />
                                                )}
                                            </div>
                                            <div className="min-w-0">
                                                <div className="text-xs font-medium truncate">
                                                    {algo.name}
                                                </div>
                                                <div className="text-[11px] text-muted-foreground/70 leading-snug mt-0.5 line-clamp-2">
                                                    {algo.approach}
                                                </div>
                                            </div>
                                        </div>
                                    </button>
                                )
                            })}
                        </div>
                    </div>
                ))}
            </div>

            <Button
                onClick={handleConfirm}
                disabled={disabled || selected.size === 0}
                size="sm"
                className="gap-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg"
            >
                Implement {selected.size > 0 ? `(${selected.size})` : ""}
                <ArrowRight className="h-3.5 w-3.5" />
            </Button>
        </div>
    )
}
