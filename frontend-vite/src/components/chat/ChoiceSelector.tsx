import { useState } from "react"
import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"
import { cn } from "@/lib/utils"
import type { ChoiceOption } from "@/hooks/use-chat"

interface ChoiceSelectorProps {
    title?: string
    options: ChoiceOption[]
    multiSelect?: boolean
    onConfirm: (values: string[]) => void
    disabled?: boolean
}

export function ChoiceSelector({
    title,
    options,
    multiSelect = false,
    onConfirm,
    disabled = false,
}: ChoiceSelectorProps) {
    const [selected, setSelected] = useState<Set<string>>(new Set())

    const toggle = (value: string) => {
        if (disabled) return
        setSelected((prev) => {
            const next = new Set(prev)
            if (multiSelect) {
                if (next.has(value)) {
                    next.delete(value)
                } else {
                    next.add(value)
                }
            } else {
                // Single select — toggle or replace
                if (next.has(value)) {
                    next.clear()
                } else {
                    next.clear()
                    next.add(value)
                }
            }
            return next
        })
    }

    const selectAll = () => {
        if (disabled) return
        if (selected.size === options.length) {
            setSelected(new Set())
        } else {
            setSelected(new Set(options.map((o) => o.value)))
        }
    }

    const handleConfirm = () => {
        if (selected.size === 0) return
        onConfirm(Array.from(selected))
    }

    return (
        <div className="space-y-3 w-full">
            <div className="flex items-center justify-between">
                {title && (
                    <span className="text-xs font-medium text-muted-foreground">
                        {title}
                    </span>
                )}
                {multiSelect && (
                    <button
                        onClick={selectAll}
                        disabled={disabled}
                        className="text-xs text-violet-400 hover:text-violet-300 transition-colors disabled:opacity-50"
                    >
                        {selected.size === options.length ? "Deselect all" : "Select all"}
                    </button>
                )}
            </div>

            <div className="flex flex-wrap gap-2">
                {options.map((opt) => {
                    const isSelected = selected.has(opt.value)
                    return (
                        <button
                            key={opt.value}
                            onClick={() => toggle(opt.value)}
                            disabled={disabled}
                            className={cn(
                                "text-left rounded-xl px-4 py-2.5 border transition-all duration-150",
                                "hover:border-violet-500/40 disabled:opacity-50 disabled:cursor-not-allowed",
                                isSelected
                                    ? "bg-violet-500/15 border-violet-500/40 text-foreground"
                                    : "bg-secondary/50 border-border text-muted-foreground hover:bg-secondary"
                            )}
                        >
                            <div className="text-xs font-medium">
                                {opt.label}
                            </div>
                            {opt.description && (
                                <div className="text-[11px] text-muted-foreground/70 leading-snug mt-0.5">
                                    {opt.description}
                                </div>
                            )}
                        </button>
                    )
                })}
            </div>

            <Button
                onClick={handleConfirm}
                disabled={disabled || selected.size === 0}
                size="sm"
                className="gap-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg"
            >
                Continue
                <ArrowRight className="h-3.5 w-3.5" />
            </Button>
        </div>
    )
}
