import { Check, Loader2 } from "lucide-react"
import { type BenchmarkProgress } from "@/hooks/use-chat"

interface SandboxPanelProps {
    progress: BenchmarkProgress
}

export function SandboxPanel({ progress }: SandboxPanelProps) {
    const allComplete = progress.status === "complete"

    return (
        <div className="flex flex-col h-full bg-background border-r border-border">
            {/* Header */}
            <div className="flex items-center gap-2.5 h-12 px-4 border-b border-border/60 flex-shrink-0">
                {allComplete ? (
                    <Check className="h-4 w-4 text-emerald-400" />
                ) : (
                    <Loader2 className="h-4 w-4 animate-spin text-violet-400" />
                )}
                <span className="text-sm font-medium">
                    {allComplete ? "Benchmarks Complete" : "Running Benchmarks"}
                </span>
            </div>

            {/* Sandbox boxes */}
            <div className="flex-1 overflow-auto p-4">
                <div className="grid gap-3">
                    {progress.algorithms.map((algo) => {
                        const pct = algo.total > 0 ? Math.round((algo.completed / algo.total) * 100) : 0
                        const isComplete = pct >= 100

                        return (
                            <div
                                key={algo.name}
                                className="relative rounded-xl border border-border/60 overflow-hidden bg-card"
                            >
                                {/* Algorithm name header */}
                                <div className="flex items-center justify-between px-3 py-2 border-b border-border/40">
                                    <span className="text-xs font-medium truncate text-foreground">
                                        {algo.name}
                                    </span>
                                    <span className={`text-[11px] font-mono ${isComplete ? "text-emerald-400" : "text-muted-foreground"}`}>
                                        {isComplete ? "Complete" : `${pct}%`}
                                    </span>
                                </div>

                                {/* Sand container */}
                                <div className="relative h-28 bg-muted/20">
                                    {/* Sand fill — rises from bottom */}
                                    <div
                                        className="absolute bottom-0 left-0 right-0 transition-all duration-700 ease-out"
                                        style={{ height: `${pct}%` }}
                                    >
                                        {/* Sand gradient */}
                                        <div className="absolute inset-0 bg-gradient-to-t from-violet-500/40 via-violet-400/25 to-violet-300/10" />

                                        {/* Sand grain texture — subtle dots */}
                                        <div className="absolute inset-0 opacity-30">
                                            <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
                                                <defs>
                                                    <pattern id={`sand-${algo.name}`} x="0" y="0" width="6" height="6" patternUnits="userSpaceOnUse">
                                                        <circle cx="1" cy="1" r="0.5" fill="currentColor" className="text-violet-300" />
                                                        <circle cx="4" cy="4" r="0.4" fill="currentColor" className="text-violet-200" />
                                                    </pattern>
                                                </defs>
                                                <rect width="100%" height="100%" fill={`url(#sand-${algo.name})`} />
                                            </svg>
                                        </div>

                                        {/* Surface wave effect */}
                                        {!isComplete && (
                                            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-violet-400/30 to-transparent animate-pulse" />
                                        )}
                                    </div>

                                    {/* Center percentage */}
                                    <div className="absolute inset-0 flex items-center justify-center">
                                        {isComplete ? (
                                            <Check className="h-6 w-6 text-emerald-400" />
                                        ) : (
                                            <span className="text-lg font-bold text-foreground/60 font-mono tabular-nums">
                                                {pct}%
                                            </span>
                                        )}
                                    </div>

                                    {/* Run counter */}
                                    <div className="absolute bottom-1.5 right-2 text-[10px] text-muted-foreground/50 font-mono">
                                        {algo.completed}/{algo.total} runs
                                    </div>
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>
        </div>
    )
}
