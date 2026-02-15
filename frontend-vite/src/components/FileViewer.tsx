import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { FileCode, Loader2 } from "lucide-react"
import { CodeViewer } from "./CodeViewer"
import { cn } from "@/lib/utils"

interface Algorithm {
    name: string
    created_at: string
}

interface FileViewerProps {
    className?: string
    isCollapsed?: boolean
}

export function FileViewer({ className, isCollapsed }: FileViewerProps) {
    const [algorithms, setAlgorithms] = useState<Algorithm[]>([])
    const [selectedAlgo, setSelectedAlgo] = useState<string | null>(null)
    const [code, setCode] = useState<string>("")
    const [isLoading, setIsLoading] = useState(false)
    const [isViewerOpen, setIsViewerOpen] = useState(false)

    const fetchAlgorithms = async () => {
        try {
            const res = await fetch("/api/algorithms")
            if (res.ok) {
                const data = await res.json()
                setAlgorithms(data)
            }
        } catch (error) {
            console.error("Failed to fetch algorithms", error)
        }
    }

    // Poll for new algorithms every 5 seconds or just fetch on mount
    useEffect(() => {
        fetchAlgorithms()
        const interval = setInterval(fetchAlgorithms, 5000)
        return () => clearInterval(interval)
    }, [])

    const handleSelectAlgo = async (name: string) => {
        setIsLoading(true)
        setSelectedAlgo(name)
        try {
            const res = await fetch(`/api/algorithms/${name}`)
            if (res.ok) {
                const data = await res.json()
                setCode(data.code)
                setIsViewerOpen(true)
            }
        } catch (error) {
            console.error("Failed to fetch algorithm code", error)
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <>
            <div className={cn("flex flex-col h-full", className)}>
                {!isCollapsed && <div className="px-4 pb-2 text-xs font-semibold text-muted-foreground/70 tracking-wider uppercase">Generated Algorithms</div>}
                <ScrollArea className="flex-1">
                    <div className="px-2 space-y-1">
                        {algorithms.length === 0 && !isCollapsed && (
                            <div className="px-2 py-4 text-xs text-muted-foreground text-center">
                                No algorithms generated yet.
                            </div>
                        )}
                        {algorithms.map(algo => (
                            <Button
                                key={algo.name}
                                variant="ghost"
                                className={cn("w-full justify-start gap-2 h-9 text-xs font-normal truncate", isCollapsed ? "justify-center px-0" : "")}
                                onClick={() => handleSelectAlgo(algo.name)}
                                title={algo.name}
                            >
                                <FileCode className="h-3.5 w-3.5 shrink-0 text-blue-500" />
                                {!isCollapsed && <span className="truncate">{algo.name}.py</span>}
                                {isLoading && selectedAlgo === algo.name && <Loader2 className="h-3 w-3 animate-spin ml-auto" />}
                            </Button>
                        ))}
                    </div>
                </ScrollArea>
            </div>

            <CodeViewer
                isOpen={isViewerOpen}
                onClose={() => setIsViewerOpen(false)}
                title={`${selectedAlgo}.py`}
                code={code}
            />
        </>
    )
}
