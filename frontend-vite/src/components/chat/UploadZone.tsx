import { useRef, useState } from "react"
import { X } from "lucide-react"
import { cn } from "@/lib/utils"

interface UploadZoneProps {
    accept: string
    multiple?: boolean
    icon: React.ReactNode
    title: string
    subtitle: string
    files: File[]
    onFilesChange: (files: File[]) => void
    className?: string
}

export function UploadZone({
    accept,
    multiple = false,
    icon,
    title,
    subtitle,
    files,
    onFilesChange,
    className,
}: UploadZoneProps) {
    const inputRef = useRef<HTMLInputElement>(null)
    const [isDragOver, setIsDragOver] = useState(false)

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragOver(true)
    }

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragOver(false)
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragOver(false)

        const droppedFiles = Array.from(e.dataTransfer.files)
        // Filter by accepted extensions
        const extensions = accept.split(",").map((a) => a.trim().toLowerCase())
        const filtered = droppedFiles.filter((f) =>
            extensions.some((ext) => f.name.toLowerCase().endsWith(ext))
        )
        if (filtered.length === 0) return

        if (multiple) {
            onFilesChange([...files, ...filtered])
        } else {
            onFilesChange([filtered[0]])
        }
    }

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files) return
        const selected = Array.from(e.target.files)
        if (multiple) {
            onFilesChange([...files, ...selected])
        } else {
            onFilesChange(selected.slice(0, 1))
        }
        // Reset input so same file can be re-selected
        e.target.value = ""
    }

    const removeFile = (idx: number) => {
        onFilesChange(files.filter((_, i) => i !== idx))
    }

    return (
        <div
            onClick={() => inputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={cn(
                "relative flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-4 py-6 cursor-pointer transition-all duration-200",
                isDragOver
                    ? "border-violet-500 bg-violet-500/10 scale-[1.02]"
                    : "border-border/60 bg-secondary/30 hover:border-muted-foreground/40 hover:bg-secondary/50",
                className
            )}
        >
            <input
                ref={inputRef}
                type="file"
                accept={accept}
                multiple={multiple}
                className="hidden"
                onChange={handleInputChange}
            />

            <div className="text-muted-foreground/60">{icon}</div>
            <div className="text-center">
                <div className="text-sm font-medium text-foreground/80">{title}</div>
                <div className="text-[11px] text-muted-foreground/50 mt-0.5">{subtitle}</div>
            </div>

            {/* Uploaded file chips */}
            {files.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-1 w-full justify-center" onClick={(e) => e.stopPropagation()}>
                    {files.map((f, i) => (
                        <div
                            key={i}
                            className="flex items-center gap-1.5 bg-muted/80 px-2.5 py-1 rounded-lg text-[11px] font-medium max-w-full"
                        >
                            <span className="truncate max-w-[100px]">{f.name}</span>
                            <button
                                type="button"
                                onClick={(e) => {
                                    e.stopPropagation()
                                    removeFile(i)
                                }}
                                className="text-muted-foreground hover:text-foreground transition-colors flex-shrink-0"
                            >
                                <X size={10} />
                            </button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
