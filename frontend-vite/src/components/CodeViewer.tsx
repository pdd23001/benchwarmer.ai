import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"

interface CodeViewerProps {
    isOpen: boolean
    onClose: () => void
    title: string
    code: string
}

export function CodeViewer({ isOpen, onClose, title, code }: CodeViewerProps) {
    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="max-w-4xl h-[80vh] flex flex-col p-0 gap-0">
                <DialogHeader className="px-6 py-4 border-b">
                    <DialogTitle className="font-mono text-lg">{title}</DialogTitle>
                </DialogHeader>
                <div className="flex-1 overflow-hidden bg-slate-950 text-slate-50 relative">
                    <ScrollArea className="h-full w-full">
                        <pre className="p-6 font-mono text-sm leading-relaxed whitespace-pre font-medium">
                            {code}
                        </pre>
                    </ScrollArea>
                </div>
            </DialogContent>
        </Dialog>
    )
}
