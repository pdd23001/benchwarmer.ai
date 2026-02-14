import { Beaker, History, Settings, Plus } from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";

export function Sidebar() {
    return (
        <aside className="w-64 border-r border-border bg-card/50 backdrop-blur-sm hidden md:flex flex-col h-screen sticky top-0">
            <div className="p-4 border-b border-border">
                <Link href="/" className="flex items-center gap-2 font-semibold text-lg hover:opacity-80 transition-opacity">
                    <Beaker className="w-5 h-5 text-primary" />
                    <span>Benchwarmer.ai</span>
                </Link>
            </div>

            <div className="p-2">
                <button className="w-full flex items-center gap-2 px-3 py-2 text-sm font-medium bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors">
                    <Plus className="w-4 h-4" />
                    New Benchmark
                </button>
            </div>

            <div className="flex-1 overflow-auto py-2">
                <div className="px-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                    History
                </div>
                <nav className="space-y-1 px-2">
                    {["QuickSort vs MergeSort", "Dijkstra vs A*", "Binary Tree Search"].map((item, i) => (
                        <button
                            key={i}
                            className={cn(
                                "w-full flex items-center gap-2 px-3 py-2 text-sm rounded-md transition-colors",
                                "text-muted-foreground hover:text-foreground hover:bg-muted"
                            )}
                        >
                            <History className="w-4 h-4" />
                            <span className="truncate">{item}</span>
                        </button>
                    ))}
                </nav>
            </div>

            <div className="p-4 border-t border-border">
                <button className="w-full flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-muted rounded-md transition-colors">
                    <Settings className="w-4 h-4" />
                    Settings
                </button>
            </div>
        </aside>
    );
}
