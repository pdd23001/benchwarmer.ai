"use client";

import { useState } from "react";
import { Play, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface ExperimentFormProps {
    onSubmit: (query: string) => void;
    isLoading: boolean;
}

export function ExperimentForm({ onSubmit, isLoading }: ExperimentFormProps) {
    const [query, setQuery] = useState("");

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (query.trim()) {
            onSubmit(query);
        }
    };

    return (
        <form onSubmit={handleSubmit} className="w-full space-y-4">
            <div className="relative">
                <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Describe your experiment (e.g., 'Benchmark Dijkstra vs A* on a 100x100 grid')"
                    className={cn(
                        "w-full min-h-[120px] p-4 rounded-xl bg-card border border-border text-foreground",
                        "placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all",
                        "resize-none font-medium text-lg"
                    )}
                    disabled={isLoading}
                />
                <div className="absolute bottom-3 right-3">
                    <span className="text-xs text-muted-foreground">Cmd+Enter to run</span>
                </div>
            </div>

            <div className="flex justify-end">
                <button
                    type="submit"
                    disabled={!query.trim() || isLoading}
                    className={cn(
                        "flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all shadow-sm",
                        "bg-primary text-primary-foreground hover:opacity-90 active:scale-95 disabled:opacity-50 disabled:pointer-events-none"
                    )}
                >
                    {isLoading ? (
                        <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            <span>Initializing Lab...</span>
                        </>
                    ) : (
                        <>
                            <Play className="w-5 h-5 fill-current" />
                            <span>Run Experiment</span>
                        </>
                    )}
                </button>
            </div>
        </form>
    );
}
