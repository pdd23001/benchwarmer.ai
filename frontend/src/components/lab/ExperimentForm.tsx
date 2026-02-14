"use client";

import { useState, useRef } from "react";
import { Play, Loader2, Upload, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface ExperimentFormProps {
    onSubmit: (query: string, pdfs: File[], pyFiles: File[], algorithmDescription?: string) => void;
    isLoading: boolean;
}

export function ExperimentForm({ onSubmit, isLoading }: ExperimentFormProps) {
    const [query, setQuery] = useState("");
    const [algorithmDescription, setAlgorithmDescription] = useState("");
    const [pdfFiles, setPdfFiles] = useState<File[]>([]);
    const [pyFiles, setPyFiles] = useState<File[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const pyFileInputRef = useRef<HTMLInputElement>(null);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (query.trim()) {
            onSubmit(query, pdfFiles, pyFiles, algorithmDescription || undefined);
        }
    };

    const handlePdfChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []);
        const pdfOnly = files.filter(f => f.type === "application/pdf" || f.name.endsWith(".pdf"));
        setPdfFiles(prev => [...prev, ...pdfOnly]);
    };

    const handlePyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []);
        const pyOnly = files.filter(f => f.type === "text/x-python" || f.name.endsWith(".py"));
        setPyFiles(prev => [...prev, ...pyOnly]);
    };

    const removePdfFile = (index: number) => {
        setPdfFiles(prev => prev.filter((_, i) => i !== index));
    };

    const removePyFile = (index: number) => {
        setPyFiles(prev => prev.filter((_, i) => i !== index));
    };

    const triggerPdfInput = () => {
        fileInputRef.current?.click();
    };

    const triggerPyInput = () => {
        pyFileInputRef.current?.click();
    };

    return (
        <form onSubmit={handleSubmit} className="w-full space-y-4">
            {/* Problem Description */}
            <div className="relative">
                <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => {
                        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSubmit(e);
                        }
                    }}
                    placeholder="Describe your problem (e.g., 'Compare max cut algorithms on sparse graphs')"
                    className={cn(
                        "w-full min-h-[100px] p-4 rounded-xl bg-card border border-border text-foreground",
                        "placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all",
                        "resize-none font-medium text-base"
                    )}
                    disabled={isLoading}
                />
                <div className="absolute bottom-3 right-3">
                    <span className="text-xs text-muted-foreground">Ctrl+Enter to run</span>
                </div>
            </div>

            {/* Python File Upload Section (Your Algorithm) */}
            <div className="space-y-2">
                <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-foreground">
                        Your Algorithm (.py file)
                    </label>
                    <button
                        type="button"
                        onClick={triggerPyInput}
                        disabled={isLoading}
                        className={cn(
                            "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all",
                            "bg-card border border-border hover:bg-muted disabled:opacity-50 disabled:pointer-events-none"
                        )}
                    >
                        <Upload className="w-4 h-4" />
                        <span>Add .py</span>
                    </button>
                </div>

                <input
                    ref={pyFileInputRef}
                    type="file"
                    accept=".py,text/x-python"
                    multiple
                    onChange={handlePyChange}
                    className="hidden"
                />

                {pyFiles.length > 0 && (
                    <div className="space-y-2">
                        {pyFiles.map((file, index) => (
                            <div
                                key={index}
                                className="flex items-center justify-between p-3 rounded-lg bg-card border border-border"
                            >
                                <div className="flex items-center gap-2 flex-1 min-w-0">
                                    <span className="text-sm font-medium truncate">{file.name}</span>
                                    <span className="text-xs text-muted-foreground">
                                        ({(file.size / 1024).toFixed(0)} KB)
                                    </span>
                                </div>
                                <button
                                    type="button"
                                    onClick={() => removePyFile(index)}
                                    disabled={isLoading}
                                    className="p-1 hover:bg-muted rounded transition-colors"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* PDF Upload Section (Papers to Compare) */}
            <div className="space-y-2">
                <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-foreground">
                        Papers to Compare (Optional)
                    </label>
                    <button
                        type="button"
                        onClick={triggerPdfInput}
                        disabled={isLoading}
                        className={cn(
                            "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all",
                            "bg-card border border-border hover:bg-muted disabled:opacity-50 disabled:pointer-events-none"
                        )}
                    >
                        <Upload className="w-4 h-4" />
                        <span>Add PDF</span>
                    </button>
                </div>

                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,application/pdf"
                    multiple
                    onChange={handlePdfChange}
                    className="hidden"
                />

                {pdfFiles.length > 0 && (
                    <div className="space-y-2">
                        {pdfFiles.map((file, index) => (
                            <div
                                key={index}
                                className="flex items-center justify-between p-3 rounded-lg bg-card border border-border"
                            >
                                <div className="flex items-center gap-2 flex-1 min-w-0">
                                    <span className="text-sm font-medium truncate">{file.name}</span>
                                    <span className="text-xs text-muted-foreground">
                                        ({(file.size / 1024).toFixed(0)} KB)
                                    </span>
                                </div>
                                <button
                                    type="button"
                                    onClick={() => removePdfFile(index)}
                                    disabled={isLoading}
                                    className="p-1 hover:bg-muted rounded transition-colors"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Algorithm Description (Optional) */}
            {pdfFiles.length > 0 && (
                <div className="space-y-2">
                    <label className="text-sm font-medium text-foreground">
                        Algorithm Instructions (Optional)
                    </label>
                    <textarea
                        value={algorithmDescription}
                        onChange={(e) => setAlgorithmDescription(e.target.value)}
                        placeholder="e.g., 'Implement the greedy algorithm from Section 3 of the first paper'"
                        className={cn(
                            "w-full min-h-[80px] p-3 rounded-lg bg-card border border-border text-foreground text-sm",
                            "placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all",
                            "resize-none"
                        )}
                        disabled={isLoading}
                    />
                </div>
            )}

            {/* Submit Button */}
            <div className="flex flex-col items-end gap-2">
                {pyFiles.length === 0 && pdfFiles.length === 0 && (
                    <p className="text-xs text-muted-foreground">
                        Upload your .py file and/or papers to compare
                    </p>
                )}
                <button
                    type="submit"
                    disabled={!query.trim() || (pyFiles.length === 0 && pdfFiles.length === 0) || isLoading}
                    className={cn(
                        "flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all shadow-sm",
                        "bg-primary text-primary-foreground hover:opacity-90 active:scale-95 disabled:opacity-50 disabled:pointer-events-none"
                    )}
                >
                    {isLoading ? (
                        <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            <span>Running Benchmark...</span>
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
