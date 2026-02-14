"use client";

import { useState } from "react";
import { ExperimentForm } from "./ExperimentForm";
import { LiveTerminal } from "./LiveTerminal";
import { ResultsChart } from "./ResultsChart";

// Mock Data Types
interface LogEntry {
    timestamp: string;
    message: string;
    type: "info" | "error" | "success";
}

export function ExperimentView() {
    const [status, setStatus] = useState<"idle" | "running" | "complete">("idle");
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [results, setResults] = useState<any>(null); // To be typed later

    const addLog = (message: string, type: "info" | "error" | "success" = "info") => {
        setLogs(prev => [...prev, {
            timestamp: new Date().toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: '2-digit' }),
            message,
            type
        }]);
    };

    const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    const handleRun = async (query: string) => {
        setStatus("running");
        setLogs([]);
        setResults(null);

        // Clear existing logs
        addLog("Analyzing request...", "info");

        try {
            const response = await fetch("/api/python/benchmark", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ query }),
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const data = await response.json();

            // Transform backend response to match chart needs if necessary, 
            // but our backend logic aligns with the component props.
            setResults(data);
            setStatus("complete");
            addLog("Benchmark complete!", "success");

        } catch (error) {
            console.error(error);
            addLog(error instanceof Error ? error.message : "An error occurred", "error");
            setStatus("idle"); // or 'error' state if we had one
        }
    };

    return (
        <div className="space-y-8 pb-20">
            <div className="space-y-2">
                <h2 className="text-2xl font-bold tracking-tight">The Lab</h2>
                <p className="text-muted-foreground">
                    Describe the algorithm comparison you want to run.
                </p>
            </div>

            <ExperimentForm onSubmit={handleRun} isLoading={status === "running"} />

            <LiveTerminal
                logs={logs}
                isVisible={status !== "idle"}
            />

            {status === "complete" && results && (
                <ResultsChart
                    title={results.title}
                    xLabel={results.xLabel}
                    yLabel={results.yLabel}
                    series={results.series}
                    data={results.data}
                />
            )}
        </div>
    );
}
