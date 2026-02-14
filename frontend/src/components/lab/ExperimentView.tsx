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

        // Mock Simulation
        addLog("Analyzing request...", "info");
        await delay(1000);
        addLog(`Identified goal: Benchmark ${query}`, "success");
        await delay(800);
        addLog("Searching for algorithm implementations...", "info");
        await delay(1200);
        addLog("Generating Python benchmark harness...", "info");
        await delay(1000);
        addLog("Sending code to Modal sandbox...", "info");

        // Simulate running
        for (let n of [10, 100, 1000, 10000]) {
            await delay(500);
            addLog(`Running benchmark for N=${n}...`, "info");
        }

        addLog("Processing results...", "success");
        await delay(500);

        // Mock Results
        setResults({
            title: "Dijkstra vs A* Performance",
            xLabel: "Grid Size (N)",
            yLabel: "Time (ms)",
            series: [
                { name: "Dijkstra", color: "#ef4444", dataKey: "dijkstra" },
                { name: "A*", color: "#3b82f6", dataKey: "astar" }
            ],
            data: [
                { x: 10, dijkstra: 0.5, astar: 0.2 },
                { x: 100, dijkstra: 5.2, astar: 1.1 },
                { x: 1000, dijkstra: 45.0, astar: 12.5 },
                { x: 10000, dijkstra: 420.0, astar: 110.0 },
            ]
        });
        setStatus("complete");
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
