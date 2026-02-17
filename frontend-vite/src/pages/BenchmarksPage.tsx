import { useState } from "react"
import { Layout } from "@/components/Layout"
import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"

import { BenchmarkChart } from "@/components/BenchmarkChart"
import { Play } from "lucide-react"

export default function BenchmarksPage() {
    const [isRunning, setIsRunning] = useState(false)
    const [results, setResults] = useState<any[]>([])

    // Placeholder data
    const dummyData = [
        { name: "Max Cut 50", algo1: 45, algo2: 48, algo3: 42 },
        { name: "Max Cut 100", algo1: 90, algo2: 95, algo3: 88 },
        { name: "Max Cut 200", algo1: 180, algo2: 192, algo3: 175 },
    ]

    const handleRunBenchmark = async () => {
        setIsRunning(true)
        // Simulate run
        await new Promise(r => setTimeout(r, 2000))
        setResults(dummyData)
        setIsRunning(false)
    }

    return (
        <Layout>
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-2xl font-bold tracking-tight">Benchmark Dashboard</h2>
                        <p className="text-muted-foreground">View and compare algorithm performance.</p>
                    </div>
                    <Button onClick={handleRunBenchmark} disabled={isRunning}>
                        <Play className="mr-2 h-4 w-4" />
                        {isRunning ? "Running..." : "Run Standard Benchmark"}
                    </Button>
                </div>

                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Total Runs</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">12</div>
                            <p className="text-xs text-muted-foreground">+2 from last hour</p>
                        </CardContent>
                    </Card>
                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Best Algorithm</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">SDP Rounding</div>
                            <p className="text-xs text-muted-foreground">Accuracy 98.5%</p>
                        </CardContent>
                    </Card>
                </div>

                {results.length > 0 && (
                    <div className="grid gap-4">
                        <BenchmarkChart
                            data={results}
                            type="bar"
                            title="Max Cut Performance (Cut Size)"
                            xAxisKey="name"
                            lines={[{ key: "algo1", color: "#8884d8" }, { key: "algo2", color: "#82ca9d" }, { key: "algo3", color: "#ffc658" }]}
                        />
                    </div>
                )}
            </div>
        </Layout>
    )
}
