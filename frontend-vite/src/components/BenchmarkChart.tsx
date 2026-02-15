import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    LineChart,
    Line
} from "recharts"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"

interface BenchmarkChartProps {
    data: any[]
    type: "bar" | "line"
    title: string
    xAxisKey: string
    lines: { key: string, color: string }[]
}

export function BenchmarkChart({ data, type, title, xAxisKey, lines }: BenchmarkChartProps) {
    if (!data || data.length === 0) return null

    return (
        <Card>
            <CardHeader>
                <CardTitle>{title}</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="h-[300px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        {type === "bar" ? (
                            <BarChart data={data}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey={xAxisKey} />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                {lines.map((l) => (
                                    <Bar key={l.key} dataKey={l.key} fill={l.color} />
                                ))}
                            </BarChart>
                        ) : (
                            <LineChart data={data}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey={xAxisKey} />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                {lines.map((l) => (
                                    <Line key={l.key} type="monotone" dataKey={l.key} stroke={l.color} />
                                ))}
                            </LineChart>
                        )}
                    </ResponsiveContainer>
                </div>
            </CardContent>
        </Card>
    )
}
