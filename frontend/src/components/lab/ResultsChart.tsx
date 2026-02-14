"use client";

import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from "recharts";

interface DataPoint {
    x: number;
    [key: string]: number | string;
}

interface Series {
    name: string;
    color: string;
    dataKey: string;
}

interface ResultsChartProps {
    data: DataPoint[];
    series: Series[];
    xLabel: string;
    yLabel: string;
    title: string;
}

export function ResultsChart({ data, series, xLabel, yLabel, title }: ResultsChartProps) {
    return (
        <div className="w-full bg-card border border-border rounded-xl p-6 shadow-sm">
            <div className="mb-6">
                <h3 className="text-lg font-semibold">{title}</h3>
            </div>

            <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                        data={data}
                        margin={{
                            top: 5,
                            right: 30,
                            left: 20,
                            bottom: 20,
                        }}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.1} />
                        <XAxis
                            dataKey="x"
                            label={{ value: xLabel, position: 'insideBottomRight', offset: -10 }}
                            tick={{ fill: 'currentColor' }}
                        />
                        <YAxis
                            label={{ value: yLabel, angle: -90, position: 'insideLeft' }}
                            tick={{ fill: 'currentColor' }}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: 'var(--card)', borderColor: 'var(--border)', borderRadius: '8px' }}
                            itemStyle={{ color: 'var(--foreground)' }}
                        />
                        <Legend verticalAlign="top" height={36} />
                        {series.map((s) => (
                            <Line
                                key={s.name}
                                type="monotone"
                                dataKey={s.dataKey}
                                name={s.name}
                                stroke={s.color}
                                strokeWidth={2}
                                dot={{ r: 4 }}
                                activeDot={{ r: 8 }}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
