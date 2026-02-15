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

interface Series {
  name: string;
  color: string;
  dataKey: string;
}

interface ChartBlockProps {
  title: string;
  xLabel: string;
  yLabel: string;
  series: Series[];
  data: { x: number; [key: string]: number | string }[];
}

export function ChartBlock({
  title,
  xLabel,
  yLabel,
  series,
  data,
}: ChartBlockProps) {
  return (
    <div className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-5 mt-3">
      <h4 className="text-sm font-semibold text-foreground mb-4">{title}</h4>
      <div className="h-[320px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 4, right: 24, left: 8, bottom: 16 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#27272a"
              opacity={0.5}
            />
            <XAxis
              dataKey="x"
              label={{
                value: xLabel,
                position: "insideBottomRight",
                offset: -8,
                fill: "#71717a",
                fontSize: 11,
              }}
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              stroke="#27272a"
            />
            <YAxis
              label={{
                value: yLabel,
                angle: -90,
                position: "insideLeft",
                fill: "#71717a",
                fontSize: 11,
              }}
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              stroke="#27272a"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0f0f12",
                borderColor: "#27272a",
                borderRadius: "8px",
                fontSize: 12,
              }}
              itemStyle={{ color: "#fafafa" }}
              labelStyle={{ color: "#a1a1aa" }}
            />
            <Legend
              verticalAlign="top"
              height={32}
              wrapperStyle={{ fontSize: 12 }}
            />
            {series.map((s) => (
              <Line
                key={s.name}
                type="monotone"
                dataKey={s.dataKey}
                name={s.name}
                stroke={s.color}
                strokeWidth={2}
                dot={{ r: 3, strokeWidth: 0, fill: s.color }}
                activeDot={{ r: 5 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
