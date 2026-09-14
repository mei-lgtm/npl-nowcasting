"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Point = {
  period: string;
  value: number | null;
  observation_type: string;
  low?: number | null;
  high?: number | null;
  actual?: number;
};

export function NowcastChart({ data }: { data: Point[] }) {
  // Build unified chart rows
  const periods = Array.from(new Set(data.map((d) => d.period))).sort();
  const byPeriod: Record<string, any> = {};
  for (const p of periods) byPeriod[p] = { period: p };
  for (const d of data) {
    const row = byPeriod[d.period];
    if (d.observation_type === "actual") row.actual = d.value;
    if (d.observation_type === "nowcast") {
      row.nowcast = d.value;
      row.low = d.low;
      row.high = d.high;
    }
    if (d.observation_type === "forecast") row.forecast = d.value;
    if (d.observation_type === "nowcast_eval") row.nowcast_eval = d.value;
  }
  const rows = periods.map((p) => byPeriod[p]).slice(-36);

  return (
    <div style={{ width: "100%", height: 320 }}>
      <ResponsiveContainer>
        <LineChart data={rows} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="rgba(148,175,198,0.12)" strokeDasharray="3 3" />
          <XAxis dataKey="period" tick={{ fill: "#8fa3b8", fontSize: 11 }} minTickGap={24} />
          <YAxis tick={{ fill: "#8fa3b8", fontSize: 11 }} domain={["auto", "auto"]} width={42} />
          <Tooltip
            contentStyle={{
              background: "#141e2a",
              border: "1px solid rgba(148,175,198,0.2)",
              borderRadius: 10,
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="actual"
            name="Actual NPL"
            stroke="#e8eef5"
            strokeWidth={2.2}
            dot={false}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="nowcast_eval"
            name="Historical Nowcast"
            stroke="#2ec4b6"
            strokeWidth={1.6}
            strokeDasharray="4 3"
            dot={false}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="nowcast"
            name="Current Nowcast"
            stroke="#f0a202"
            strokeWidth={2.4}
            dot={{ r: 4 }}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="forecast"
            name="Forecast"
            stroke="#8fa3b8"
            strokeWidth={1.5}
            strokeDasharray="2 4"
            dot={false}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function StressChart({
  data,
}: {
  data: Array<{ period: string; news_stress: number }>;
}) {
  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="stressFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#e4572e" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#e4572e" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(148,175,198,0.12)" strokeDasharray="3 3" />
          <XAxis dataKey="period" tick={{ fill: "#8fa3b8", fontSize: 11 }} minTickGap={28} />
          <YAxis domain={[-1, 1]} tick={{ fill: "#8fa3b8", fontSize: 11 }} width={36} />
          <Tooltip
            contentStyle={{
              background: "#141e2a",
              border: "1px solid rgba(148,175,198,0.2)",
              borderRadius: 10,
            }}
          />
          <Area
            type="monotone"
            dataKey="news_stress"
            name="News Stress Index"
            stroke="#e4572e"
            fill="url(#stressFill)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
