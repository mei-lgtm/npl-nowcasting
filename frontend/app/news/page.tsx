"use client";

import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";
import { StressChart } from "@/components/Charts";
import { DemoBanner, Panel } from "@/components/ui";

export default function NewsPage() {
  const [news, setNews] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<any>("/api/news")
      .then((r) => setNews(r.data))
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <Panel title="Error">{error}</Panel>;
  if (!news) return <div className="muted"><span className="spinner" /> Loading news…</div>;

  const maxStress = Math.max(...(news.sector_heatmap?.map((s: any) => Math.abs(s.stress)) || [1]), 0.01);

  return (
    <div className="stack" style={{ gap: "1rem" }}>
      <div className="topbar">
        <div>
          <h1 className="page-title">News Sentiment</h1>
          <p className="page-sub">
            Distress-oriented sentiment for credit quality, financial stress, and banking risk.
            Index scale: −1 soft / +1 high stress.
          </p>
        </div>
        <DemoBanner show={!!news.is_synthetic} />
      </div>

      <div className="grid grid-4">
        <Panel title="News Stress Index">
          <div className="metric">{news.current.toFixed(2)}</div>
        </Panel>
        <Panel title="Previous Month">
          <div className="metric">{news.previous.toFixed(2)}</div>
        </Panel>
        <Panel title="Change">
          <div className="metric">{news.change >= 0 ? "+" : ""}{news.change.toFixed(2)}</div>
        </Panel>
        <Panel title="Negative Share">
          <div className="metric">{(news.negative_ratio * 100).toFixed(0)}%</div>
          <div className="muted" style={{ marginTop: "0.4rem" }}>
            {news.article_count} relevant articles (30d)
          </div>
        </Panel>
      </div>

      <Panel title="News Stress Index — Monthly">
        <StressChart data={news.series} />
      </Panel>

      <Panel title="Sector Credit Stress">
        <div className="heatmap">
          {news.sector_heatmap.map((s: any) => {
            const intensity = Math.min(Math.abs(s.stress) / maxStress, 1);
            const bg =
              s.stress >= 0
                ? `rgba(228, 87, 46, ${0.08 + intensity * 0.35})`
                : `rgba(46, 196, 182, ${0.08 + intensity * 0.3})`;
            return (
              <div className="heat-cell" key={s.sector} style={{ background: bg }}>
                <strong>{s.sector}</strong>
                <span className="metric-sm">{s.stress.toFixed(2)}</span>
              </div>
            );
          })}
        </div>
      </Panel>

      <Panel title="Recent Articles">
        {news.recent_articles.map((a: any) => (
          <div className="article" key={a.id}>
            <div className="row">
              <strong style={{ fontSize: "0.95rem" }}>{a.headline}</strong>
              <span className="chip">{a.sentiment}</span>
            </div>
            <div className="muted" style={{ fontSize: "0.8rem", marginTop: "0.25rem" }}>
              {a.source} · {a.sector} · relevance {(a.npl_relevance * 100).toFixed(0)}% ·{" "}
              {String(a.published_at).slice(0, 10)}
            </div>
          </div>
        ))}
      </Panel>
    </div>
  );
}
