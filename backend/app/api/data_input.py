"""Data input endpoints: multi-format indicator upload + Google News."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import quote_plus

import httpx
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

router = APIRouter(prefix="/data", tags=["data-input"])

RECOMMENDED_PATTERNS = [
    (r"npl|non.?performing|gross_npl|net_npl", "Target / kualitas kredit langsung", 100),
    (r"credit_growth|loan_growth|pertumbuhan.?kredit", "Pertumbuhan kredit — driver klasik NPL", 95),
    (r"lending_rate|suku.?bunga.?kredit|loan_rate", "Suku bunga kredit — biaya pinjaman", 92),
    (r"policy_rate|bi.?rate|suku.?bunga.?kebijakan", "Suku bunga kebijakan", 90),
    (r"gdp|pdb|growth", "Aktivitas ekonomi riil", 88),
    (r"unemployment|pengangguran", "Kondisi tenaga kerja", 86),
    (r"inflation|inflasi|cpi", "Tekanan harga", 84),
    (r"exchange|usdidr|fx|nilai.?tukar", "Nilai tukar / volatilitas eksternal", 82),
    (r"sml|special.?mention|lar|loan.?at.?risk|restructur", "Leading indicator kualitas kredit", 91),
    (r"pmi|industrial|produksi|retail|cement|electricity|penjualan", "High-frequency activity proxy", 78),
    (r"ihsg|bond|spread|vix|interbank", "Kondisi pasar keuangan", 76),
    (r"car|roa|roe|ldr|provision|cost.?of.?credit", "Indikator ketahanan perbankan", 80),
    (r"consumer.?conf|business.?conf|confidence", "Kepercayaan pelaku ekonomi", 72),
    (r"export|import|oil|commodity|property", "Sektor dan harga komoditas", 68),
    (r"news.?stress|sentiment|sentimen", "Sentimen berita", 85),
]


class GoogleNewsRequest(BaseModel):
    query: str = Field(
        default='NPL OR "kredit bermasalah" OR "non performing loan" OR "kualitas kredit" Indonesia',
    )
    keywords: Optional[List[str]] = Field(
        default=None,
        description="Jika diisi, setiap kata kunci diambil terpisah lalu digabung (lebih lengkap).",
    )
    language: str = "id"
    country: str = "ID"
    # Google News RSS when: operator, e.g. 1d / 7d / 30d / 90d; kosong = tanpa batas waktu
    time_range: Optional[str] = Field(default="90d", description="Rentang waktu berita (when:Xd).")
    # Rentang kalender kustom (YYYY-MM-DD); jika diisi, mengabaikan time_range preset
    date_from: Optional[str] = Field(default=None, description="Tanggal mulai (YYYY-MM-DD).")
    date_to: Optional[str] = Field(default=None, description="Tanggal akhir (YYYY-MM-DD).")
    # None / omitted = ambil semua item yang dikembalikan RSS (tanpa batas artifisial)
    max_articles: Optional[int] = Field(default=None, ge=1, le=8000)
    # auto: single-shot bila rentang pendek; monthly windows bila rentang panjang
    strategy: Optional[str] = Field(
        default="auto",
        description="auto | single | monthly — monthly memecah rentang per bulan (rekomendasi untuk multi-tahun).",
    )


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _parse_ymd(val: Optional[str]) -> Optional[date]:
    if not val:
        return None
    try:
        return datetime.strptime(val.strip()[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _month_windows(lo: date, hi: date) -> List[Tuple[date, date]]:
    """Pecah [lo, hi] menjadi jendela bulanan agar RSS Google tidak terjebak ~100 item total."""
    if hi < lo:
        lo, hi = hi, lo
    windows: List[Tuple[date, date]] = []
    cur = date(lo.year, lo.month, 1)
    while cur <= hi:
        last = monthrange(cur.year, cur.month)[1]
        w_lo = max(cur, lo)
        w_hi = min(date(cur.year, cur.month, last), hi)
        if w_lo <= w_hi:
            windows.append((w_lo, w_hi))
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return windows


def _parse_rss(xml_text: str, max_articles: Optional[int] = None) -> List[dict]:
    root = ET.fromstring(xml_text)
    items: List[dict] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source = (item.findtext("source") or "").strip()
        pub = item.findtext("pubDate") or ""
        desc_raw = item.findtext("description") or ""
        desc = _strip_html(desc_raw)
        image = None
        m_img = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', desc_raw, re.I)
        if m_img:
            image = m_img.group(1)
        else:
            media = item.find("{http://search.yahoo.com/mrss/}content")
            if media is not None and media.get("url"):
                image = media.get("url")
        published_at = None
        if pub:
            try:
                published_at = parsedate_to_datetime(pub).astimezone(timezone.utc).isoformat()
            except Exception:
                published_at = pub
        if not title:
            continue
        items.append(
            {
                "headline": title,
                "url": link,
                "source": source or "Google News",
                "published_at": published_at,
                "snippet": desc[:400] if desc else None,
                "image": image,
                "provider": "google_news",
            }
        )
        if max_articles is not None and len(items) >= max_articles:
            break
    return items


def _article_key(a: dict) -> str:
    url = (a.get("url") or "").strip().lower()
    if url:
        return "u:" + url
    return "h:" + (a.get("headline") or "").strip().lower()


async def _fetch_one_query(
    client: httpx.AsyncClient,
    query: str,
    language: str,
    country: str,
    max_articles: Optional[int],
) -> List[dict]:
    q = quote_plus(query)
    url = (
        f"https://news.google.com/rss/search?q={q}"
        f"&hl={language}&gl={country}&ceid={country}:{language}"
    )
    resp = await client.get(
        url,
        headers={"User-Agent": "NPL-Nowcasting/1.0 (research; +local)"},
    )
    resp.raise_for_status()
    return _parse_rss(resp.text, max_articles)


def _base_keyword_queries(tags: List[str], default_query: str) -> List[str]:
    GENERAL_QUERIES = [
        "Indonesia ekonomi",
        "Indonesia perbankan",
        "kredit bank Indonesia",
        "Bank Indonesia",
        "inflasi Indonesia",
        "suku bunga Indonesia",
        "NPL bank",
        "rupiah pasar keuangan",
    ]
    bases: list[str] = []
    if tags:
        for t in tags:
            base = f'"{t}"' if (" " in t and not t.startswith('"')) else t
            bases.append(base)
    else:
        bases.extend(GENERAL_QUERIES)
        dq = (default_query or "").strip()
        if dq and dq not in bases and "when:" not in dq and "after:" not in dq:
            bases.insert(0, dq)
    # unik jaga urutan
    seen: set[str] = set()
    out: list[str] = []
    for b in bases:
        if b in seen:
            continue
        seen.add(b)
        out.append(b)
    return out


def _article_month_key(a: dict) -> Optional[str]:
    pub = a.get("published_at")
    if not pub:
        return None
    try:
        dt = datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
        return f"{dt.year:04d}-{dt.month:02d}"
    except Exception:
        s = str(pub)
        m = re.match(r"^(\d{4})-(\d{2})", s)
        return f"{m.group(1)}-{m.group(2)}" if m else None


def _balance_across_months(articles: List[dict], hard_cap: int) -> Tuple[List[dict], dict[str, int]]:
    """Ambil sampel merata antar bulan agar tiap periode terwakili sebelum mengisi sisa kuota."""
    if hard_cap <= 0 or len(articles) <= hard_cap:
        cov: dict[str, int] = {}
        for a in articles:
            mk = _article_month_key(a) or "unknown"
            cov[mk] = cov.get(mk, 0) + 1
        return articles, dict(sorted(cov.items()))

    buckets: dict[str, list] = {}
    for a in articles:
        mk = _article_month_key(a) or "unknown"
        buckets.setdefault(mk, []).append(a)
    for mk in buckets:
        buckets[mk].sort(key=lambda x: x.get("published_at") or "", reverse=True)

    months = sorted(k for k in buckets.keys() if k != "unknown")
    if "unknown" in buckets:
        months.append("unknown")
    if not months:
        return articles[:hard_cap], {}

    # Fase 1: jatah minimal merata
    base = max(1, hard_cap // len(months))
    picked: list[dict] = []
    used: dict[str, int] = {m: 0 for m in months}
    for m in months:
        take = min(base, len(buckets[m]), hard_cap - len(picked))
        if take <= 0:
            break
        picked.extend(buckets[m][:take])
        used[m] = take

    # Fase 2: round-robin sisa kuota dari bulan yang masih punya artikel
    if len(picked) < hard_cap:
        progressed = True
        while len(picked) < hard_cap and progressed:
            progressed = False
            for m in months:
                if len(picked) >= hard_cap:
                    break
                idx = used[m]
                if idx < len(buckets[m]):
                    picked.append(buckets[m][idx])
                    used[m] = idx + 1
                    progressed = True

    picked.sort(key=lambda x: x.get("published_at") or "", reverse=True)
    coverage = {m: used[m] for m in months if used[m] > 0}
    return picked, dict(sorted(coverage.items()))


@router.post("/google-news")
async def fetch_google_news(req: GoogleNewsRequest):
    """Ambil berita Google News RSS dengan strategi terbaik untuk rentang panjang.

    Google News RSS ≈100 item/query tanpa pagination. Untuk rentang multi-bulan/tahun,
    rentang dipecah per bulan × kata kunci, dengan kuota per bulan agar hasil tersebar
    merata di seluruh periode (bukan hanya menumpuk di bulan terbaru).
    """
    tags = [t.strip() for t in (req.keywords or []) if t and str(t).strip()]
    date_from = (req.date_from or "").strip() or None
    date_to = (req.date_to or "").strip() or None
    when = (req.time_range or "").strip().lower()
    if date_from or date_to:
        when = ""
    elif when in ("", "all", "semua", "*"):
        when = ""
    elif not re.fullmatch(r"\d+[hdwmy]", when):
        when = "7d"

    lo = _parse_ymd(date_from)
    hi = _parse_ymd(date_to) or datetime.now(timezone.utc).date()
    if lo and not date_to:
        hi = datetime.now(timezone.utc).date()
    if hi and not lo:
        lo = hi - timedelta(days=89)

    span_days = (hi - lo).days + 1 if (lo and hi) else None
    strategy = (req.strategy or "auto").strip().lower()
    if strategy not in ("auto", "single", "monthly"):
        strategy = "auto"
    use_monthly = strategy == "monthly" or (
        strategy == "auto" and span_days is not None and span_days > 45 and bool(lo and hi)
    )

    bases = _base_keyword_queries(tags, req.query)
    hard_cap = req.max_articles or (5000 if use_monthly else 800)

    def _decorate(base: str, w_lo: Optional[date], w_hi: Optional[date], when_op: str) -> str:
        q = base
        if when_op:
            q = f"{q} when:{when_op}"
        if w_lo:
            q = f"{q} after:{w_lo.isoformat()}"
        if w_hi:
            before = (w_hi + timedelta(days=1)).isoformat()
            q = f"{q} before:{before}"
        return q

    def _parse_ts(pub: str):
        try:
            return datetime.fromisoformat(str(pub).replace("Z", "+00:00")).timestamp()
        except Exception:
            return None

    def _in_range(a: dict, w_lo: Optional[date], w_hi: Optional[date]) -> bool:
        if not (w_lo or w_hi):
            return True
        pub = a.get("published_at")
        if not pub:
            return False
        ts = _parse_ts(pub)
        if ts is None:
            return False
        if w_lo is not None:
            lo_ts = datetime(w_lo.year, w_lo.month, w_lo.day, tzinfo=timezone.utc).timestamp()
            if ts < lo_ts:
                return False
        if w_hi is not None:
            hi_ts = datetime(w_hi.year, w_hi.month, w_hi.day, 23, 59, 59, tzinfo=timezone.utc).timestamp()
            if ts > hi_ts:
                return False
        return True

    errors: list[str] = []
    queries_run = 0
    windows_meta: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(4)

    async def _run_job(client: httpx.AsyncClient, q: str) -> Tuple[str, List[dict], Optional[str]]:
        async with sem:
            try:
                batch = await _fetch_one_query(client, q, req.language, req.country, None)
                return q, batch, None
            except httpx.HTTPError as e:
                return q, [], str(e)

    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    used_strategy = "single"

    try:
        timeout = httpx.Timeout(45.0, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            if use_monthly and lo and hi:
                used_strategy = "monthly_balanced"
                windows = _month_windows(lo, hi)
                n_win = max(1, len(windows))
                # Kuota target per bulan + buffer fetch agar ada pilihan saat balancing
                month_quota = max(20, hard_cap // n_win)
                month_fetch_cap = max(month_quota * 2, month_quota + 15)

                for w_lo, w_hi in windows:
                    month_jobs = [_decorate(base, w_lo, w_hi, "") for base in bases]
                    queries_run += len(month_jobs)
                    month_arts: list[dict] = []
                    month_seen: set[str] = set()
                    # fetch semua keyword bulan ini
                    for i in range(0, len(month_jobs), 8):
                        part = month_jobs[i : i + 8]
                        results = await asyncio.gather(*[_run_job(client, q) for q in part])
                        for q, batch, err in results:
                            if err:
                                errors.append(f"{q[:80]}: {err}")
                            for a in batch:
                                if not _in_range(a, w_lo, w_hi):
                                    continue
                                k = _article_key(a)
                                if k in month_seen or k in seen:
                                    continue
                                month_seen.add(k)
                                month_arts.append({**a, "matched_query": q, "window": f"{w_lo.isoformat()}:{w_hi.isoformat()}"})
                        if len(month_arts) >= month_fetch_cap:
                            break
                    # simpan hingga fetch_cap; balancing global nanti
                    month_arts.sort(key=lambda x: x.get("published_at") or "", reverse=True)
                    kept = month_arts[:month_fetch_cap]
                    for a in kept:
                        k = _article_key(a)
                        if k in seen:
                            continue
                        seen.add(k)
                        pool.append(a)
                    windows_meta.append(
                        {
                            "from": w_lo.isoformat(),
                            "to": w_hi.isoformat(),
                            "fetched": len(month_arts),
                            "kept": len(kept),
                            "quota_target": month_quota,
                        }
                    )
                    await asyncio.sleep(0.08)
            else:
                used_strategy = "single"
                jobs = [
                    _decorate(
                        base,
                        lo if (date_from or date_to) else None,
                        hi if (date_from or date_to) else None,
                        when,
                    )
                    for base in bases
                ]
                queries_run = len(jobs)
                if lo and hi:
                    windows_meta = [{"from": lo.isoformat(), "to": hi.isoformat()}]
                for i in range(0, len(jobs), 24):
                    part = jobs[i : i + 24]
                    results = await asyncio.gather(*[_run_job(client, q) for q in part])
                    for q, batch, err in results:
                        if err:
                            errors.append(f"{q[:80]}: {err}")
                        for a in batch:
                            k = _article_key(a)
                            if k in seen:
                                continue
                            if (date_from or date_to) and not _in_range(a, lo, hi):
                                continue
                            seen.add(k)
                            pool.append({**a, "matched_query": q})
                    if len(pool) >= hard_cap * 2:
                        break
                    if i + 24 < len(jobs):
                        await asyncio.sleep(0.15)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Gagal mengambil Google News: {e}") from e

    if not pool and errors:
        raise HTTPException(status_code=502, detail="Gagal mengambil Google News: " + "; ".join(errors[:3]))

    # Filter global + when: untuk mode single
    articles = pool
    if used_strategy == "single" and when and not (date_from or date_to):
        m = re.fullmatch(r"(\d+)([hdwmy])", when)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            mult = {"h": 3600, "d": 86400, "w": 604800, "m": 2592000, "y": 31536000}[unit]
            cutoff = datetime.now(timezone.utc).timestamp() - n * mult
            filtered = []
            for a in articles:
                pub = a.get("published_at")
                if not pub:
                    filtered.append(a)
                    continue
                ts = _parse_ts(pub)
                if ts is None or ts >= cutoff:
                    filtered.append(a)
            articles = filtered

    if used_strategy.startswith("monthly"):
        articles, coverage = _balance_across_months(articles, hard_cap)
    else:
        articles.sort(key=lambda x: x.get("published_at") or "", reverse=True)
        articles = articles[:hard_cap]
        coverage = {}
        for a in articles:
            mk = _article_month_key(a) or "unknown"
            coverage[mk] = coverage.get(mk, 0) + 1
        coverage = dict(sorted(coverage.items()))

    months_with = len([k for k, v in coverage.items() if k != "unknown" and v > 0])
    months_empty = 0
    if used_strategy.startswith("monthly") and windows_meta:
        months_empty = sum(1 for w in windows_meta if int(w.get("kept") or 0) == 0)

    range_label = (
        f"{(lo.isoformat() if lo else date_from) or '…'} → {(hi.isoformat() if hi else date_to) or '…'}"
        if (date_from or date_to or lo)
        else (when or "all")
    )

    return {
        "data": {
            "query": " | ".join(bases[:8]) + ("…" if len(bases) > 8 else ""),
            "keywords": tags or None,
            "time_range": range_label,
            "count": len(articles),
            "articles": articles,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "Google News RSS",
            "strategy": used_strategy,
            "windows": len(windows_meta),
            "window_span_days": span_days,
            "queries_run": queries_run,
            "period_coverage": coverage,
            "months_covered": months_with,
            "months_empty": months_empty or None,
            "partial_errors": errors[:12] or None,
            "error_count": len(errors) or None,
            "capped_at": hard_cap if len(articles) >= hard_cap else None,
        },
        "is_synthetic": False,
        "message": (
            f"Berita diambil via Google News RSS · strategi {used_strategy}"
            f" ({queries_run} query, {len(windows_meta)} jendela, {len(articles)} unik,"
            f" {months_with} bulan terisi)."
        ),
    }


def _norm_period(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, datetime):
        return val.strftime("%Y-%m")
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m")
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    # 2024-01-0100:00:00 or 2024-01-01 00:00:00
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/]?(\d{0,2})", s.replace(" ", ""))
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = re.match(r"^(\d{4})[-/](\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%Y-%m")
    except Exception:
        pass
    return s[:16]


def _to_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", "")
    if not s or s.lower() in ("nan", "none", "-", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _slug(name: str) -> str:
    s = re.sub(r"[^\w]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", s).strip("_") or "indicator"


def _df_to_parsed(df: pd.DataFrame, source_name: str = "upload") -> dict[str, Any]:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # drop fully empty columns/rows
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError("Tabel kosong setelah pembersihan")

    lower = {c.lower(): c for c in df.columns}
    long_keys = {"indicator", "reference_period", "value"}
    if long_keys.issubset(set(lower.keys())):
        observations = []
        for _, r in df.iterrows():
            ind = str(r[lower["indicator"]]).strip()
            period = _norm_period(r[lower["reference_period"]])
            value = _to_float(r[lower["value"]])
            if not ind or not period or value is None:
                continue
            observations.append(
                {
                    "indicator": _slug(ind),
                    "indicator_label": ind,
                    "reference_period": period,
                    "value": value,
                    "publication_date": str(r[lower["publication_date"]]).strip()
                    if "publication_date" in lower and pd.notna(r.get(lower["publication_date"]))
                    else None,
                    "unit": str(r[lower["unit"]]).strip() if "unit" in lower else "percent",
                    "frequency": str(r[lower["frequency"]]).strip() if "frequency" in lower else "monthly",
                    "source": str(r[lower["source"]]).strip() if "source" in lower else source_name,
                }
            )
        return _finalize(observations, "long", df)

    # Wide format
    period_col = None
    for cand in ("period", "reference_period", "date", "bulan", "tanggal", "time", "month", "datetime"):
        if cand in lower:
            period_col = lower[cand]
            break
    if period_col is None:
        period_col = df.columns[0]

    observations = []
    for _, r in df.iterrows():
        period = _norm_period(r[period_col])
        if not period:
            continue
        for col in df.columns:
            if col == period_col:
                continue
            value = _to_float(r[col])
            if value is None:
                continue
            observations.append(
                {
                    "indicator": _slug(col),
                    "indicator_label": str(col).strip(),
                    "reference_period": period,
                    "value": value,
                    "publication_date": None,
                    "unit": "percent",
                    "frequency": "monthly",
                    "source": source_name,
                }
            )
    return _finalize(observations, "wide", df, period_col=period_col)


def _recommend(indicators: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for info in indicators:
        name = info["id"]
        label = info.get("label") or name
        score = 40
        reason = "Indikator umum — pertimbangkan jika relevan secara ekonomi"
        for pat, why, sc in RECOMMENDED_PATTERNS:
            if re.search(pat, name, re.I) or re.search(pat, label, re.I):
                score = sc
                reason = why
                break
        # coverage bonus
        cov = info.get("coverage", 0)
        if cov >= 24:
            score = min(100, score + 5)
        elif cov < 6:
            score = max(10, score - 15)
            reason += " · cakupan waktu pendek"
        out.append(
            {
                "id": name,
                "label": label,
                "score": score,
                "recommended": score >= 75,
                "reason": reason,
                "n_obs": info.get("n_obs", 0),
                "coverage": cov,
                "missing_pct": info.get("missing_pct", 0),
            }
        )
    out.sort(key=lambda x: (-x["score"], x["label"]))
    return out


def _outlier_count(vals: list[float]) -> int:
    if len(vals) < 8:
        return 0
    s = pd.Series(vals, dtype="float64")
    z = (s - s.mean()) / (s.std(ddof=0) + 1e-8)
    return int((z.abs() > 3).sum())


def _finalize(observations: list[dict], fmt: str, df: pd.DataFrame, period_col: str | None = None) -> dict[str, Any]:
    if not observations:
        raise ValueError("Tidak ada observasi numerik yang dapat dibaca")

    # indicator stats
    by_ind: dict[str, list] = {}
    labels: dict[str, str] = {}
    for o in observations:
        by_ind.setdefault(o["indicator"], []).append(o)
        labels[o["indicator"]] = o.get("indicator_label") or o["indicator"]

    # periods sorted
    periods = sorted({o["reference_period"] for o in observations})
    ind_ids = sorted(by_ind.keys())

    # wide matrix for neat preview
    lookup = {(o["indicator"], o["reference_period"]): o["value"] for o in observations}
    preview_rows = []
    for p in periods:
        row = {"period": p}
        for ind in ind_ids:
            v = lookup.get((ind, p))
            row[ind] = None if v is None else round(float(v), 4)
        preview_rows.append(row)

    ind_meta = []
    total_outliers = 0
    for ind in ind_ids:
        vals = by_ind[ind]
        per_set = {o["reference_period"] for o in vals}
        coverage = len(per_set)
        missing_pct = round(100 * (1 - coverage / max(len(periods), 1)), 1)
        num_vals = [float(o["value"]) for o in vals if o.get("value") is not None]
        outliers = _outlier_count(num_vals)
        total_outliers += outliers
        ind_meta.append(
            {
                "id": ind,
                "label": labels[ind],
                "n_obs": len(vals),
                "coverage": coverage,
                "missing_pct": missing_pct,
                "outlier_count": outliers,
                "first": min(per_set),
                "last": max(per_set),
                "status": "ok" if missing_pct < 10 and outliers == 0 else ("warning" if missing_pct < 25 else "attention"),
            }
        )

    recommendations = _recommend(ind_meta)
    default_selected = [r["id"] for r in recommendations if r["recommended"]]
    if not default_selected:
        default_selected = [r["id"] for r in recommendations[: min(8, len(recommendations))]]

    quality = {
        "n_series": len(ind_ids),
        "n_periods": len(periods),
        "n_observations": len(observations),
        "missing_cells": sum(1 for r in preview_rows for ind in ind_ids if r.get(ind) is None),
        "outlier_count": total_outliers,
        "avg_missing_pct": round(sum(m["missing_pct"] for m in ind_meta) / max(len(ind_meta), 1), 1),
        "period_start": periods[0] if periods else None,
        "period_end": periods[-1] if periods else None,
        "series": ind_meta,
    }

    return {
        "format": fmt,
        "n_rows": len(periods),
        "n_cols": len(ind_ids) + 1,
        "n_observations": len(observations),
        "indicators": ind_ids,
        "indicator_meta": ind_meta,
        "periods": periods,
        "observations": observations,
        "preview_rows": preview_rows,
        "preview_limit": min(12, len(preview_rows)),
        "recommendations": recommendations,
        "default_selected": default_selected,
        "raw_columns": list(df.columns),
        "period_column": period_col,
        "quality": quality,
    }


def _read_pdf_tables(raw: bytes) -> pd.DataFrame:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ValueError("Parser PDF tidak tersedia (install pypdf)") from e

    reader = PdfReader(io.BytesIO(raw))
    lines = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.splitlines():
            line = line.strip()
            if line:
                lines.append(line)
    if not lines:
        raise ValueError("PDF tidak berisi teks yang dapat diekstrak (mungkin hasil scan)")

    # Try delimiter detection on densest lines
    candidates = []
    for sep in ["\t", ",", ";", "|", "  "]:
        rows = []
        for line in lines:
            parts = [p.strip() for p in re.split(r"\t|,|;|\|" if sep != "  " else r"\s{2,}", line) if p.strip()]
            if len(parts) >= 2:
                rows.append(parts)
        if len(rows) >= 3:
            width = max(len(r) for r in rows)
            good = [r for r in rows if len(r) == width]
            if len(good) >= 3:
                candidates.append(good)

    if not candidates:
        # fallback: whitespace split
        rows = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                rows.append(parts)
        if len(rows) < 3:
            raise ValueError("Tidak menemukan tabel terstruktur di PDF")
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows if len(r) >= 2]
        header, body = rows[0], rows[1:]
    else:
        best = max(candidates, key=len)
        header, body = best[0], best[1:]

    # sanitize header
    cols = []
    for i, h in enumerate(header):
        name = h if h else f"col_{i}"
        cols.append(name)
    # unique
    seen = {}
    uniq = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            uniq.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            uniq.append(c)
    return pd.DataFrame(body, columns=uniq)


def parse_upload_bytes(filename: str, raw: bytes) -> dict[str, Any]:
    name = (filename or "upload").lower()
    source = Path(filename).stem if filename else "upload"

    if name.endswith((".csv", ".txt", ".tsv")):
        sep = "\t" if name.endswith(".tsv") else ","
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        df = pd.read_csv(io.StringIO(text.lstrip("\ufeff")), sep=sep)
        return _df_to_parsed(df, source_name=source)

    if name.endswith((".xlsx", ".xlsm")):
        df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        return _df_to_parsed(df, source_name=source)

    if name.endswith(".xls"):
        try:
            df = pd.read_excel(io.BytesIO(raw), engine="xlrd")
        except Exception:
            df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        return _df_to_parsed(df, source_name=source)

    if name.endswith(".json"):
        payload = json.loads(raw.decode("utf-8"))
        # Already-normalized upload payload (from prior save / API)
        if isinstance(payload, dict) and isinstance(payload.get("observations"), list):
            obs = payload["observations"]
            ind_ids = payload.get("indicators") or sorted(
                {o.get("indicator") for o in obs if o.get("indicator")}
            )
            # Rebuild a minimal DataFrame for the standard pipeline
            rows = []
            for o in obs:
                rows.append(
                    {
                        "indicator": o.get("indicator"),
                        "reference_period": o.get("reference_period") or o.get("period"),
                        "value": o.get("value"),
                    }
                )
            if not rows:
                raise ValueError("JSON tidak berisi observasi")
            df = pd.DataFrame(rows)
            return _df_to_parsed(df, source_name=source)
        if isinstance(payload, list):
            df = pd.DataFrame(payload)
        elif isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
            df = pd.DataFrame(payload["data"])
        elif isinstance(payload, dict):
            # dict-of-lists / wide object — require equal lengths
            try:
                df = pd.DataFrame(payload)
            except ValueError as e:
                raise ValueError(
                    "JSON tidak bisa dibaca sebagai tabel. "
                    "Gunakan array objek, format panjang (indicator/period/value), "
                    "atau CSV/Excel wide."
                ) from e
        else:
            raise ValueError("Struktur JSON tidak dikenali")
        return _df_to_parsed(df, source_name=source)

    if name.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(raw))
        return _df_to_parsed(df, source_name=source)

    if name.endswith(".pdf"):
        df = _read_pdf_tables(raw)
        return _df_to_parsed(df, source_name=source)

    # Try CSV sniff as last resort
    try:
        text = raw.decode("utf-8")
        df = pd.read_csv(io.StringIO(text.lstrip("\ufeff")))
        return _df_to_parsed(df, source_name=source)
    except Exception as e:
        raise ValueError(
            f"Format '{name}' tidak didukung atau tidak dapat dibaca. "
            "Gunakan CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet, atau PDF berisi tabel teks."
        ) from e


@router.post("/upload-indicators")
async def upload_indicators(
    file: UploadFile = File(...),
    replace: str = Form(default="false"),
):
    """Unggah indikator: CSV / Excel / PDF / JSON / Parquet / TSV."""
    name = file.filename or "upload.bin"
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "File kosong")

    try:
        parsed = parse_upload_bytes(name, raw)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(400, f"Gagal membaca file: {e}") from e

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    saved_as = f"indicators_{stamp}.json"
    # Persist best-effort: Vercel/serverless only allows /tmp writes
    try:
        candidates = [
            Path("/tmp") / "npl-uploads",
            Path(__file__).resolve().parents[2] / "data" / "uploads",
        ]
        saved = False
        save_payload = {
            "filename": name,
            "format": parsed["format"],
            "indicators": parsed["indicators"],
            "observations": parsed["observations"],
            "recommendations": parsed["recommendations"],
            "default_selected": parsed["default_selected"],
            "quality": parsed.get("quality"),
        }
        for out_dir in candidates:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / saved_as
                out_path.write_text(json.dumps(save_payload), encoding="utf-8")
                saved = True
                break
            except OSError:
                continue
        if not saved:
            saved_as = None  # in-memory only; response still carries observations
    except Exception:
        saved_as = None

    return {
        "data": {
            "filename": name,
            "format": parsed["format"],
            "n_rows": parsed["n_rows"],
            "n_cols": parsed["n_cols"],
            "n_observations": parsed["n_observations"],
            "indicators": parsed["indicators"],
            "indicator_meta": parsed["indicator_meta"],
            "periods": parsed["periods"],
            "preview_rows": parsed["preview_rows"][: parsed["preview_limit"]],
            "preview_total_rows": parsed["n_rows"],
            "recommendations": parsed["recommendations"],
            "default_selected": parsed["default_selected"],
            "observations": parsed["observations"],
            "quality": parsed.get("quality"),
            "saved_as": saved_as,
            "replace": str(replace).lower() in ("1", "true", "yes"),
        },
        "is_synthetic": False,
        "message": "File berhasil dibaca dan divalidasi.",
    }


@router.get("/upload-template")
def upload_template():
    return {
        "data": {
            "supported_formats": [
                "CSV (.csv)",
                "TSV (.tsv)",
                "Excel (.xlsx, .xls)",
                "JSON (.json)",
                "Parquet (.parquet)",
                "PDF berisi tabel teks (.pdf)",
            ],
            "long_format_columns": [
                "indicator",
                "reference_period",
                "value",
                "publication_date",
                "unit",
                "frequency",
                "source",
            ],
            "long_format_example": (
                "indicator,reference_period,value,publication_date,unit,frequency,source\n"
                "credit_growth,2026-06,8.2,2026-07-20,percent,monthly,OJK\n"
                "gdp_growth,2026-06,5.1,2026-08-05,percent,monthly,BPS"
            ),
            "wide_format_example": (
                "period,gdp_growth,credit_growth,lending_rate\n"
                "2026-04,5.0,8.1,10.2\n"
                "2026-05,5.1,8.0,10.3\n"
                "2026-06,5.0,7.9,10.4"
            ),
        }
    }
