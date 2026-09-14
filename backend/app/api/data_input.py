"""Data input endpoints: multi-format indicator upload + Google News."""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, List, Optional
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
    # None / omitted = ambil semua item yang dikembalikan RSS (tanpa batas artifisial)
    max_articles: Optional[int] = Field(default=None, ge=1, le=5000)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _parse_rss(xml_text: str, max_articles: Optional[int] = None) -> List[dict]:
    root = ET.fromstring(xml_text)
    items: List[dict] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source = (item.findtext("source") or "").strip()
        pub = item.findtext("pubDate") or ""
        desc = _strip_html(item.findtext("description") or "")
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


@router.post("/google-news")
async def fetch_google_news(req: GoogleNewsRequest):
    """Ambil sebanyak mungkin artikel dari Google News RSS.

    Setiap kata kunci diambil terpisah (bukan digabung OR) agar cakupan lebih lengkap,
    lalu digabung dan deduplikasi. Google News RSS biasanya ~100 item per query;
    multi-keyword menghilangkan batas artifisial di sisi aplikasi.
    """
    tags = [t.strip() for t in (req.keywords or []) if t and str(t).strip()]
    queries: list[str] = []
    if tags:
        for t in tags:
            # spasi → kutip agar frasa utuh
            queries.append(f'"{t}"' if (" " in t and not t.startswith('"')) else t)
    else:
        queries = [req.query]

    seen: set[str] = set()
    articles: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=40.0, follow_redirects=True) as client:
            for q in queries:
                try:
                    batch = await _fetch_one_query(
                        client, q, req.language, req.country, req.max_articles
                    )
                    for a in batch:
                        k = _article_key(a)
                        if k in seen:
                            continue
                        seen.add(k)
                        a = {**a, "matched_query": q}
                        articles.append(a)
                except httpx.HTTPError as e:
                    errors.append(f"{q}: {e}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Gagal mengambil Google News: {e}") from e

    if not articles and errors:
        raise HTTPException(status_code=502, detail="Gagal mengambil Google News: " + "; ".join(errors[:3]))

    # terbaru dulu
    def _sort_key(a: dict[str, Any]) -> str:
        return a.get("published_at") or ""

    articles.sort(key=_sort_key, reverse=True)

    return {
        "data": {
            "query": " OR ".join(queries) if len(queries) > 1 else queries[0],
            "keywords": tags or None,
            "count": len(articles),
            "articles": articles,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "Google News RSS",
            "queries_run": len(queries),
            "partial_errors": errors or None,
        },
        "is_synthetic": False,
        "message": f"Berita diambil dari Google News RSS ({len(queries)} query, {len(articles)} unik).",
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
        if isinstance(payload, list):
            df = pd.DataFrame(payload)
        elif isinstance(payload, dict) and "data" in payload:
            df = pd.DataFrame(payload["data"])
        else:
            df = pd.DataFrame(payload)
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

    out_dir = Path(__file__).resolve().parents[2] / "data" / "uploads"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"indicators_{stamp}.json"
    # Don't dump huge observations twice in response-only path; save compact
    save_payload = {
        "filename": name,
        "format": parsed["format"],
        "indicators": parsed["indicators"],
        "observations": parsed["observations"],
        "recommendations": parsed["recommendations"],
        "default_selected": parsed["default_selected"],
        "quality": parsed.get("quality"),
    }
    out_path.write_text(json.dumps(save_payload), encoding="utf-8")

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
            "saved_as": out_path.name,
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
