#!/usr/bin/env python3
"""Generate SINAR architecture learning PDF for presentation."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "SINAR-Materi-Presentasi.pdf"
LOGO = ROOT / "backend" / "static" / "sinar-wordmark.png"

NAVY = colors.HexColor("#0C1F3A")
BLUE = colors.HexColor("#2563EB")
SLATE = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748B")
LIGHT = colors.HexColor("#F1F5F9")
BORDER = colors.HexColor("#CBD5E1")
WHITE = colors.white


def styles():
    base = getSampleStyleSheet()
    s = {
        "cover_title": ParagraphStyle(
            "cover_title", parent=base["Title"],
            fontName="Helvetica-Bold", fontSize=26, leading=32,
            textColor=NAVY, alignment=TA_CENTER, spaceAfter=8,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", parent=base["Normal"],
            fontName="Helvetica", fontSize=12, leading=16,
            textColor=MUTED, alignment=TA_CENTER, spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"],
            fontName="Helvetica-Bold", fontSize=16, leading=20,
            textColor=NAVY, spaceBefore=16, spaceAfter=8,
            borderPadding=3,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"],
            fontName="Helvetica-Bold", fontSize=12.5, leading=16,
            textColor=BLUE, spaceBefore=12, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"],
            fontName="Helvetica-Bold", fontSize=11, leading=14,
            textColor=SLATE, spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontName="Helvetica", fontSize=9.5, leading=13.5,
            textColor=SLATE, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"],
            fontName="Helvetica", fontSize=9.5, leading=13,
            textColor=SLATE, leftIndent=0, spaceAfter=2,
        ),
        "note": ParagraphStyle(
            "note", parent=base["Normal"],
            fontName="Helvetica-Oblique", fontSize=8.5, leading=12,
            textColor=MUTED, spaceBefore=4, spaceAfter=8,
        ),
        "code": ParagraphStyle(
            "code", parent=base["Code"],
            fontName="Courier", fontSize=7.5, leading=10,
            textColor=NAVY, backColor=LIGHT, leftIndent=4, rightIndent=4,
            spaceBefore=4, spaceAfter=8,
        ),
        "toc": ParagraphStyle(
            "toc", parent=base["Normal"],
            fontName="Helvetica", fontSize=10.5, leading=16,
            textColor=SLATE, leftIndent=8, spaceAfter=2,
        ),
        "footer": ParagraphStyle(
            "footer", parent=base["Normal"],
            fontName="Helvetica", fontSize=8, textColor=MUTED, alignment=TA_CENTER,
        ),
        "table": ParagraphStyle(
            "table", parent=base["Normal"],
            fontName="Helvetica", fontSize=8, leading=11, textColor=SLATE,
        ),
        "table_h": ParagraphStyle(
            "table_h", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=8, leading=11, textColor=WHITE,
        ),
    }
    return s


def p(text: str, style):
    return Paragraph(text.replace("\n", "<br/>"), style)


def bullets(items, style):
    return ListFlowable(
        [ListItem(Paragraph(i, style), leftIndent=12, bulletColor=BLUE) for i in items],
        bulletType="bullet",
        start="•",
        leftIndent=15,
        bulletFontSize=9,
    )


def simple_table(headers, rows, col_widths):
    S = styles()
    data = [[Paragraph(h, S["table_h"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), S["table"]) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("BACKGROUND", (0, 1), (-1, -1), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def add_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BLUE)
    canvas.setLineWidth(1.2)
    canvas.line(1.8 * cm, A4[1] - 1.2 * cm, A4[0] - 1.8 * cm, A4[1] - 1.2 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.8 * cm, A4[1] - 1.0 * cm, "SINAR — Materi Belajar Presentasi")
    canvas.drawRightString(A4[0] - 1.8 * cm, A4[1] - 1.0 * cm, "NPL Nowcasting")
    canvas.setStrokeColor(BORDER)
    canvas.line(1.8 * cm, 1.3 * cm, A4[0] - 1.8 * cm, 1.3 * cm)
    canvas.drawCentredString(A4[0] / 2, 0.85 * cm, f"Halaman {doc.page}")
    canvas.restoreState()


def build():
    S = styles()
    story = []

    # —— COVER ——
    story.append(Spacer(1, 1.5 * cm))
    if LOGO.exists():
        img = Image(str(LOGO), width=12 * cm, height=6 * cm, kind="proportional")
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Spacer(1, 0.6 * cm))
    story.append(p("SINAR", S["cover_title"]))
    story.append(p(
        "Sistem Nowcasting dan Analisis Risiko NPL Konsumsi",
        S["cover_sub"],
    ))
    story.append(Spacer(1, 0.4 * cm))
    story.append(p(
        "<b>Materi Belajar Presentasi</b><br/>"
        "Penjelasan Rinci: Frontend · Backend · Workflow Keseluruhan",
        S["cover_sub"],
    ))
    story.append(Spacer(1, 1.2 * cm))
    box = Table(
        [[Paragraph(
            "<b>Dokumen pembelajaran internal</b><br/>"
            "Disusun dari kode aktual proyek NPL/SINAR.<br/>"
            "Mode DEMO memakai data sintetis — bukan statistik resmi.",
            ParagraphStyle("box", parent=S["body"], alignment=TA_CENTER, textColor=NAVY),
        )]],
        colWidths=[14 * cm],
    )
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 1, BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    box.hAlign = "CENTER"
    story.append(box)
    story.append(Spacer(1, 1.5 * cm))
    story.append(p(
        "Cara pakai dokumen ini:<br/>"
        "1) Baca Bab 1 untuk narasi pembuka presentasi.<br/>"
        "2) Bab 2–3 untuk demo teknis frontend &amp; backend.<br/>"
        "3) Bab 4 untuk alur end-to-end saat live demo.<br/>"
        "4) Bab 5 untuk konsep inti yang sering ditanya audiens.",
        S["note"],
    ))
    story.append(PageBreak())

    # —— TOC ——
    story.append(p("Daftar Isi", S["h1"]))
    for line in [
        "1. Ringkasan Eksekutif &amp; Konsep Produk",
        "2. Frontend (SPA SINAR)",
        "3. Backend (FastAPI &amp; Engine)",
        "4. Workflow End-to-End",
        "5. Konsep Kunci untuk Presentasi (Q&amp;A)",
        "6. Struktur Folder &amp; Referensi Cepat",
        "7. Cara Menjalankan &amp; Deploy",
        "8. Checklist Presentasi Besok",
    ]:
        story.append(p(line, S["toc"]))
    story.append(PageBreak())

    # —— 1 ——
    story.append(p("1. Ringkasan Eksekutif &amp; Konsep Produk", S["h1"]))
    story.append(p(
        "<b>SINAR</b> adalah sistem cerdas untuk <b>mengestimasi kondisi NPL (Non-Performing Loan) "
        "konsumsi terkini</b> sebelum statistik resmi keluar, lalu menganalisis risiko melalui "
        "indikator makro, sentimen berita, model nowcasting, simulasi skenario, dan policy insight.",
        S["body"],
    ))
    story.append(p("1.1 Tiga jenis angka yang harus dibedakan", S["h2"]))
    story.append(simple_table(
        ["Tipe", "Arti", "Contoh di UI"],
        [
            ["Actual NPL", "Nilai historis resmi yang sudah terbit", "Garis biru “Aktual”"],
            ["Nowcast NPL", "Estimasi model untuk periode yang sudah lewat kalender tetapi belum rilis resmi", "Titik/garis oranye “Nowcast”"],
            ["Forecast NPL", "Proyeksi untuk periode masa depan", "Garis putus “Prediksi / Baseline Proyeksi”"],
        ],
        [3.2 * cm, 7.5 * cm, 5.3 * cm],
    ))
    story.append(p(
        "Pesan penting untuk audiens: <b>nowcast ≠ forecast</b>. Nowcast menjawab “berapa NPL sekarang yang belum dirilis?”, "
        "sedangkan forecast menjawab “kemana arah NPL ke depan?”.",
        S["note"],
    ))

    story.append(p("1.2 Tech stack ringkas", S["h2"]))
    story.append(simple_table(
        ["Lapisan", "Teknologi", "Catatan"],
        [
            ["UI utama", "SPA HTML/CSS/JS monolitik", "backend/static/index.html dilayani di /"],
            ["UI alternatif", "npl-nowcasting.html", "Salinan hampir identik (path aset relatif)"],
            ["Frontend opsional", "Next.js 14 + React", "folder frontend/ — bukan UI utama FastAPI"],
            ["Backend", "FastAPI + pandas + scikit-learn + statsmodels", "XGBoost/LightGBM opsional"],
            ["Lokal", "uvicorn :8000", "http://127.0.0.1:8000"],
            ["Deploy", "Vercel + Mangum", "api/index.py, batas serverless ~60 detik"],
        ],
        [3.2 * cm, 5.5 * cm, 7.3 * cm],
    ))

    story.append(p("1.3 Narasi pembuka yang bisa dibacakan", S["h2"]))
    story.append(p(
        "“SINAR membantu analis memantau risiko kredit konsumsi secara lebih dini. "
        "Sistem menggabungkan data indikator, sinyal berita, dan model statistik/ML dengan disiplin vintage "
        "agar tidak memakai informasi yang belum tersedia pada tanggal as-of. Hasilnya ditampilkan sebagai "
        "nowcast, lalu diuji melalui skenario dan dirangkum menjadi policy insight.”",
        S["body"],
    ))
    story.append(PageBreak())

    # —— 2 FRONTEND ——
    story.append(p("2. Frontend (SPA SINAR)", S["h1"]))
    story.append(p(
        "Frontend produksi yang dipakai bersama backend adalah <b>single-page application</b> dalam satu file "
        "<font face='Courier'>backend/static/index.html</font>. Tidak memakai React di jalur utama; seluruh navigasi, "
        "state, chart SVG, dan orkestrasi pipeline ada di JavaScript browser.",
        S["body"],
    ))

    story.append(p("2.1 Halaman navigasi", S["h2"]))
    story.append(simple_table(
        ["ID halaman", "Label menu", "Fungsi utama"],
        [
            ["dashboard", "Dashboard", "KPI NPL, aktual vs nowcast/forecast, ringkasan"],
            ["data", "Data", "Upload indikator, Google News, apply sumber data"],
            ["indicators", "Indikator", "Deret makro/pasar/perbankan, korelasi, berita"],
            ["models", "Model", "Konfigurasi, leaderboard backtest, prediksi fitted"],
            ["scenario", "Skenario", "Baseline Proyeksi vs Custom, sensitivitas"],
            ["policy", "Policy Insight", "Risiko, driver, rekomendasi kebijakan"],
        ],
        [3.0 * cm, 3.2 * cm, 9.8 * cm],
    ))
    story.append(p(
        "Navigasi dikendalikan fungsi <font face='Courier'>go(p)</font>: menyembunyikan semua "
        "<font face='Courier'>.page</font>, menampilkan <font face='Courier'>#pg-{id}</font>, "
        "lalu memanggil renderer terkait (<font face='Courier'>renderDashboard</font>, "
        "<font face='Courier'>renderIndicators</font>, dst.). Array <font face='Courier'>PAGES</font> "
        "mendefinisikan menu; <font face='Courier'>PAGE_ALIASES</font> memetakan nama lama "
        "(mis. nowcast→dashboard).",
        S["body"],
    ))

    story.append(p("2.2 State penting di browser", S["h2"]))
    story.append(simple_table(
        ["Objek", "Isi"],
        [
            ["CFG", "Konfigurasi pipeline: target, frekuensi, asOf, horizon, daftar model, useNews, vars, backtest window, …"],
            ["RES", "Hasil terakhir: dataset (ds), leaderboard, model terpilih (chosen), path proyeksi, nowcast, kontribusi driver, fits"],
            ["INPUT_STATE", "Status upload: file, hasil parse, indikator dipilih, artikel berita, flag dataApplied"],
            ["RAW / META", "Deret bulanan per indikator + metadata (nama, unit, lag publikasi, sumber)"],
            ["SC_BASE, SC_LIST, SC_ACTIVE, SC_CACHE, SC_UI", "State skenario: nilai baseline, variabel aktif, cache path Baseline/Custom, timeframe Dari–Sampai"],
            ["DASH_UI / IND_UI / MD_UI", "State timeframe grafik per halaman"],
        ],
        [4.5 * cm, 11.5 * cm],
    ))

    story.append(p("2.3 Pola grafik &amp; timeframe", S["h2"]))
    story.append(bullets([
        "Fungsi <b>lineChart()</b> menggambar SVG multi-series (aktual, fitted, prediksi, band keyakinan).",
        "Hampir semua grafik punya kontrol <b>Dari / Sampai</b> (input type=month).",
        "<b>resolveMonthRange()</b> menormalisasi rentang; helper <b>ensure*HistRange()</b> per halaman.",
        "Di Skenario: Aktual + <b>Baseline Proyeksi</b> + Custom, dengan snap agar garis berimpit jika asumsi sama.",
    ], S["bullet"]))

    story.append(p("2.4 Mode DEMO vs DATA", S["h2"]))
    story.append(p(
        "Fungsi <font face='Courier'>hasAppliedData()</font> menentukan apakah user sudah menerapkan unggahan. "
        "Tanpa data applied, banyak halaman menampilkan empty state. Setelah apply, badge UI berubah ke "
        "<b>DATA</b>; jika masih sintetis, tetap <b>DEMO</b>. Pipeline default menolak jalan tanpa data, "
        "kecuali opsi khusus <font face='Courier'>forceDemo</font>.",
        S["body"],
    ))

    story.append(p("2.5 API yang dipanggil dari frontend", S["h2"]))
    story.append(simple_table(
        ["Method &amp; Path", "Dipakai untuk"],
        [
            ["GET /api/data/upload-template", "Unduh template indikator"],
            ["POST /api/data/upload-indicators", "Parse CSV/Excel/PDF/JSON/Parquet"],
            ["POST /api/data/google-news", "Ambil berita RSS (sebaran bulanan)"],
            ["POST /api/sentiment/batch", "Skor NLP batch → News Stress Index"],
            ["POST /api/workflow/run", "Pipeline backend (jalur demo/tanpa upload user)"],
        ],
        [6.5 * cm, 9.5 * cm],
    ))
    story.append(p(
        "Catatan arsitektur: setelah user <b>menerapkan unggahan</b>, evaluasi model utama dijalankan "
        "<b>di browser</b> agar Model/Skenario selaras dengan deret user. Backend tetap dipakai untuk "
        "upload, Google News, dan sentiment batch.",
        S["note"],
    ))
    story.append(PageBreak())

    # —— 3 BACKEND ——
    story.append(p("3. Backend (FastAPI &amp; Engine)", S["h1"]))
    story.append(p(
        "Backend Python mengorkestrasi data demo, vintage panel, NLP, feature engineering, backtest, "
        "dan nowcast. Entry: <font face='Courier'>backend/main.py</font> → "
        "<font face='Courier'>from app import app</font>.",
        S["body"],
    ))

    story.append(p("3.1 Susunan paket", S["h2"]))
    story.append(Preformatted(
        "backend/\n"
        "  main.py\n"
        "  app/\n"
        "    __init__.py          # FastAPI, CORS, /api, /health, mount static\n"
        "    static_mount.py      # GET / → index.html ; /static/*\n"
        "    api/                 # endpoint HTTP\n"
        "    core/config.py       # NPL_* settings\n"
        "    data/                # schemas, synthetic, connectors\n"
        "    engines/\n"
        "      vintage/           # anti look-ahead by as-of\n"
        "      sentiment/         # NLP + News Stress Index\n"
        "      features/          # feature matrix & selection\n"
        "      nowcasting/        # katalog model + metrik\n"
        "      evaluation/        # expanding/rolling backtest\n"
        "    services/\n"
        "      nowcasting_service.py  # orkestrasi utama\n"
        "      bootstrap.py           # siapkan data demo\n"
        "  static/index.html\n"
        "  data/demo/             # parquet + meta.json sintetis",
        S["code"],
    ))

    story.append(p("3.2 Static mount", S["h2"]))
    story.append(bullets([
        "<b>GET /</b> → FileResponse(backend/static/index.html)",
        "<b>/static/*</b> → aset (logo, gambar gedung BI, …)",
        "<b>/api/*</b> → JSON API",
        "<b>GET /health</b> → health check",
        "Dokumentasi interaktif: <b>/docs</b> (Swagger)",
    ], S["bullet"]))

    story.append(p("3.3 Endpoint utama", S["h2"]))
    story.append(simple_table(
        ["Endpoint", "Fungsi"],
        [
            ["POST /api/workflow/run", "Full pipeline: backtest → nowcast → timeseries → news → analyst"],
            ["POST /api/nowcast", "Nowcast saja"],
            ["POST /api/backtest", "Leaderboard RMSE/MAE/MAPE"],
            ["GET /api/timeseries", "Deret actual/nowcast/forecast"],
            ["POST /api/sentiment/batch", "Skor banyak headline + NSI"],
            ["POST /api/scenario", "Simulasi skenario di server"],
            ["POST /api/data/upload-indicators", "Upload &amp; parse indikator"],
            ["POST /api/data/google-news", "Fetch berita Google News RSS"],
            ["GET /api/catalog, /api/meta, /api/alerts, …", "Katalog, meta demo, alert risiko, dsb."],
        ],
        [6.2 * cm, 9.8 * cm],
    ))

    story.append(p("3.4 Engine inti (jelaskan berurutan di presentasi)", S["h2"]))
    story.append(bullets([
        "<b>VintageEngine</b> — hanya data dengan tanggal publikasi ≤ as-of (cegah look-ahead bias).",
        "<b>SentimentEngine</b> — TF-IDF/logistic (dan transformer jika tersedia) → skor → agregasi bulanan NSI.",
        "<b>Feature engineering</b> — bangun matriks fitur + seleksi otomatis.",
        "<b>Nowcasting models</b> — katalog econometric/ML/hybrid (VAR, Ridge, MIDAS-news, dll.).",
        "<b>Evaluation/backtest</b> — validasi temporal (expanding/rolling), ranking by RMSE.",
        "<b>NowcastingService</b> — mengikat semua langkah di atas untuk API.",
    ], S["bullet"]))

    story.append(p("3.5 Data demo sintetis", S["h2"]))
    story.append(p(
        "Pada startup, <font face='Courier'>bootstrap_demo_if_needed()</font> dapat menulis file di "
        "<font face='Courier'>backend/data/demo/</font> (observations/news parquet + meta.json). "
        "Label mode: <b>DEMO / SYNTHETIC DATA</b>. Tekankan di slide: bukan angka OJK/BI resmi.",
        S["body"],
    ))
    story.append(PageBreak())

    # —— 4 WORKFLOW ——
    story.append(p("4. Workflow End-to-End", S["h1"]))
    story.append(p(
        "Alur user yang paling natural untuk live demo presentasi:",
        S["body"],
    ))
    story.append(Preformatted(
        "1. DATA\n"
        "   Upload indikator → pilih kolom → Terapkan\n"
        "   (opsional) Ambil Google News + set rentang tanggal\n"
        "        ↓\n"
        "2. INDIKATOR\n"
        "   Lihat deret & korelasi · proses sentimen berita → News Stress Index\n"
        "        ↓\n"
        "3. MODEL\n"
        "   Atur as-of / kandidat model · Jalankan nowcast (runPipeline)\n"
        "   → leaderboard backtest → model terpilih → path aktual/fitted/prediksi\n"
        "        ↓\n"
        "4. DASHBOARD\n"
        "   Baca KPI NPL terkini, nowcast, outlook, driver utama\n"
        "        ↓\n"
        "5. SKENARIO\n"
        "   Bandingkan Baseline Proyeksi vs Custom (slider asumsi)\n"
        "   Analisis sensitivitas + ringkasan insight\n"
        "        ↓\n"
        "6. POLICY INSIGHT\n"
        "   Ringkas risiko kredit + rekomendasi kebijakan",
        S["code"],
    ))

    story.append(p("4.1 Apa yang terjadi saat “Jalankan nowcast”", S["h2"]))
    story.append(p(
        "Fungsi frontend <font face='Courier'>runPipeline()</font> adalah orkestrator UI:",
        S["body"],
    ))
    story.append(bullets([
        "Cek gate: butuh data applied (kecuali forceDemo).",
        "Jika useNews aktif → POST /api/sentiment/batch untuk skor korpus.",
        "Bangun panel vintage di browser (buildDataset + lag publikasi).",
        "Jika belum ada upload applied → boleh align ke POST /api/workflow/run (data demo backend).",
        "Jika sudah apply upload → backtest &amp; prediksi di browser memakai deret user → isi objek RES.",
        "renderAll() memperbarui semua halaman.",
    ], S["bullet"]))

    story.append(p("4.2 Alur skenario (poin demo yang menarik)", S["h2"]))
    story.append(bullets([
        "<b>ensureScenarioList()</b> menentukan variabel yang bisa digeser.",
        "<b>projectScenarioPath(overrides)</b> membangun ulang dataset dengan asumsi shock, lalu memprediksi maju dengan model terpilih.",
        "<b>Baseline Proyeksi</b>: asumsi baseline eksak (bukan posisi slider).",
        "<b>Custom</b>: dari slider/input angka; jika hampir sama dengan baseline, garis dibuat berimpit.",
        "Timeframe Dari/Sampai memfilter jendela historis + horizon proyeksi.",
    ], S["bullet"]))

    story.append(p("4.3 Dua “mesin” dalam satu produk", S["h2"]))
    story.append(simple_table(
        ["Mesin", "Kapan dipakai", "Sumber data"],
        [
            ["Python FastAPI", "Demo tanpa upload / endpoint API / NLP batch / upload parse", "Parquet demo + request"],
            ["JavaScript in-browser", "Setelah user apply indikator unggahan (jalur utama UI)", "RAW/META di memori browser"],
        ],
        [4.0 * cm, 6.5 * cm, 5.5 * cm],
    ))
    story.append(p(
        "Ini jawaban kuat jika ditanya “kenapa hasil model ikut berubah setelah saya upload?” — karena jalur evaluasi mengikuti data yang diterapkan di UI.",
        S["note"],
    ))
    story.append(PageBreak())

    # —— 5 CONCEPTS ——
    story.append(p("5. Konsep Kunci untuk Presentasi (Q&amp;A)", S["h1"]))

    story.append(p("5.1 Nowcasting", S["h2"]))
    story.append(p(
        "Nowcasting = estimasi kondisi terkini yang belum tersedia di statistik resmi. "
        "Dalam kredit, ini berguna untuk early warning sebelum angka NPL bulanan/kuartalan terbit.",
        S["body"],
    ))

    story.append(p("5.2 Vintage / as-of", S["h2"]))
    story.append(p(
        "Setiap indikator punya <b>lag publikasi</b>. Pada tanggal as-of tertentu, sistem hanya memakai "
        "observasi yang sudah seharusnya tersedia. Ini mencegah kebocoran informasi masa depan ke model "
        "(look-ahead bias) — kredibilitas metodologi yang penting di hadapan audiens teknis/BI.",
        S["body"],
    ))

    story.append(p("5.3 News Stress Index (NSI)", S["h2"]))
    story.append(p(
        "Headline berita diberi skor sentimen, lalu diagregasi ke frekuensi bulanan menjadi indeks tekanan media. "
        "NSI dapat diikutkan sebagai fitur model (CFG.useNews). Tekankan: asosiasi, bukan klaim kausal langsung.",
        S["body"],
    ))

    story.append(p("5.4 Backtest temporal &amp; metrik", S["h2"]))
    story.append(bullets([
        "<b>RMSE</b> — akar rata-rata kuadrat error (utama untuk ranking model di backend).",
        "<b>MAE</b> — rata-rata absolut error.",
        "<b>MAPE</b> — error persentase (sering ditampilkan di UI model).",
        "Validasi memakai skema waktu (expanding/rolling), bukan acak silang yang merusak urutan waktu.",
    ], S["bullet"]))

    story.append(p("5.5 Scenario stress testing", S["h2"]))
    story.append(p(
        "User mengubah asumsi driver (mis. pertumbuhan kredit, suku bunga, NSI). Model yang sama memproyeksikan "
        "ulang jalur NPL. Perbandingan Baseline Proyeksi vs Custom menunjukkan sensitivitas risiko.",
        S["body"],
    ))

    story.append(p("5.6 Risk signal", S["h2"]))
    story.append(p(
        "Status risiko diturunkan dari level nowcast dan perubahan (pp) terhadap rilis terakhir "
        "(mis. low → elevated → high → critical). Cocok ditampilkan di Dashboard/Policy Insight.",
        S["body"],
    ))

    story.append(p("5.7 Disclaimer wajib", S["h2"]))
    story.append(p(
        "Di mode demo, semua angka bersifat <b>sintetis</b>. Jangan dipresentasikan sebagai statistik resmi "
        "Indonesia. Driver/contribution adalah <b>asosiasi model</b>, bukan bukti kausalitas.",
        S["body"],
    ))
    story.append(PageBreak())

    # —— 6 STRUCTURE ——
    story.append(p("6. Struktur Folder &amp; Referensi Cepat", S["h1"]))
    story.append(simple_table(
        ["Topik presentasi", "File yang dibuka"],
        [
            ["UI / branding SPA", "backend/static/index.html"],
            ["Salinan SPA root", "npl-nowcasting.html"],
            ["Entry FastAPI", "backend/main.py , backend/app/__init__.py"],
            ["Static /", "backend/app/static_mount.py"],
            ["Semua API", "backend/app/api/__init__.py , data_input.py"],
            ["Orkestrasi", "backend/app/services/nowcasting_service.py"],
            ["Vintage", "backend/app/engines/vintage/engine.py"],
            ["NLP / NSI", "backend/app/engines/sentiment/engine.py"],
            ["Model + metrik", "backend/app/engines/nowcasting/models.py"],
            ["Backtest", "backend/app/engines/evaluation/backtest.py"],
            ["Schema", "backend/app/data/schemas/models.py"],
            ["Data demo", "backend/data/demo/ , synthetic.py"],
            ["Deploy", "api/index.py , vercel.json , README.md"],
        ],
        [5.0 * cm, 11.0 * cm],
    ))
    story.append(PageBreak())

    # —— 7 RUN ——
    story.append(p("7. Cara Menjalankan &amp; Deploy", S["h1"]))
    story.append(p("7.1 Lokal (rekomendasi demo presentasi)", S["h2"]))
    story.append(Preformatted(
        "cd backend\n"
        "source .venv/bin/activate\n"
        "uvicorn main:app --reload --host 127.0.0.1 --port 8000\n"
        "\n"
        "Buka:  http://127.0.0.1:8000/\n"
        "API:   http://127.0.0.1:8000/docs\n"
        "Health:http://127.0.0.1:8000/health",
        S["code"],
    ))
    story.append(p("7.2 Deploy ringkas", S["h2"]))
    story.append(p(
        "Proyek dapat di-deploy ke Vercel melalui wrapper Mangum di <font face='Courier'>api/index.py</font>. "
        "Perhatikan batas waktu serverless (~60 detik) untuk pekerjaan ML berat — untuk presentasi teknis, "
        "jalankan lokal dengan uvicorn lebih aman dan responsif.",
        S["body"],
    ))
    story.append(PageBreak())

    # —— 8 CHECKLIST ——
    story.append(p("8. Checklist Presentasi Besok", S["h1"]))
    story.append(p("8.1 Alur slide yang disarankan (15–20 menit)", S["h2"]))
    story.append(bullets([
        "Slide 1 — Apa itu SINAR &amp; masalah yang diselesaikan (early warning NPL).",
        "Slide 2 — Actual vs Nowcast vs Forecast (tabel 3 kolom).",
        "Slide 3 — Arsitektur big picture (UI ↔ API ↔ Engine).",
        "Slide 4 — Live demo Data → Model → Dashboard.",
        "Slide 5 — Live demo Skenario (Baseline Proyeksi vs Custom).",
        "Slide 6 — Policy Insight &amp; rekomendasi.",
        "Slide 7 — Metodologi: vintage, NSI, backtest temporal.",
        "Slide 8 — Batasan: data sintetis, asosiasi ≠ kausalitas.",
        "Slide 9 — Q&amp;A.",
    ], S["bullet"]))

    story.append(p("8.2 Script demo 5 menit", S["h2"]))
    story.append(bullets([
        "Buka http://127.0.0.1:8000/ — tunjukkan logo SINAR &amp; navigasi.",
        "Data: upload/apply sampel (atau jelaskan mode DEMO).",
        "Model: jalankan nowcast — tunjukkan leaderboard &amp; grafik fitted/prediksi.",
        "Dashboard: baca KPI nowcast + timeframe Dari/Sampai.",
        "Skenario: geser satu asumsi — bandingkan Baseline Proyeksi vs Custom.",
        "Policy Insight: tutup dengan status risiko &amp; rekomendasi.",
    ], S["bullet"]))

    story.append(p("8.3 Pertanyaan yang sering muncul + jawaban singkat", S["h2"]))
    story.append(simple_table(
        ["Pertanyaan", "Jawaban singkat"],
        [
            ["Apakah angka ini resmi?", "Tidak. Mode demo = data sintetis / simulasi."],
            ["Nowcast beda apa dengan forecast?", "Nowcast: periode belum rilis. Forecast: masa depan."],
            ["Kenapa perlu as-of/vintage?", "Agar model tidak “curang” memakai data yang belum terbit."],
            ["Berita masuk bagaimana?", "Skor NLP → agregat bulanan NSI → opsional jadi fitur."],
            ["Model mana yang dipakai?", "Auto-select dari backtest temporal (umumnya RMSE terendah) atau pilihan user."],
            ["Skenario mengubah apa?", "Asumsi driver; model yang sama memproyeksikan ulang jalur NPL."],
        ],
        [5.0 * cm, 11.0 * cm],
    ))

    story.append(Spacer(1, 1.0 * cm))
    closing = Table(
        [[Paragraph(
            "<b>Semoga berhasil presentasinya.</b><br/><br/>"
            "Inti yang perlu diingat: SINAR menggabungkan data, sentimen, dan model "
            "dengan disiplin waktu (vintage) untuk memberi early insight risiko NPL, "
            "lalu mengujinya lewat skenario dan merangkumnya sebagai policy insight.",
            ParagraphStyle("close", parent=S["body"], alignment=TA_CENTER, textColor=NAVY),
        )]],
        colWidths=[16 * cm],
    )
    closing.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 1.5, BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(closing)

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="SINAR — Materi Belajar Presentasi",
        author="SINAR / NPL Nowcasting",
    )
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
