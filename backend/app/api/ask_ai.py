"""Ask AI chatbot — Qwen (DashScope OpenAI-compatible) integration."""

from __future__ import annotations

from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings

router = APIRouter(tags=["ask-ai"])

SYSTEM_PROMPT = """Anda adalah Ask AI di aplikasi SINAR (Sistem Nowcasting dan Analisis Risiko NPL Konsumsi).
Tugas Anda: membantu pengguna memahami nowcasting NPL, indikator, model, skenario, dan policy insight.

Aturan:
1. Jawab dalam Bahasa Indonesia yang jelas dan ringkas.
2. Jangan mengarang angka. Jika konteks aplikasi tersedia, gunakan angka itu; jika tidak, katakan data belum tersedia.
3. Bedakan Actual (rilis resmi), Nowcast (estimasi periode belum rilis), dan Forecast (proyeksi masa depan).
4. Tekankan asosiasi model ≠ kausalitas.
5. Mode demo memakai data sintetis — bukan statistik resmi.
6. Jika pertanyaan di luar SINAR/NPL, jawab singkat lalu arahkan kembali ke topik risiko kredit/NPL.
"""


class AskMessage(BaseModel):
    role: str
    content: str


class AskAIRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[AskMessage] = Field(default_factory=list)
    context: Optional[dict[str, Any]] = None


class AskAIResponse(BaseModel):
    reply: str
    model: str
    provider: str
    offline: bool = False


def _context_block(ctx: Optional[dict[str, Any]]) -> str:
    if not ctx:
        return "Konteks aplikasi: (belum ada ringkasan dari UI)."
    try:
        import json

        return "Konteks aplikasi (JSON):\n" + json.dumps(ctx, ensure_ascii=False, indent=2)[:6000]
    except Exception:
        return "Konteks aplikasi: (gagal dibaca)."


def _offline_reply(message: str, ctx: Optional[dict[str, Any]]) -> str:
    q = (message or "").lower()
    now = (ctx or {}).get("nowcast")
    model = (ctx or {}).get("model")
    status = (ctx or {}).get("risk_status")
    page = (ctx or {}).get("page")

    if any(k in q for k in ("nowcast", "apa itu npl", "npl", "forecast", "aktual")):
        return (
            "Di SINAR ada tiga jenis angka:\n"
            "• Actual — NPL resmi yang sudah terbit\n"
            "• Nowcast — estimasi untuk periode yang belum rilis resmi\n"
            "• Forecast — proyeksi ke depan\n\n"
            + (
                f"Nowcast terkini di sesi ini: {now}% (model {model or '—'}, status {status or '—'})."
                if now is not None
                else "Jalankan nowcast di halaman Model agar angka sesi tersedia."
            )
            + "\n\nCatatan: kunci API Qwen belum dikonfigurasi, jadi jawaban ini dari mode offline."
        )
    if any(k in q for k in ("skenario", "baseline", "custom", "stress")):
        return (
            "Halaman Skenario membandingkan Baseline Proyeksi (asumsi dasar) dengan Custom (slider/asumsi Anda). "
            "Geser variabel driver lalu lihat dampaknya pada jalur NPL.\n\n"
            "Untuk jawaban AI penuh, set NPL_QWEN_API_KEY di backend."
        )
    if any(k in q for k in ("vintage", "as-of", "as of", "look-ahead")):
        return (
            "Vintage/as-of memastikan model hanya memakai data yang sudah tersedia pada tanggal acuan, "
            "sehingga menghindari look-ahead bias."
        )
    if any(k in q for k in ("berita", "nsi", "sentimen", "news")):
        return (
            "News Stress Index (NSI) mengagregasi skor sentimen berita ke frekuensi bulanan dan dapat "
            "dipakai sebagai fitur model jika opsi berita aktif."
        )
    return (
        "Ask AI (mode offline): kunci Qwen belum diset.\n\n"
        "Tambahkan di environment backend:\n"
        "  NPL_QWEN_API_KEY=sk-...\n"
        "  NPL_QWEN_MODEL=qwen-plus  (opsional)\n\n"
        f"Halaman aktif: {page or '—'}. "
        "Setelah kunci aktif, saya bisa menjawab pertanyaan Anda dengan bantuan Qwen."
    )


@router.post("/ask-ai", response_model=AskAIResponse)
async def ask_ai(req: AskAIRequest):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(400, "Pesan kosong")

    api_key = (settings.qwen_api_key or "").strip()
    model = (settings.qwen_model or "qwen-plus").strip()
    base = (settings.qwen_base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").rstrip("/")

    if not api_key:
        return AskAIResponse(
            reply=_offline_reply(message, req.context),
            model="offline-faq",
            provider="local",
            offline=True,
        )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + _context_block(req.context)},
    ]
    for h in (req.history or [])[-12:]:
        role = h.role if h.role in ("user", "assistant") else "user"
        content = (h.content or "").strip()
        if content:
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": message[:4000]})

    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 1200,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Gagal menghubungi Qwen: {e}") from e

    if r.status_code >= 400:
        detail = r.text[:500]
        raise HTTPException(r.status_code, f"Qwen API error: {detail}")

    data = r.json()
    try:
        reply = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(502, f"Respons Qwen tidak valid: {data}") from e

    if not reply:
        reply = "Maaf, Qwen mengembalikan jawaban kosong. Coba ajukan pertanyaan lagi."

    return AskAIResponse(reply=reply, model=model, provider="qwen", offline=False)


@router.get("/ask-ai/status")
def ask_ai_status():
    has_key = bool((settings.qwen_api_key or "").strip())
    return {
        "configured": has_key,
        "model": settings.qwen_model,
        "provider": "qwen" if has_key else "offline",
        "base_url": settings.qwen_base_url,
    }
