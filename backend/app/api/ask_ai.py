"""Ask AI chatbot — Qwen via DashScope (OpenAI-compatible)."""

from __future__ import annotations

import os
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

# DashScope regions (OpenAI-compatible)
BASE_URLS = [
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
]


class AskMessage(BaseModel):
    role: str
    content: str


class AskAIRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[AskMessage] = Field(default_factory=list)
    context: Optional[dict[str, Any]] = None
    # Optional per-request overrides (e.g. key from Ask AI settings UI)
    api_key: Optional[str] = Field(default=None, max_length=256)
    model: Optional[str] = Field(default=None, max_length=128)
    base_url: Optional[str] = Field(default=None, max_length=512)


class AskAIResponse(BaseModel):
    reply: str
    model: str
    provider: str
    offline: bool = False
    base_url: Optional[str] = None


def _resolve_api_key(override: Optional[str] = None) -> str:
    for candidate in (
        override,
        settings.qwen_api_key,
        os.environ.get("NPL_QWEN_API_KEY"),
        os.environ.get("DASHSCOPE_API_KEY"),
        os.environ.get("QWEN_API_KEY"),
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return ""


def _resolve_model(override: Optional[str] = None) -> str:
    return (override or settings.qwen_model or "qwen-plus").strip() or "qwen-plus"


def _resolve_bases(override: Optional[str] = None) -> list[str]:
    if override and override.strip():
        return [override.strip().rstrip("/")]
    preferred = (settings.qwen_base_url or "").strip().rstrip("/")
    out: list[str] = []
    if preferred:
        out.append(preferred)
    for u in BASE_URLS:
        if u not in out:
            out.append(u)
    return out


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
            + "\n\nQwen belum terhubung. Buka pengaturan Ask AI (ikon ⚙) lalu tempel API key DashScope."
        )
    return (
        "Qwen belum terhubung.\n\n"
        "Cara mengaktifkan:\n"
        "1. Dapatkan API key di https://dashscope.console.aliyun.com/\n"
        "2. Klik ikon ⚙ pada panel Ask AI\n"
        "3. Tempel API key, simpan, lalu kirim pertanyaan lagi.\n\n"
        "Atau set environment `NPL_QWEN_API_KEY` / `DASHSCOPE_API_KEY` di server."
    )


async def _chat_qwen(
    *,
    api_key: str,
    model: str,
    bases: list[str],
    messages: list[dict[str, str]],
) -> tuple[str, str, str]:
    """Returns (reply, model, base_url_used). Tries each base URL until one succeeds."""
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
    errors: list[str] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for base in bases:
            url = f"{base.rstrip('/')}/chat/completions"
            try:
                r = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as e:
                errors.append(f"{base}: network {e}")
                continue
            if r.status_code >= 400:
                errors.append(f"{base}: HTTP {r.status_code} {r.text[:240]}")
                # auth error — no point trying other regions with same key style? still try
                continue
            data = r.json()
            try:
                reply = (data["choices"][0]["message"]["content"] or "").strip()
            except Exception:
                errors.append(f"{base}: invalid payload")
                continue
            if not reply:
                reply = "Maaf, Qwen mengembalikan jawaban kosong. Coba ajukan pertanyaan lagi."
            return reply, model, base
    raise HTTPException(
        502,
        "Gagal memanggil Qwen. Periksa API key DashScope dan region. Detail: "
        + " | ".join(errors)[:800],
    )


@router.post("/ask-ai", response_model=AskAIResponse)
async def ask_ai(req: AskAIRequest):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(400, "Pesan kosong")

    api_key = _resolve_api_key(req.api_key)
    model = _resolve_model(req.model)
    bases = _resolve_bases(req.base_url)

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

    reply, used_model, used_base = await _chat_qwen(
        api_key=api_key, model=model, bases=bases, messages=messages
    )
    return AskAIResponse(
        reply=reply,
        model=used_model,
        provider="qwen",
        offline=False,
        base_url=used_base,
    )


@router.get("/ask-ai/status")
def ask_ai_status():
    has_key = bool(_resolve_api_key())
    return {
        "configured": has_key,
        "model": settings.qwen_model,
        "provider": "qwen" if has_key else "offline",
        "base_url": settings.qwen_base_url,
        "hint": None
        if has_key
        else "Tempel API key DashScope di pengaturan Ask AI, atau set NPL_QWEN_API_KEY / DASHSCOPE_API_KEY.",
    }


@router.post("/ask-ai/test")
async def ask_ai_test(req: AskAIRequest):
    """Quick connectivity test with optional client-provided key."""
    api_key = _resolve_api_key(req.api_key)
    if not api_key:
        return {"ok": False, "offline": True, "message": "API key belum diisi"}
    model = _resolve_model(req.model)
    bases = _resolve_bases(req.base_url)
    try:
        reply, used_model, used_base = await _chat_qwen(
            api_key=api_key,
            model=model,
            bases=bases,
            messages=[
                {"role": "system", "content": "Jawab sangat singkat dalam Bahasa Indonesia."},
                {"role": "user", "content": req.message or "Katakan: Qwen siap untuk SINAR."},
            ],
        )
        return {
            "ok": True,
            "offline": False,
            "model": used_model,
            "base_url": used_base,
            "reply": reply[:240],
        }
    except HTTPException as e:
        return {"ok": False, "offline": False, "message": str(e.detail)}
