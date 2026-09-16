"""Ask AI chatbot — Google Gemini (free-tier flash models)."""

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

# Free-tier friendly chat models (tried in order if preferred fails)
FREE_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class AskMessage(BaseModel):
    role: str
    content: str


class AskAIRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[AskMessage] = Field(default_factory=list)
    context: Optional[dict[str, Any]] = None
    api_key: Optional[str] = Field(default=None, max_length=256)
    model: Optional[str] = Field(default=None, max_length=128)
    base_url: Optional[str] = Field(default=None, max_length=512)  # unused for Gemini; kept for UI compat


class AskAIResponse(BaseModel):
    reply: str
    model: str
    provider: str
    offline: bool = False
    base_url: Optional[str] = None


def _resolve_api_key(override: Optional[str] = None) -> str:
    for candidate in (
        override,
        settings.gemini_api_key,
        settings.qwen_api_key,  # legacy fallback if only old env set
        os.environ.get("NPL_GEMINI_API_KEY"),
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
        os.environ.get("NPL_QWEN_API_KEY"),
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return ""


def _model_candidates(override: Optional[str] = None) -> list[str]:
    preferred = (override or settings.gemini_model or "gemini-3.6-flash").strip()
    out: list[str] = []
    if preferred:
        out.append(preferred)
    for m in FREE_MODELS:
        if m not in out:
            out.append(m)
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
            + "\n\nGemini belum terhubung. Set NPL_GEMINI_API_KEY di server, atau tempel key di pengaturan Ask AI (⚙)."
        )
    return (
        "Gemini belum terhubung.\n\n"
        "Set environment `NPL_GEMINI_API_KEY` (Google AI Studio), "
        "atau klik ⚙ pada panel Ask AI lalu tempel API key.\n"
        "Model gratis yang dipakai: gemini-3.6-flash."
    )


def _to_gemini_contents(history: list[AskMessage], message: str) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for h in (history or [])[-12:]:
        role = "user" if h.role == "user" else "model"
        text = (h.content or "").strip()
        if text:
            contents.append({"role": role, "parts": [{"text": text[:4000]}]})
    contents.append({"role": "user", "parts": [{"text": message[:4000]}]})
    return contents


async def _chat_gemini(
    *,
    api_key: str,
    models: list[str],
    system: str,
    contents: list[dict[str, Any]],
) -> tuple[str, str]:
    """Returns (reply, model_used). Tries free models until one succeeds."""
    payload_base = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1200,
        },
    }
    errors: list[str] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for model in models:
            url = f"{GEMINI_BASE}/models/{model}:generateContent"
            try:
                r = await client.post(
                    url,
                    params={"key": api_key},
                    headers={"Content-Type": "application/json"},
                    json=payload_base,
                )
            except httpx.HTTPError as e:
                errors.append(f"{model}: network {e}")
                continue
            if r.status_code >= 400:
                errors.append(f"{model}: HTTP {r.status_code} {r.text[:220]}")
                continue
            data = r.json()
            try:
                parts = data["candidates"][0]["content"]["parts"]
                reply = "".join(p.get("text", "") for p in parts).strip()
            except Exception:
                # blocked / empty
                block = (data.get("promptFeedback") or {}).get("blockReason")
                errors.append(f"{model}: invalid/empty ({block or 'no candidates'})")
                continue
            if not reply:
                errors.append(f"{model}: empty reply")
                continue
            return reply, model
    raise HTTPException(
        502,
        "Gagal memanggil Gemini. Periksa API key / kuota gratis. Detail: " + " | ".join(errors)[:800],
    )


@router.post("/ask-ai", response_model=AskAIResponse)
async def ask_ai(req: AskAIRequest):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(400, "Pesan kosong")

    api_key = _resolve_api_key(req.api_key)
    models = _model_candidates(req.model)

    if not api_key:
        return AskAIResponse(
            reply=_offline_reply(message, req.context),
            model="offline-faq",
            provider="local",
            offline=True,
        )

    system = SYSTEM_PROMPT + "\n\n" + _context_block(req.context)
    contents = _to_gemini_contents(req.history or [], message)
    reply, used_model = await _chat_gemini(
        api_key=api_key, models=models, system=system, contents=contents
    )
    return AskAIResponse(
        reply=reply,
        model=used_model,
        provider="gemini",
        offline=False,
        base_url=GEMINI_BASE,
    )


@router.get("/ask-ai/status")
def ask_ai_status():
    has_key = bool(_resolve_api_key())
    return {
        "configured": has_key,
        "model": settings.gemini_model,
        "provider": "gemini" if has_key else "offline",
        "base_url": GEMINI_BASE,
        "hint": None
        if has_key
        else "Set NPL_GEMINI_API_KEY (Google AI Studio) atau tempel key di pengaturan Ask AI.",
    }


@router.post("/ask-ai/test")
async def ask_ai_test(req: AskAIRequest):
    api_key = _resolve_api_key(req.api_key)
    if not api_key:
        return {"ok": False, "offline": True, "message": "API key belum diisi"}
    models = _model_candidates(req.model)
    try:
        reply, used_model = await _chat_gemini(
            api_key=api_key,
            models=models,
            system="Jawab sangat singkat dalam Bahasa Indonesia.",
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": req.message or "Katakan: Gemini siap untuk SINAR."}],
                }
            ],
        )
        return {
            "ok": True,
            "offline": False,
            "model": used_model,
            "provider": "gemini",
            "base_url": GEMINI_BASE,
            "reply": reply[:240],
        }
    except HTTPException as e:
        return {"ok": False, "offline": False, "message": str(e.detail)}
